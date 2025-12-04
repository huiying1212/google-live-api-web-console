#!/usr/bin/env python3
"""
多模态向量数据库构建器
使用本地CLIP模型为文字和图片数据创建向量嵌入
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
from tqdm import tqdm
import pickle
import argparse
from pathlib import Path

class MultimodalVectorDatabase:
    def __init__(self, model_path="./models/clip-vit-h-14", device="auto"):
        """
        初始化多模态向量数据库
        
        Args:
            model_path: 本地CLIP模型路径
            device: 计算设备 ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        print(f"使用设备: {self.device}")
        
        # 加载本地CLIP模型
        print("=" * 60)
        print("正在加载嵌入模型...")
        print(f"模型路径: {model_path}")
        try:
            # 优先使用safetensors格式以避免安全问题
            self.model = CLIPModel.from_pretrained(model_path, use_safetensors=True)
            print("✓ 使用 safetensors 格式加载")
        except:
            # 如果safetensors不可用，降级使用普通格式
            print("Safetensors不可用，使用普通格式...")
            self.model = CLIPModel.from_pretrained(model_path, use_safetensors=False)
        
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 获取嵌入维度和模型信息
        self.embedding_dim = self.model.config.projection_dim
        model_type = self.model.config.model_type if hasattr(self.model.config, 'model_type') else "CLIP"
        
        # 显示模型详细信息
        print(f"✓ 模型加载成功")
        print(f"模型类型: {model_type.upper()}")
        print(f"模型架构: {self.model.config.architectures[0] if hasattr(self.model.config, 'architectures') else 'CLIP'}")
        print(f"嵌入维度: {self.embedding_dim}")
        print(f"视觉编码器: {self.model.config.vision_config.model_type if hasattr(self.model.config, 'vision_config') else 'ViT'}")
        print(f"文本编码器: {self.model.config.text_config.model_type if hasattr(self.model.config, 'text_config') else 'Transformer'}")
        print("=" * 60)
        
        # 初始化FAISS索引
        self.text_index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
        self.image_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 存储元数据
        self.text_metadata = []
        self.image_metadata = []
    
    def _get_device(self, device):
        """确定计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def encode_text(self, texts, batch_size=32):
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            numpy.ndarray: 文本嵌入向量
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
                batch_texts = texts[i:i + batch_size]
                
                # 预处理文本
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                # 获取文本嵌入
                text_embeddings = self.model.get_text_features(**inputs)
                
                # 归一化
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings.append(text_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_images(self, image_paths, batch_size=16):
        """
        编码图片为向量
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小
            
        Returns:
            numpy.ndarray: 图片嵌入向量
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图片"):
                batch_paths = image_paths[i:i + batch_size]
                
                # 加载和预处理图片
                images = []
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert('RGB')
                        images.append(image)
                    except Exception as e:
                        print(f"无法加载图片 {path}: {e}")
                        # 创建一个空白图片作为占位符
                        images.append(Image.new('RGB', (224, 224), color='white'))
                
                if not images:
                    continue
                
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # 获取图片嵌入
                image_embeddings = self.model.get_image_features(**inputs)
                
                # 归一化
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_embeddings.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def build_text_database(self, content_file):
        """
        构建文本向量数据库
        
        Args:
            content_file: 文本内容JSON文件路径
        """
        print("\n" + "=" * 60)
        print("构建文本向量数据库...")
        print(f"使用嵌入模型进行文本编码: {self.model_path}")
        print("=" * 60)
        
        # 读取文本数据（支持 JSONL 格式）
        content_data = []
        with open(content_file, 'r', encoding='utf-8') as f:
            # 尝试读取第一行来判断是 JSON 还是 JSONL
            first_line = f.readline()
            f.seek(0)
            
            if first_line.strip().startswith('['):
                # JSON 数组格式
                content_data = json.load(f)
            else:
                # JSONL 格式（每行一个 JSON 对象）
                for line in f:
                    if line.strip():
                        content_data.append(json.loads(line))
        
        texts = []
        metadata = []
        
        for item in content_data:
            # 检查数据格式：如果有 'text' 字段，说明是已分块的数据（JSONL格式）
            # 如果有 'chapter_test' 字段，说明是原始未分块的数据（JSON格式）
            if 'text' in item:
                # 已分块的 JSONL 格式数据，直接使用
                texts.append(item['text'])
                metadata.append({
                    'chapter_number': item.get('chapter_number', 0),
                    'chapter_name': item.get('chapter_name', ''),
                    'chunk_id': item.get('chunk_id', 0),
                    'text': item['text'],
                    'type': 'text',
                    'id': item.get('id', '')
                })
            elif 'chapter_test' in item:
                # 原始未分块的 JSON 格式数据，需要分块
                chapter_text = item['chapter_test']
                chapter_name = item['chapter_name']
                chapter_number = item['chapter_number']
                
                # 按句子分割文本，每个片段约500字符
                sentences = chapter_text.split('. ')
                current_chunk = ""
                chunk_id = 0
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 500:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk.strip():
                            texts.append(current_chunk.strip())
                            metadata.append({
                                'chapter_number': chapter_number,
                                'chapter_name': chapter_name,
                                'chunk_id': chunk_id,
                                'text': current_chunk.strip(),
                                'type': 'text'
                            })
                            chunk_id += 1
                        current_chunk = sentence + '. '
                
                # 添加最后一个片段
                if current_chunk.strip():
                    texts.append(current_chunk.strip())
                    metadata.append({
                        'chapter_number': chapter_number,
                        'chapter_name': chapter_name,
                        'chunk_id': chunk_id,
                        'text': current_chunk.strip(),
                        'type': 'text'
                    })
        
        # 编码文本
        embeddings = self.encode_text(texts)
        
        # 添加到FAISS索引
        self.text_index.add(embeddings.astype('float32'))
        self.text_metadata.extend(metadata)
        
        print(f"添加了 {len(texts)} 个文本片段到数据库")
    
    def build_image_database(self, image_file, image_dir):
        """
        构建图片向量数据库
        
        Args:
            image_file: 图片元数据JSON文件路径
            image_dir: 图片目录路径
        """
        print("\n" + "=" * 60)
        print("构建图片向量数据库...")
        print(f"使用嵌入模型进行图片编码: {self.model_path}")
        print("=" * 60)
        
        # 读取图片元数据
        with open(image_file, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
        
        image_paths = []
        metadata = []
        
        for item in image_data:
            # 兼容不同的字段名：image_url 或 path
            image_filename = item.get('image_url') or item.get('path', '')
            image_path = os.path.join(image_dir, image_filename)
            if os.path.exists(image_path):
                image_paths.append(image_path)
                metadata.append({
                    'chapter_number': item['chapter_number'],
                    'chapter_name': item['chapter_name'],
                    'image_id': item.get('image_ID', ''),
                    'image_url': image_filename,
                    'image_description': item.get('image_description') or item.get('caption_text', ''),
                    'image_path': image_path,
                    'type': 'image',
                    'tags': item.get('tags', []),
                    'image_detail': item.get('image_detail', '')
                })
            else:
                print(f"图片文件不存在: {image_path}")
        
        if not image_paths:
            print("未找到有效的图片文件")
            return
        
        # 编码图片
        embeddings = self.encode_images(image_paths)
        
        if embeddings.size > 0:
            # 添加到FAISS索引
            self.image_index.add(embeddings.astype('float32'))
            self.image_metadata.extend(metadata)
            
            print(f"添加了 {len(image_paths)} 张图片到数据库")
        else:
            print("图片编码失败")
    
    def save_database(self, output_dir):
        """
        保存向量数据库
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("保存向量数据库...")
        
        # 保存FAISS索引
        faiss.write_index(self.text_index, os.path.join(output_dir, "text_index.faiss"))
        faiss.write_index(self.image_index, os.path.join(output_dir, "image_index.faiss"))
        
        # 保存元数据
        with open(os.path.join(output_dir, "text_metadata.pkl"), 'wb') as f:
            pickle.dump(self.text_metadata, f)
        
        with open(os.path.join(output_dir, "image_metadata.pkl"), 'wb') as f:
            pickle.dump(self.image_metadata, f)
        
        # 保存配置信息
        model_type = self.model.config.model_type if hasattr(self.model.config, 'model_type') else "CLIP"
        model_architecture = self.model.config.architectures[0] if hasattr(self.model.config, 'architectures') else 'CLIPModel'
        
        config = {
            'embedding_dim': self.embedding_dim,
            'text_count': len(self.text_metadata),
            'image_count': len(self.image_metadata),
            'model_path': self.model_path,
            'model_type': model_type,
            'model_architecture': model_architecture
        }
        
        with open(os.path.join(output_dir, "database_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("=" * 60)
        print(f"✓ 向量数据库已保存到: {output_dir}")
        print(f"使用的嵌入模型: {model_architecture}")
        print(f"模型路径: {self.model_path}")
        print(f"嵌入维度: {self.embedding_dim}")
        print(f"文本片段数量: {len(self.text_metadata)}")
        print(f"图片数量: {len(self.image_metadata)}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="构建多模态向量数据库")
    parser.add_argument("--data_dir", default="./public/DesignBook", 
                      help="知识库数据目录")
    parser.add_argument("--output_dir", default="./vector_database", 
                      help="输出目录")
    parser.add_argument("--model_path", default="./models/clip-vit-h-14", 
                      help="CLIP模型路径")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                      help="计算设备")
    
    args = parser.parse_args()
    
    # 检查输入文件
    content_file = os.path.join(args.data_dir, "content2.jsonl")
    image_file = os.path.join(args.data_dir, "image2.json")
    image_dir = os.path.join(args.data_dir, "images")
    
    if not os.path.exists(content_file):
        print(f"错误: 文本文件不存在: {content_file}")
        return
    
    if not os.path.exists(image_file):
        print(f"错误: 图片元数据文件不存在: {image_file}")
        return
    
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录不存在: {image_dir}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: CLIP模型目录不存在: {args.model_path}")
        return
    
    print("\n" + "=" * 60)
    print("开始构建多模态向量数据库")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"嵌入模型: {args.model_path}")
    print(f"计算设备: {args.device}")
    print("=" * 60 + "\n")
    
    # 构建向量数据库
    db = MultimodalVectorDatabase(args.model_path, args.device)
    
    # 构建文本数据库
    db.build_text_database(content_file)
    
    # 构建图片数据库
    db.build_image_database(image_file, image_dir)
    
    # 保存数据库
    db.save_database(args.output_dir)
    
    print("\n" + "=" * 60)
    print("✓ 多模态向量数据库构建完成!")
    print("=" * 60)

if __name__ == "__main__":
    main() 