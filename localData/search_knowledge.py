#!/usr/bin/env python3
"""
多模态知识检索系统
使用向量数据库进行RAG检索
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import argparse
from typing import List, Dict, Union, Tuple

class MultimodalKnowledgeRetriever:
    def __init__(self, database_dir="./vector_database", model_path="./models/clip-vit-base-patch32", device="auto"):
        """
        初始化多模态知识检索器
        
        Args:
            database_dir: 向量数据库目录
            model_path: CLIP模型路径
            device: 计算设备
        """
        self.database_dir = database_dir
        self.device = self._get_device(device)
        
        # 加载配置
        config_file = os.path.join(database_dir, "database_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 解析与加载CLIP模型（支持本地路径或Hub模型ID）
        print("加载CLIP模型...")
        resolved_model_id_or_path = self._resolve_model_id_or_path(model_path)
        try:
            # 优先使用safetensors格式以避免安全问题
            self.model = CLIPModel.from_pretrained(resolved_model_id_or_path, use_safetensors=True)
        except Exception:
            # 如果safetensors不可用，降级使用普通格式
            print("Safetensors不可用，使用普通格式...")
            self.model = CLIPModel.from_pretrained(resolved_model_id_or_path, use_safetensors=False)
        
        self.processor = CLIPProcessor.from_pretrained(resolved_model_id_or_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载FAISS索引
        print("加载向量索引...")
        self.text_index = faiss.read_index(os.path.join(database_dir, "text_index.faiss"))
        self.image_index = faiss.read_index(os.path.join(database_dir, "image_index.faiss"))
        
        # 加载元数据
        print("加载元数据...")
        with open(os.path.join(database_dir, "text_metadata.pkl"), 'rb') as f:
            self.text_metadata = pickle.load(f)
        
        with open(os.path.join(database_dir, "image_metadata.pkl"), 'rb') as f:
            self.image_metadata = pickle.load(f)
        
        print(f"数据库加载完成:")
        print(f"  文本片段: {len(self.text_metadata)}")
        print(f"  图片: {len(self.image_metadata)}")
    
    def _resolve_model_id_or_path(self, model_path: str) -> str:
        """将传入的模型路径解析为可用的本地路径或Hub模型ID。
        优先级：
          1. 环境变量 CLIP_MODEL_ID（或 HF_CLIP_MODEL_ID）
          2. 传入路径存在且包含config.json → 作为本地路径使用
          3. 回退到 Hugging Face Hub ID: "openai/clip-vit-base-patch32"
        """
        # 1) 环境变量覆盖
        env_model_id = os.getenv("CLIP_MODEL_ID") or os.getenv("HF_CLIP_MODEL_ID")
        if env_model_id:
            print(f"使用环境变量模型ID: {env_model_id}")
            return env_model_id
        
        # 2) 本地路径存在且看起来像有效模型目录
        if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json")):
            print(f"使用本地模型路径: {model_path}")
            return model_path
        
        # 3) 如果传入的参数本身就是一个Hub ID，也直接返回
        # 简单判断：包含'/'基本可视为命名空间/仓库名
        if "/" in model_path and not model_path.startswith("./") and not model_path.startswith(".\\"):
            print(f"使用指定的Hub模型ID: {model_path}")
            return model_path
        
        # 4) 最终回退：默认Hub模型
        default_id = "openai/clip-vit-base-patch32"
        print(f"未找到本地模型，回退到Hub模型: {default_id}")
        return default_id
    
    def _get_device(self, device):
        """确定计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def encode_query_text(self, query: str) -> np.ndarray:
        """
        编码查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            查询向量
        """
        with torch.no_grad():
            inputs = self.processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            text_embedding = self.model.get_text_features(**inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
            return text_embedding.cpu().numpy()
    
    def encode_query_image(self, image_path: str) -> np.ndarray:
        """
        编码查询图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            查询向量
        """
        image = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            inputs = self.processor(
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            image_embedding = self.model.get_image_features(**inputs)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            return image_embedding.cpu().numpy()
    
    def search_text(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """
        在文本数据库中搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            min_score: 最小相似度阈值
            
        Returns:
            搜索结果列表
        """
        query_vector = self.encode_query_text(query)
        
        # 在文本索引中搜索
        scores, indices = self.text_index.search(query_vector.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score and idx < len(self.text_metadata):
                result = self.text_metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def search_images(self, query: str, top_k: int = 5, min_score: float = 0.2) -> List[Dict]:
        """
        在图片数据库中搜索（使用文本查询）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            min_score: 最小相似度阈值
            
        Returns:
            搜索结果列表
        """
        query_vector = self.encode_query_text(query)
        
        # 在图片索引中搜索
        scores, indices = self.image_index.search(query_vector.astype('float32'), top_k)
        
        print(f"图片搜索 - 查询: {query}")
        print(f"得分: {scores[0][:3]}")  # 显示前3个得分
        print(f"索引: {indices[0][:3]}")  # 显示前3个索引
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_metadata):  # 移除最小分数限制进行调试
                result = self.image_metadata[idx].copy()
                result['similarity_score'] = float(score)
                if score >= min_score:  # 只在达到阈值时才添加
                    results.append(result)
                print(f"图片 {idx}: 得分={score:.3f}, 阈值={min_score}, 包含={'是' if score >= min_score else '否'}")
        
        print(f"图片搜索结果数量: {len(results)}")
        return results
    
    def search_by_image(self, image_path: str, top_k: int = 5, min_score: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """
        使用图片查询（图片到图片，图片到文本）
        
        Args:
            image_path: 查询图片路径
            top_k: 返回结果数量
            min_score: 最小相似度阈值
            
        Returns:
            (相似图片列表, 相关文本列表)
        """
        query_vector = self.encode_query_image(image_path)
        
        # 搜索相似图片
        image_scores, image_indices = self.image_index.search(query_vector.astype('float32'), top_k)
        image_results = []
        for score, idx in zip(image_scores[0], image_indices[0]):
            if score >= min_score and idx < len(self.image_metadata):
                result = self.image_metadata[idx].copy()
                result['similarity_score'] = float(score)
                image_results.append(result)
        
        # 搜索相关文本
        text_scores, text_indices = self.text_index.search(query_vector.astype('float32'), top_k)
        text_results = []
        for score, idx in zip(text_scores[0], text_indices[0]):
            if score >= min_score and idx < len(self.text_metadata):
                result = self.text_metadata[idx].copy()
                result['similarity_score'] = float(score)
                text_results.append(result)
        
        return image_results, text_results
    
    def multimodal_search(self, query: str, top_k: int = 10, text_weight: float = 0.5, image_weight: float = 0.5) -> Dict:
        """
        多模态综合搜索
        
        Args:
            query: 查询文本
            top_k: 每种模态返回的结果数量
            text_weight: 文本权重
            image_weight: 图片权重
            
        Returns:
            综合搜索结果
        """
        # 搜索文本和图片
        text_results = self.search_text(query, top_k)
        image_results = self.search_images(query, top_k)
        
        # 归一化得分 - 分别计算每种模态的最高分数
        max_text_score = max([r.get('similarity_score', 0.0) for r in text_results]) if text_results else 1.0
        max_image_score = max([r.get('similarity_score', 0.0) for r in image_results]) if image_results else 1.0
        
        # 避免除零错误
        max_text_score = max(max_text_score, 0.001)
        max_image_score = max(max_image_score, 0.001)
        
        # 计算标准化权重分数
        for result in text_results:
            normalized_score = result.get('similarity_score', 0.0) / max_text_score
            result['weighted_score'] = normalized_score * text_weight
            
        for result in image_results:
            normalized_score = result.get('similarity_score', 0.0) / max_image_score  
            result['weighted_score'] = normalized_score * image_weight
        
        # 组合结果，确保包含两种类型
        all_results = text_results + image_results
        try:
            all_results.sort(key=lambda x: x.get('weighted_score', 0.0), reverse=True)
        except Exception as e:
            print(f"排序错误: {e}")
        
        # 确保结果中包含两种类型 - 取前top_k个，但至少包含一些图片结果
        combined_results = []
        text_count = 0
        image_count = 0
        
        # 先添加高分结果
        for result in all_results:
            if len(combined_results) >= top_k:
                break
            if result.get('type') == 'text':
                text_count += 1
            else:
                image_count += 1
            combined_results.append(result)
        
        # 如果没有图片结果但有图片数据，至少添加一个图片结果
        if image_count == 0 and len(image_results) > 0 and len(combined_results) < top_k:
            # 移除最后一个文本结果，添加最好的图片结果
            if text_count > 0:
                combined_results = combined_results[:-1]
            combined_results.append(image_results[0])
        
        return {
            'query': query,
            'text_results': text_results,
            'image_results': image_results,
            'combined_results': combined_results,
            'total_results': len(all_results)
        }
    
    def generate_rag_context(self, query: str, max_context_length: int = 2000) -> Dict:
        """
        为RAG生成上下文信息
        
        Args:
            query: 用户查询
            max_context_length: 最大上下文长度
            
        Returns:
            RAG上下文信息
        """
        search_results = self.multimodal_search(query, top_k=10, text_weight=0.5, image_weight=0.5)
        
        # 提取文本上下文
        text_context = []
        current_length = 0
        
        for result in search_results['combined_results']:
            if result['type'] == 'text':
                text_content = result['text']
                if current_length + len(text_content) <= max_context_length:
                    text_context.append({
                        'content': text_content,
                        'source': f"Chapter {result['chapter_number']}: {result['chapter_name']}",
                        'similarity': result['weighted_score']
                    })
                    current_length += len(text_content)
                else:
                    break
        
        # 提取图片信息
        image_context = []
        for result in search_results['combined_results']:
            if result['type'] == 'image':
                image_context.append({
                    'image_url': result['image_url'],
                    'description': result['image_description'],
                    'source': f"Chapter {result['chapter_number']}: {result['chapter_name']}",
                    'similarity': result['weighted_score']
                })
        
        return {
            'query': query,
            'text_context': text_context,
            'image_context': image_context[:5],  # 最多5张相关图片
            'context_stats': {
                'text_chunks': len(text_context),
                'total_text_length': current_length,
                'related_images': len(image_context)
            }
        }

def main():
    parser = argparse.ArgumentParser(description="多模态知识检索")
    parser.add_argument("--database_dir", default="./vector_database", 
                      help="向量数据库目录")
    parser.add_argument("--model_path", default="./models/clip-vit-base-patch32", 
                      help="CLIP模型路径")
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--output", help="输出结果到JSON文件")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                      help="计算设备")
    parser.add_argument("--mode", default="multimodal", 
                      choices=["text", "image", "multimodal", "rag"],
                      help="搜索模式")
    
    args = parser.parse_args()
    
    # 初始化检索器
    retriever = MultimodalKnowledgeRetriever(
        database_dir=args.database_dir,
        model_path=args.model_path,
        device=args.device
    )
    
    # 执行搜索
    if args.mode == "text":
        results = retriever.search_text(args.query, args.top_k)
    elif args.mode == "image":
        results = retriever.search_images(args.query, args.top_k)
    elif args.mode == "multimodal":
        results = retriever.multimodal_search(args.query, args.top_k)
    elif args.mode == "rag":
        results = retriever.generate_rag_context(args.query)
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {args.output}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 