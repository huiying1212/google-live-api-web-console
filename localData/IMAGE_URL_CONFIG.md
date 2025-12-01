# 图片URL配置说明

## 配置完成

已成功配置图片静态文件服务，使图片能够通过HTTP URL在前端正确显示。

## 配置内容

### 1. 后端静态文件服务 (knowledge_api.py)

添加了FastAPI静态文件服务配置：

```python
from fastapi.staticfiles import StaticFiles

# 配置静态文件服务 - 提供图片访问
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "DesignBook", "images")
if os.path.exists(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
```

**图片访问路径**: `http://localhost:8000/images/{filename}`

### 2. 检索器URL转换 (search_knowledge.py)

#### 添加的配置参数
- `api_base_url`: API服务器基础URL（默认: `http://localhost:8000`）
- 可通过环境变量 `API_BASE_URL` 覆盖

#### 添加的辅助方法
```python
def _convert_image_url(self, image_filename: str) -> str:
    """将图片文件名转换为完整的HTTP URL"""
    # 如果已经是完整URL，直接返回（防止重复转换）
    if image_filename.startswith('http://') or image_filename.startswith('https://'):
        return image_filename
    return f"{self.api_base_url}/images/{image_filename}"

def _process_image_result(self, result: Dict) -> Dict:
    """处理图片搜索结果，将文件名转换为完整URL"""
    if 'image_url' in result:
        result['image_url'] = self._convert_image_url(result['image_url'])
    return result
```

**注意**: `_convert_image_url` 方法包含防重复转换检查，避免URL被多次拼接。

#### 修改的方法
所有返回图片结果的方法都已更新：
- `search_images()` - 文本搜索图片
- `search_by_image()` - 图片搜索图片
- `generate_rag_context()` - RAG上下文生成

## 数据格式示例

### 之前（仅文件名）
```json
{
  "image_url": "01020.png",
  "description": "Qin Shi Huang...",
  "source": "Chapter 1: THE EARLY ORIGINS OF DESIGN"
}
```

### 现在（完整HTTP URL）
```json
{
  "image_url": "http://localhost:8000/images/01020.png",
  "description": "Qin Shi Huang...",
  "source": "Chapter 1: THE EARLY ORIGINS OF DESIGN"
}
```

## 前端显示

前端 `Whiteboard.tsx` 组件直接使用 `image.url` 属性：

```tsx
<img src={image.url} alt={image.description} />
```

现在会正确解析为：
```html
<img src="http://localhost:8000/images/01020.png" alt="..." />
```

## 测试验证

✅ **静态文件服务测试**
```bash
curl http://localhost:8000/images/01020.png --head
# 返回: HTTP/1.1 200 OK
```

✅ **健康检查**
```bash
curl http://localhost:8000/health
# 返回: {"status":"healthy","database_info":{...}}
```

## 环境变量配置（可选）

如果需要部署到不同的环境，可以设置：

```bash
export API_BASE_URL=https://your-domain.com
```

这样图片URL会自动变为：
```
https://your-domain.com/images/01020.png
```

## 注意事项

1. **图片目录**: 确保 `public/DesignBook/images/` 目录存在且包含图片文件
2. **CORS配置**: 已配置允许 `localhost:3000` 访问
3. **服务器重启**: 修改代码后服务器会自动重新加载（uvicorn reload模式）
4. **防重复转换**: `_convert_image_url` 方法会检查URL是否已经是完整路径，避免重复拼接

## 完成状态

✅ 静态文件服务已配置  
✅ URL转换逻辑已实现  
✅ 所有搜索方法已更新  
✅ 测试验证通过  

