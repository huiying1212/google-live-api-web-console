#!/bin/bash

echo "1. 停止旧服务..."
pkill -f "python localData/knowledge_api.py"
pkill -f "serve -s build"

echo "2. 拉取新代码..."
git pull

# echo "3. 重新构建前端..."
# npm install
# npm run build

echo "4. 启动新服务..."
. venv/bin/activate
# export CLIP_MODEL_ID="./models/clip-vit-h-14"
# 设置 API 基础 URL 为服务器实际 IP，使远程客户端能正确访问图片
# 使用 HTTPS（通过 nginx 代理，端口 8443）
export API_BASE_URL="https://10.20.104.250:8443"
nohup python localData/knowledge_api.py > backend.log 2>&1 &
# nohup npx serve -s build -l 3000 > frontend.log 2>&1 &

echo "✅ 部署完成！"
echo ""
echo "访问地址: https://10.20.104.250:8443"
echo "首次访问请点击 '高级' -> '继续访问'"