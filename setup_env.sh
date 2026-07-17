#!/bin/bash
# setup_env.sh — Auto SLR 超算环境安装脚本
# 用法: bash setup_env.sh

set -e

echo "============================================"
echo "  Auto SLR — Environment Setup"
echo "============================================"

# 1. 创建 Conda 环境
echo ""
echo "[1/4] 创建 Conda 环境: auto_slr"
conda create -n auto_slr python=3.10 -y
source activate auto_slr
# 或使用: conda activate auto_slr

# 2. 安装 PyTorch（根据超算 CUDA 版本选择）
echo ""
echo "[2/4] 安装 PyTorch + CUDA"
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖
echo ""
echo "[3/4] 安装项目依赖"
pip install --upgrade pip
pip install -r requirements.txt

# 4. 配置环境变量
echo ""
echo "[4/4] 配置环境变量"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  → 已创建 .env，请编辑填入实际值"
    echo "  → 编辑 .env 设置:"
    echo "    - OPENAI_API_KEY"
    echo "    - CLEF_TAR_BASE (数据路径)"
    echo "    - BASE_MODEL_PATH (Qwen2.5 模型路径)"
else
    echo "  → .env 已存在，跳过"
fi

echo ""
echo "============================================"
echo "  ✅ 环境安装完成"
echo "============================================"
echo ""
echo "下一步:"
echo "  1. 编辑 .env 文件填入实际路径和 API Key"
echo "  2. 激活环境: conda activate auto_slr"
echo "  3. 验证安装: python check_env.py"
