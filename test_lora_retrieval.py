"""完整论文检索系统流程 — 基于 QLoRA 微调模型测试

五阶段流程：
  Stage 0: 稠密检索 — 从测试 topic 的所有候选论文中筛选 top-k 篇
  Stage 1: 多维度信息提取 — LLM 从 title+abstract 提取 5 个维度信息
  Stage 2: 构造 Qwen 推理格式 — 将提取信息转成 messages 格式
  Stage 3: QLoRA 模型推理 — 对每个 (paper, dimension) 三元组预测 label
  Stage 4: 论文级决策 + 统计 metrics

用法：
    python test_lora_retrieval.py
    python test_lora_retrieval.py --topics CD005139,CD005253
    python test_lora_retrieval.py --dense-top-k 100 --workers 4
    python test_lora_retrieval.py --skip-extraction  # 跳过Stage1，用缓存
"""

import json
import os
import re
import sys
import math
import argparse
import logging
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import numpy as np

# ── 路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_lora_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⚠ 请在超算上修改以下路径
CLEF_TAR_BASE = os.environ.get("CLEF_TAR_BASE", "")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "")
LORA_PATH = os.path.join(PROJECT_ROOT, "finetune_data", "qwen_lora_output", "final_lora")

# ── 日志 ──
log_filename = f"test_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, log_filename), encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_lora_retrieval")

# ── 导入 ──
from llm_client import chat_with_system_for_json, chat_with_system
