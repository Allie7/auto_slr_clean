#!/usr/bin/env python
"""完整 LoRA 推理与排序评估脚本
合并 test_lora_retrieval.py 和 rank_lora_retrieval.py 的功能：
1. 运行 LoRA 推理生成 predictions.json
2. 计算 -logit("exclude") 分数并进行排序
3. 输出分类和排序评估指标

用法：
    python combined_lora_eval.py --topics CD012164
    python combined_lora_eval.py --all-topics --skip-extraction
    python combined_lora_eval.py --topics CD012164,CD005253 --dense-top-k 30
    python combined_lora_eval.py --mode ranking-only  # 仅运行排序评估
"""

import json
import os
import sys
import re
import math
import argparse
import logging
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Set
import numpy as np

# 设置环境
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_lora_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志配置
log_filename = f"combined_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, log_filename), encoding="utf-8"),
    ],
)
logger = logging.getLogger("combined_lora_eval")

# 常量 — ⚠ 请在超算上通过环境变量设置
CLEF_TAR_BASE = os.environ.get("CLEF_TAR_BASE", "")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "")
LORA_PATH = os.path.join(PROJECT_ROOT, "finetune_data", "qwen_lora_output", "final_lora")
