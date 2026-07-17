"""Build fine-tuning training data.

Workflow:
  Stage 0: Dense retrieval (same as train_clef_tar_dense.py), top 30% papers
  Stage 1: Multi-dimensional information extraction from title+abstract for each paper
           (Objectives, TypesofStudies, TypesofParticipants, TypesofInterventions, TypesofOutcomeMeasures)
  Stage 2: Field-level comparison between extracted info and protocols.jsonl
           2a. Blind annotation: executor annotates WITHOUT seeing gold label
           2b. Validation: check if annotation is consistent with gold label rules
           2c. Trainer-Executor loop (up to 3 times):
               - Trainer analyzes error -> provides feedback
               - Executor re-annotates with trainer's feedback
               - Validate again
           2d. Fallback: if still invalid after 3 loops, provide gold label + constraints
               directly to executor (GOLD_GUIDED mode)
           -> finetune_data.jsonl (final annotations)
           -> annotation_errors.jsonl (error analysis records)

Gold label validation rules:
  - gold=exclude: at least one field must be labeled "exclude"
  - gold=include: at least one "include", and NO field "exclude"

Usage:
    python build_finetune_data.py --topics CD005139,CD005253
    python build_finetune_data.py --topics CD005139,CD005253 --dense-method sentence_bge_base
    python build_finetune_data.py --topics CD005139,CD005253 --dense-top-k 100
    python build_finetune_data.py --all-topics
"""
import json
import os
import re
import sys
import random
import logging
import math
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── rankify / pyserini env vars ──
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import numpy as np

# ── Path setup ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "finetune_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⚠ 请在超算上修改此路径为 CLEF-TAR 2019 数据实际存放位置
CLEF_TAR_BASE = os.environ.get("CLEF_TAR_BASE", "")

# ── Logging ──
log_filename = f"build_finetune_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, log_filename), encoding="utf-8"),
    ],
)
logger = logging.getLogger("build_finetune_data")

from llm_client import chat_with_system, chat_with_system_for_json
