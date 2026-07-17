"""
将 finetune_data.jsonl 拆分为逐维度的 Qwen2.5-1.5B-Instruct 微调格式。

每条原始记录包含 topic_id, pmid, gold_label, 以及5个维度（Objectives, TypesofStudies,
TypesofParticipants, TypesofInterventions, TypesofOutcomeMeasures），每个维度含 query/paper/label/rationale。

转换后：每个 (record, dimension) 三元组 -> 一条 messages 格式训练样本，
兼容 Qwen2.5 tokenizer.apply_chat_template()。

输出格式（两种可选）：
  --format messages    : {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
  --format conversations: {"conversations": [{"from":"system",...}, {"from":"human",...}, {"from":"gpt",...}]}

用法：
    python convert_to_qwen_format.py
    python convert_to_qwen_format.py --format messages --output finetune_data/qwen_finetune.jsonl
    python convert_to_qwen_format.py --format conversations --output finetune_data/qwen_finetune_conversations.jsonl
"""

import json
import os
import argparse
import random

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 五个维度
PROTOCOL_DIMENSIONS = [
    "Objectives",
    "TypesofStudies",
