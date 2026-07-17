# Auto SLR Clean — 自动化系统文献综述 Pipeline

## 概述

基于 **Qwen2.5-1.5B-Instruct** + **QLoRA** 微调，用于 CLEF-TAR 2019 系统综述（Systematic Literature Review）的论文筛选排序任务。

## Pipeline 流程

```
Step 1: 数据标注              build_finetune_data.py
Step 2: 格式转换 + think注入   convert_to_qwen_format.py
Step 3: 数据平衡（可选）       balance_dataset_with_rationale.py
Step 4: QLoRA 微调            train_qwen_lora.py
Step 5: 推理 + 排序评估        combined_lora_eval.py
```

## 环境安装

```bash
# 创建环境
conda create -n auto_slr python=3.10 -y
conda activate auto_slr

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key 和数据路径
```

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `CLEF_TAR_BASE` | CLEF-TAR 2019 数据根目录 | `/path/to/2019-TAR` |
| `BASE_MODEL_PATH` | Qwen2.5-1.5B-Instruct 路径 | `/path/to/Qwen2.5-1.5B-Instruct` |
| `OPENAI_API_KEY` | LLM API Key（build_finetune_data） | `sk-xxx` |
| `OPENAI_BASE_URL` | LLM API 地址 | `https://api.deepseek.com/` |
| `OPENAI_MODEL` | LLM 模型名 | `deepseek-v4-flash` |

## 数据目录结构

CLEF-TAR 2019 数据需按以下结构存放：
```
{CLEF_TAR_BASE}/
├── Task1/
│   ├── Training/Intervention/protocols.jsonl
│   └── Testing/Intervention/protocols.jsonl
└── Task2/
    ├── Training/Intervention/papers.jsonl
    ├── Training/Intervention/qrels/full.train.int.abs.2019.qrels
    ├── Testing/Intervention/papers.jsonl
    └── Testing/Intervention/qrels/full.test.intervention.abs.2019.qrels
```

## 使用方式

### 完整流程

```bash
# 1. 数据标注（需要 LLM API）
python build_finetune_data.py --topics CD012164,CD012551 --use-raw-text --dense-top-k 100

# 2. 格式转换
python convert_to_qwen_format.py

# 3. 数据平衡（可选）
python balance_dataset_with_rationale.py

# 4. QLoRA 训练
python train_qwen_lora.py --epochs 3 --batch_size 4 --grad_accum 4

# 5. 评估
python combined_lora_eval.py --topics CD012164,CD012551 --dense-top-k 100
```

### 分步说明

**build_finetune_data.py** — 三阶段数据标注
- `--topics` : 指定 topic ID（逗号分隔）
- `--all-topics` : 处理所有 topic
- `--use-raw-text` : 跳过 LLM 提取，直接用 title+abstract
- `--dense-top-k` : 检索保留篇数（默认 30%）
- `--skip-extraction` : 复用缓存提取结果
- `--skip-annotation` : 仅做信息提取，不做标注
- `--workers` : 并发线程数

**convert_to_qwen_format.py** — 格式转换
- `--input` : 输入文件（默认 finetune_data/finetune_data.jsonl）
- `--output` : 输出文件（默认 finetune_data/qwen_finetune.jsonl）
- `--format` : messages（Qwen格式）或 conversations（LLaMA-Factory格式）

**train_qwen_lora.py** — QLoRA 微调
- `--epochs` : 训练轮数（默认 3）
- `--batch_size` : batch size（默认 4）
- `--grad_accum` : 梯度累积（默认 4）
- `--lr` : 学习率（默认 2e-4）
- `--lora_r` : LoRA rank（默认 16）

**combined_lora_eval.py** — 推理 + 排序评估
- `--topics` : 测试 topic ID
- `--all-topics` : 测试所有 topic
- `--dense-top-k` : 检索保留篇数
- `--retrieval-method` : tfidf 或 random
- `--mode` : full（推理+排序）、inference-only、ranking-only
- `--skip-extraction` : 跳过信息提取
"# auto_slr_clean" 
