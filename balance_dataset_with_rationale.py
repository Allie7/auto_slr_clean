#!/usr/bin/env python3
"""
平衡包含思考内容的数据集，使三个标签的数量平衡
"""
import json
import random
import os

def extract_label_from_content(content):
    """从assistant内容中提取标签（包含思考内容）"""
    if '</think>' in content:
        parts = content.split('</think>')
        if len(parts) == 3:
            return parts[2].strip()
    content_lower = content.lower()
    if content_lower.endswith('include'):
        return 'include'
    elif content_lower.endswith('exclude'):
        return 'exclude'
    elif content_lower.endswith('not_sure'):
        return 'not_sure'
    else:
        return 'unknown'

def balance_dataset_with_rationale(input_path, output_path, seed=42, balance_strategy='downsample'):
    random.seed(seed)
    include_samples = []
    exclude_samples = []
    not_sure_samples = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data['messages'][-1]['role'] == 'assistant':
                    content = data['messages'][-1]['content']
                    label = extract_label_from_content(content)
                    if label == 'include':
                        include_samples.append(data)
                    elif label == 'exclude':
                        exclude_samples.append(data)
                    elif label == 'not_sure':
                        not_sure_samples.append(data)
                    else:
                        print(f"警告：无法识别的标签，内容: {content[:50]}...")
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                continue

    print("原始数据分布:")
    print(f"  Include: {len(include_samples)}")
    print(f"  Exclude: {len(exclude_samples)}")
    print(f"  Not sure: {len(not_sure_samples)}")
    total_original = len(include_samples) + len(exclude_samples) + len(not_sure_samples)
    print(f"  总计: {total_original}")

    if balance_strategy == 'downsample':
        target_count = min(len(include_samples), len(exclude_samples), len(not_sure_samples))
        print(f"\n使用下采样策略，目标每个类别: {target_count} 个样本")
        if len(include_samples) > target_count:
            include_samples = random.sample(include_samples, target_count)
        if len(exclude_samples) > target_count:
            exclude_samples = random.sample(exclude_samples, target_count)
        if len(not_sure_samples) > target_count:
            not_sure_samples = random.sample(not_sure_samples, target_count)
    elif balance_strategy == 'upsample':
        target_count = max(len(include_samples), len(exclude_samples), len(not_sure_samples))
        print(f"\n使用上采样策略，目标每个类别: {target_count} 个样本")
        if len(include_samples) < target_count:
            include_samples = include_samples + random.choices(include_samples, k=target_count - len(include_samples))
        if len(exclude_samples) < target_count:
            exclude_samples = exclude_samples + random.choices(exclude_samples, k=target_count - len(exclude_samples))
        if len(not_sure_samples) < target_count:
            not_sure_samples = not_sure_samples + random.choices(not_sure_samples, k=target_count - len(not_sure_samples))
    else:
        target_count = min(len(include_samples), len(exclude_samples), len(not_sure_samples))
        print(f"\n使用默认平衡策略，目标每个类别: {target_count} 个样本")
        if len(include_samples) > target_count:
            include_samples = random.sample(include_samples, target_count)
        if len(exclude_samples) > target_count:
            exclude_samples = random.sample(exclude_samples, target_count)
        if len(not_sure_samples) > target_count:
            not_sure_samples = random.sample(not_sure_samples, target_count)

    print(f"\n平衡后数据分布:")
    print(f"  Include: {len(include_samples)}")
    print(f"  Exclude: {len(exclude_samples)}")
    print(f"  Not sure: {len(not_sure_samples)}")
    total_balanced = len(include_samples) + len(exclude_samples) + len(not_sure_samples)
    print(f"  总计: {total_balanced}")

    balanced_samples = include_samples + exclude_samples + not_sure_samples
    random.shuffle(balanced_samples)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n已保存到: {output_path}")

if __name__ == "__main__":
    input_file = "finetune_data/qwen_finetune_with_rationale.jsonl"
    output_file = "finetune_data/qwen_finetune_with_rationale_balanced.jsonl"

    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        print(f"请先运行 convert_to_qwen_format.py 生成包含思考内容的训练数据")
        exit(1)

    balance_dataset_with_rationale(input_file, output_file, balance_strategy='downsample')
