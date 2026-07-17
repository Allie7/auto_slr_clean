"""QLoRA 微调 Qwen2.5-1.5B-Instruct — 系统综述维度分类任务"""
import os, json, argparse, logging
from datetime import datetime
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ⚠ 请在超算上通过环境变量设置 BASE_MODEL_PATH，或直接修改此行
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "")
if not BASE_MODEL_PATH:
    BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct")

MODEL_PATH = BASE_MODEL_PATH
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "finetune_data", "qwen_finetune.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "finetune_data", "qwen_lora_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(OUTPUT_DIR, log_filename), encoding="utf-8")])
logger = logging.getLogger("train_qwen_lora")

def load_dataset(data_path):
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {data_path}")
    return Dataset.from_list(samples)

def preprocess_function(examples, tokenizer, max_length=2048):
    all_input_ids, all_attention_mask, all_labels = [], [], []
    for messages in examples["messages"]:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_tokens = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)
        prompt_len = len(prompt_tokens["input_ids"])
        labels = full_tokens["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        all_input_ids.append(full_tokens["input_ids"])
        all_attention_mask.append(full_tokens["attention_mask"])
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4); parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16); parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048); parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=200); parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=3); parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    logger.info("=" * 60); logger.info("QLoRA Fine-tuning: Qwen2.5-1.5B-Instruct"); logger.info("=" * 60)
    logger.info(f"Model: {MODEL_PATH}"); logger.info(f"Data: {TRAIN_DATA_PATH}")
    logger.info(f"Epochs: {args.epochs}"); logger.info(f"Batch: {args.batch_size}x{args.grad_accum}={args.batch_size*args.grad_accum}")
    logger.info(f"LR: {args.lr}"); logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}"); logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = load_dataset(TRAIN_DATA_PATH)
    dataset = raw_dataset.map(lambda x: preprocess_function(x, tokenizer, args.max_length), batched=True, remove_columns=raw_dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model); model.config.use_cache = False
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
    model = get_peft_model(model, lora_config); model.print_trainable_parameters()

    effective_bs = args.batch_size * args.grad_accum
    total_steps = (len(train_dataset) // effective_bs) * args.epochs
    logger.info(f"Estimated steps: ~{total_steps}")

    training_args = TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine", logging_steps=args.logging_steps, save_steps=args.save_steps,
        eval_strategy="steps", eval_steps=args.save_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none", seed=42,
        remove_unused_columns=False, overwrite_output_dir=True)

    from transformers import EarlyStoppingCallback
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        logger.info(f"Early stopping: patience={args.early_stopping_patience}")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator, tokenizer=tokenizer, callbacks=callbacks)
    trainer.train()

    final_path = os.path.join(args.output_dir, "final_lora")
    logger.info(f"Saving LoRA adapter to {final_path}...")
    trainer.save_model(final_path); tokenizer.save_pretrained(final_path)
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    logger.info("=" * 60); logger.info("Training complete!"); logger.info(f"LoRA adapter: {final_path}"); logger.info("=" * 60)

if __name__ == "__main__":
    main()
