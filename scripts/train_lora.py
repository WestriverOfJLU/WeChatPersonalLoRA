import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def format_chat(example, tokenizer):
    messages = example["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--eval-max-samples", type=int, default=512)
    ap.add_argument("--resume-from-checkpoint")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_cls = (
        AutoModelForImageTextToText
        if config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False

    ds = load_dataset(
        "json",
        data_files={"train": args.train, "validation": args.val},
    )

    def to_parts(batch):
        prefixes = []
        answers = []
        for messages in batch["messages"]:
            full = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            answer = full[len(prefix) :] if full.startswith(prefix) else (
                messages[-1]["content"] + (tokenizer.eos_token or "")
            )
            prefixes.append(prefix)
            answers.append(answer)
        return {"prefix_text": prefixes, "answer_text": answers}

    ds = ds.map(to_parts, batched=True, remove_columns=ds["train"].column_names)

    def tokenize(batch):
        output = {"input_ids": [], "attention_mask": [], "labels": []}
        for prefix, answer in zip(batch["prefix_text"], batch["answer_text"]):
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(answer_ids) >= args.max_seq_length:
                answer_ids = answer_ids[-args.max_seq_length :]
                prefix_ids = []
            else:
                prefix_budget = args.max_seq_length - len(answer_ids)
                prefix_ids = prefix_ids[-prefix_budget:]
            input_ids = prefix_ids + answer_ids
            labels = [-100] * len(prefix_ids) + answer_ids.copy()
            output["input_ids"].append(input_ids)
            output["attention_mask"].append([1] * len(input_ids))
            output["labels"].append(labels)
        return output

    ds = ds.map(tokenize, batched=True, remove_columns=["prefix_text", "answer_text"])
    if args.eval_max_samples and len(ds["validation"]) > args.eval_max_samples:
        ds["validation"] = ds["validation"].select(range(args.eval_max_samples))

    def collate(features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        batch = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for row in labels:
            padded_labels.append(row + [-100] * (max_len - len(row)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    train_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
