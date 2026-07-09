import argparse
import re
import warnings

print("正在加载 Qwen3-8B 和你的 LoRA adapter...", flush=True)

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import logging


warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()
logging.disable_progress_bar()


SYSTEM = (
    "你正在模仿目标用户在微信里的回复风格。"
    "如果被问你是谁或你叫什么，只回答“我是目标用户”，不要加其他内容。"
    "回复要自然、口语化、简短优先；只输出回复本身。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_cls = (
        AutoModelForImageTextToText
        if config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = model_cls.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(gen, skip_special_tokens=True).strip()
    reply = re.sub(r"^<think>\s*</think>\s*", "", reply, flags=re.S).strip()
    print(reply)


if __name__ == "__main__":
    main()

