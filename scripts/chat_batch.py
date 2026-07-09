import argparse
import re
import warnings
from pathlib import Path

print("正在加载 Qwen3-8B 和你的 LoRA adapter...", flush=True)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging


warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()
logging.disable_progress_bar()


SYSTEM = (
    "你正在模仿目标用户在微信里的回复风格。"
    "如果被问你是谁或你叫什么，只回答“我是目标用户”，不要加其他内容。"
    "回复要自然、口语化、简短优先；只输出回复本身。"
)


def clean(reply: str) -> str:
    return re.sub(r"^<think>\s*</think>\s*", "", reply, flags=re.S).strip()


def generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    return clean(tokenizer.decode(gen, skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="outputs/qwen3_8b_wechat_lora")
    ap.add_argument("--prompts", default="scripts/sample_prompts.txt")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    prompt_path = Path(args.prompts)
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in prompts if line and not line.startswith("#")]
    if not prompts:
        raise SystemExit(f"No prompts found in {prompt_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print(generate(tokenizer, model, prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()

