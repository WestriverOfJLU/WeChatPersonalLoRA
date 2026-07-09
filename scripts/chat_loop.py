import argparse
import re
import warnings
from pathlib import Path

print("正在加载 Qwen3-8B 和你的 LoRA adapter，首次启动通常需要 40-90 秒，请稍等...", flush=True)

import torch
from peft import PeftModel
from prompt_toolkit import prompt as pt_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging


warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()
logging.disable_progress_bar()


SYSTEM = (
    "你正在模拟目标用户在微信里的回复风格。"
    "你是在替他回微信，不是在写小说或表演。"
    "可聊性和正面回答优先，语气模仿放在第二位。"
    "如果被问你是谁或你叫什么，只回答“我是目标用户”，不要加其他内容。"
    "不要编造当前正在做什么、在哪里、和谁聊天、要发什么文件。"
    "如果不知道当前现实状态，就说不知道或含糊回答，不要硬编。"
    "如果对方问平时喜欢什么、爱好、习惯，要按一般个人偏好回答，不要当成当前状态。"
    "连续聊天时必须记住前文，不要突然换话题。"
    "遇到追问或质疑时，要正面纠正，不要继续圆。"
    "回复要自然、口语化、短一点；只输出回复本身。"
)


CURRENT_ACTIVITY_KEYS = (
    "在干啥",
    "在干嘛",
    "在干什么",
    "你干啥呢",
    "你干嘛呢",
    "你现在干啥",
    "你现在干嘛",
    "你现在在干啥",
    "你现在在干嘛",
)

HABIT_KEYS = ("平时", "喜欢", "爱好", "一般", "通常", "经常", "没事的时候")

CLASS_KEYS = (
    "上啥课",
    "上什么课",
    "啥课",
    "什么课",
    "下的啥课",
    "早晨晚上是啥课",
    "你刚说今天有课",
    "你不是说今天有课",
    "你上课",
    "给小孩上课",
)

STATE_CLAIMS = (
    "有课",
    "上课",
    "下课",
    "刚下课",
    "睡觉",
    "睡醒",
    "躺着",
    "宿舍",
    "看球",
    "打游戏",
    "刚回来",
    "刚到",
    "在外面",
    "发个文件",
    "发个视频",
)


def apply_backspaces(text: str) -> str:
    out = []
    for ch in text:
        if ch in ("\b", "\x7f"):
            if out:
                out.pop()
            while out and out[-1] == " ":
                out.pop()
            continue
        out.append(ch)
    return "".join(out)


def sanitize_input(text: str) -> str:
    text = str(text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = apply_backspaces(text)
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def clean(reply: str) -> str:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    reply = re.sub(r"^assistant\s*", "", reply, flags=re.I).strip()
    return reply


def is_current_activity_question(prompt: str) -> bool:
    c = compact(prompt)
    if any(key in c for key in HABIT_KEYS):
        return False
    return any(key in c for key in CURRENT_ACTIVITY_KEYS)


def contains_state_claim(text: str) -> bool:
    c = compact(text)
    return any(key in c for key in STATE_CLAIMS)


def canned_reply(prompt: str, history) -> str | None:
    c = compact(prompt)
    if any(key in c for key in ("你是谁", "你叫什么", "你叫啥")):
        return "我是目标用户"

    if is_current_activity_question(prompt):
        return "没干啥，刚看了眼消息"

    if any(key in c for key in CLASS_KEYS):
        return "我刚才说乱了，我没上课"

    if any(key in c for key in ("就是的", "是啥啊", "啥是真事", "什么真事")):
        return "我刚才说乱了，当我没说"

    if any(key in c for key in ("疯疯癫癫", "前言不搭后语", "答非所问")):
        return "我刚才说乱了，别管前面那句了"

    return None


def stabilize_reply(prompt: str, reply: str) -> str:
    c = compact(prompt)
    if any(key in c for key in HABIT_KEYS):
        return reply

    if "课" in c and any(key in c for key in ("你不是说", "你刚说", "到底", "早晨", "晚上")):
        return "我刚才说乱了，我没上课"

    if contains_state_claim(reply):
        if "无聊" in c:
            return "我也有点\n没啥事"
        if any(key in c for key in ("怎么睡得着", "睡得着觉")):
            return "睡不着也没啥用啊[捂脸]"
        if is_current_activity_question(prompt):
            return "没干啥，刚看了眼消息"
        return reply

    return reply


def render_messages(messages):
    parts = []
    for message in messages:
        role = str(message["role"])
        content = sanitize_input(str(message["content"]))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def encode_text(tokenizer, text: str, device):
    # Use the underlying tokenizers engine directly. The high-level Qwen
    # tokenizer path can throw TypeError on some console-edited Chinese input.
    text = sanitize_input(str(text))
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    try:
        encoding = tokenizer._tokenizer.encode(text)
    except Exception as exc:
        debug = {
            "type": type(text).__name__,
            "repr": repr(text),
            "codepoints": [ord(ch) for ch in text[-500:]],
            "error": repr(exc),
        }
        Path("chat_loop_tokenizer_error.txt").write_text(
            "\n".join(f"{k}: {v}" for k, v in debug.items()),
            encoding="utf-8",
        )
        raise
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def encode_chat(tokenizer, messages, device):
    attempts = [
        messages,
        messages[:1] + messages[-9:],
        messages[:1] + messages[-5:],
        messages[:1] + messages[-1:],
    ]
    last_error = None
    for attempt in attempts:
        try:
            return encode_text(tokenizer, render_messages(attempt), device)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to encode chat after fallbacks: {last_error}") from last_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="outputs/qwen3_8b_wechat_lora")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--history-turns", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--top-p", type=float, default=0.85)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
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

    history = []
    print("加载完成。直接输入聊天内容；/reset 清空上下文；/exit 退出。", flush=True)
    while True:
        raw_prompt = pt_prompt("\n你> ")
        prompt = sanitize_input(raw_prompt)
        if prompt in {"/exit", "exit", "quit"}:
            break
        if prompt == "/reset":
            history.clear()
            print("模型> 上下文清空了")
            continue
        if not prompt:
            continue

        canned = canned_reply(prompt, history)
        if canned is not None:
            print("模型>", canned)
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": canned})
            continue

        history.append({"role": "user", "content": prompt})
        kept_history = history[-args.history_turns * 2 :]
        messages = [{"role": "system", "content": SYSTEM}] + kept_history
        try:
            inputs = encode_chat(tokenizer, messages, model.device)
        except Exception as exc:
            history.pop()
            print(f"模型> 这句输入里有终端残留的非法字符，我先不处理这句。请重新打一遍。({type(exc).__name__})")
            continue
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[-1] :]
        reply = clean(tokenizer.decode(gen, skip_special_tokens=True))
        reply = stabilize_reply(prompt, reply)
        print("模型>", reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

