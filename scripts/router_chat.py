import argparse
import re
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

print("正在加载 Qwen3-8B、基座能力和你的 LoRA 语气 adapter，请稍等...", flush=True)

import torch
from peft import PeftModel
from prompt_toolkit import prompt as pt_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging


warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()
logging.disable_progress_bar()


BASE_SYSTEM = (
    "你是一个严谨、实用的中文助手。"
    "回答事实、学校、专业、老师、技术、写作等问题时，优先保证准确、完整、可执行。"
    "不知道的事实必须明确说不确定，不要编造人名、学校信息、老师方向、关系、经历。"
    "如果问题需要实时信息或具体老师资料，而你没有资料，就给出核查思路和需要确认的信息。"
    "语气可以自然口语一点，但不要牺牲准确性。"
)

STYLE_SYSTEM = (
    "你正在模拟目标用户在微信里的回复风格。"
    "你的任务只是在不改变事实的前提下，把可靠内容改写成目标用户的微信口吻。"
    "必须保留原答案事实，不要新增事实、人名、学校、老师、机构、经历、排名、强弱判断。"
    "不确定的信息仍然必须保留“不确定”“需要核查”的意思。"
    "如果可靠内容里没有某个具体名字或结论，你绝对不能补出来。"
    "可聊性和正面回答优先，语气模仿放在第二位。"
    "自然、口语化、短一点，但不能把有用信息删光。"
    "只输出最终回复本身。"
)

NORMAL_SYSTEM = (
    "你正在模拟目标用户在微信里的回复风格。"
    "可聊性和正面回答优先，语气模仿放在第二位。"
    "如果被问你是谁或你叫什么，只回答“我是目标用户”。"
    "不要编造当前正在做什么、在哪里、和谁聊天、要发什么文件。"
    "不知道当前现实状态就说不知道。"
    "连续聊天时要记住前文，不要突然换话题。"
    "回复要自然、口语化、短一点。"
)

FACTS_PATH = Path("data/personal_facts.md")

ROUTE_FACT = "fact"
ROUTE_WRITE = "write"
ROUTE_REWRITE = "rewrite"
ROUTE_NORMAL = "normal"


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
    reply = re.sub(r"^<think>.*", "", reply, flags=re.S).strip()
    reply = re.sub(r"^assistant\s*", "", reply, flags=re.I).strip()
    return reply


def entity_like_terms(text: str) -> set[str]:
    terms = set()
    patterns = [
        r"[\u4e00-\u9fff]{2,8}(?:大学|学院|研究所|实验室|课题组|公司|机构|中心)",
        r"[\u4e00-\u9fff]{2,4}(?:老师|教授|导师)",
        r"(?:[A-Z][A-Za-z0-9_-]{1,12}|[a-z]{2,12})(?:\s*[A-Z][A-Za-z0-9_-]{1,12})?",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = "".join(match)
            terms.add(match.strip())
    for exact in ("国科大", "中科院", "微所", "集院", "吉大", "山大", "西电", "哈工大", "DFT", "EDA"):
        if exact in text:
            terms.add(exact)
    return {term for term in terms if term and term not in {"LoRA", "Qwen"}}


def content_terms(text: str) -> set[str]:
    terms = set()
    for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", text):
        if match in {
            "这个",
            "那个",
            "如果",
            "因为",
            "所以",
            "但是",
            "建议",
            "可以",
            "需要",
            "比较",
            "具体",
            "不确定",
        }:
            continue
        terms.add(match)
    return terms


def has_unlicensed_specifics(base_answer: str, styled_answer: str) -> tuple[bool, str]:
    base_terms = entity_like_terms(base_answer)
    styled_terms = entity_like_terms(styled_answer)
    new_terms = styled_terms - base_terms
    if new_terms:
        return True, "新增实体：" + "、".join(sorted(new_terms)[:6])

    risky_patterns = (
        "一定",
        "肯定",
        "绝对",
        "最强",
        "最好",
        "比.*强",
        "不如",
        "稳",
        "必",
    )
    base_compact = compact(base_answer)
    styled_compact = compact(styled_answer)
    for pattern in risky_patterns:
        if re.search(pattern, styled_compact) and not re.search(pattern, base_compact):
            return True, "新增过度确定判断"

    uncertainty = ("不确定", "不清楚", "没有足够", "需要核查", "建议核实", "不了解")
    if any(word in base_answer for word in uncertainty) and not any(word in styled_answer for word in uncertainty):
        return True, "删掉了不确定性"

    base_terms = content_terms(base_answer)
    styled_terms = content_terms(styled_answer)
    if len(base_answer) > 120 and len(styled_answer) < len(base_answer) * 0.38:
        return True, "删掉了太多信息"
    if len(base_terms) >= 8:
        kept = len(base_terms & styled_terms) / max(len(base_terms), 1)
        if kept < 0.25:
            return True, "关键内容保留太少"

    return False, ""


def read_facts() -> str:
    if not FACTS_PATH.exists():
        FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FACTS_PATH.write_text(
            "# 目标用户个人事实\n\n"
            "- 姓名：目标用户\n"
            "- 本科：吉林大学\n"
            "- 研究生：中国科学院大学\n"
            "- 专业相关：电子科学与技术、集成电路方向\n\n"
            "# 使用说明\n"
            "把确定无误的个人事实写在这里。模型只能引用这里的事实，不应该临场编造。\n",
            encoding="utf-8",
        )
    return FACTS_PATH.read_text(encoding="utf-8")


def render_messages(messages):
    parts = []
    for message in messages:
        role = str(message["role"])
        content = sanitize_input(str(message["content"]))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def encode_text(tokenizer, text: str, device):
    text = sanitize_input(str(text))
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    try:
        encoding = tokenizer._tokenizer.encode(text)
    except Exception as exc:
        Path("router_chat_tokenizer_error.txt").write_text(
            f"type: {type(text).__name__}\nrepr: {repr(text)}\n"
            f"codepoints: {[ord(ch) for ch in text[-500:]]}\nerror: {repr(exc)}",
            encoding="utf-8",
        )
        raise
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def generate(model, tokenizer, messages, *, max_new_tokens, temperature, top_p, adapter=True):
    text = render_messages(messages)
    inputs = encode_text(tokenizer, text, model.device)
    ctx = nullcontext() if adapter else model.disable_adapter()
    with ctx:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
            )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    return clean(tokenizer.decode(gen, skip_special_tokens=True))


def classify(prompt: str, mode: str) -> str:
    if mode in {ROUTE_FACT, ROUTE_WRITE, ROUTE_REWRITE, ROUTE_NORMAL}:
        return mode

    c = compact(prompt).lower()
    if any(k in c for k in ("改写", "润色", "换成我的语气", "微信语气")):
        return ROUTE_REWRITE
    if any(k in c for k in ("写篇", "写一篇", "作文", "文章", "报告", "总结", "邮件", "申请", "文案")):
        return ROUTE_WRITE
    if any(
        k in c
        for k in (
            "老师",
            "学校",
            "专业",
            "方向",
            "国科大",
            "微所",
            "集院",
            "吉大",
            "山大",
            "西电",
            "哈工大",
            "考研",
            "复试",
            "导师",
            "dft",
            "可测性",
            "eda",
            "芯片",
            "集成电路",
            "推荐",
            "哪个好",
            "怎么样",
            "介绍一下",
        )
    ):
        return ROUTE_FACT
    return ROUTE_NORMAL


def recent_history(history, n=8):
    return history[-n * 2 :]


def answer_base(model, tokenizer, prompt: str, history, route: str, facts: str) -> str:
    if route == ROUTE_WRITE:
        system = (
            BASE_SYSTEM
            + "用户要求写作时，直接完成正文；如果用户说继续，就延续上文，不要只答应。"
        )
    else:
        system = (
            BASE_SYSTEM
            + "回答咨询时像微信里正常聊天一样简洁，但必须把关键信息说清楚。"
            + "不要写长篇报告，不要用 Markdown 大标题；用 4 到 7 行短句。"
            + "比较两个选择时，给出条件判断，不要强行下绝对结论。"
        )

    context = ""
    if history:
        context = "\n\n最近对话：\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in recent_history(history, 5)
        )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"个人事实资料：\n{facts}\n"
                f"{context}\n\n"
                f"用户问题：{prompt}\n\n"
                "请给出可靠、有用、简洁的回答。若没有可靠资料，明确说不确定。"
            ),
        },
    ]
    max_tokens = 360 if route == ROUTE_WRITE else 180
    return generate(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_tokens,
        temperature=0.25,
        top_p=0.8,
        adapter=False,
    )


def read_line() -> str:
    if sys.stdin.isatty():
        return pt_prompt("\n你> ")
    print("\n你> ", end="", flush=True)
    return input()


def style_rewrite(model, tokenizer, prompt: str, base_answer: str, history, max_new_tokens=260, strict=False) -> str:
    extra = ""
    if strict:
        extra = (
            "注意：上一版改写丢了太多信息。请保留原文至少八成关键内容，"
            "可以口语化，但不要缩成一句空话。\n\n"
        )
    messages = [
        {"role": "system", "content": STYLE_SYSTEM},
        {
            "role": "user",
            "content": (
                extra +
                "下面是可靠内容，请改写成目标用户微信聊天时会发出的口吻。"
                "只允许改表达方式，不允许新增信息。"
                "如果原文说不确定，你也必须说不确定。"
                "如果原文没有具体名字，不要补名字。"
                "不要把原文没有的比较、排名、强弱结论写出来。\n\n"
                f"对方刚问：{prompt}\n\n"
                f"可靠内容：\n{base_answer}"
            ),
        },
    ]
    return generate(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.28 if strict else 0.35,
        top_p=0.78 if strict else 0.82,
        adapter=True,
    )


def light_wechat_style(base_answer: str) -> str:
    text = clean(base_answer)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("首先，", "先说结论，")
    text = text.replace("建议", "我建议")
    if len(text) > 500:
        text = text[:500].rstrip() + "\n后面可以再按具体方向细聊"
    return "这个真得看具体方向，不能一概而论。\n" + text


def guarded_style_rewrite(model, tokenizer, prompt: str, base_answer: str, history, max_new_tokens=220) -> str:
    styled = style_rewrite(model, tokenizer, prompt, base_answer, history, max_new_tokens=max_new_tokens)
    bad, reason = has_unlicensed_specifics(base_answer, styled)
    if bad:
        retry = style_rewrite(
            model,
            tokenizer,
            prompt,
            base_answer,
            history,
            max_new_tokens=max_new_tokens + 80,
            strict=True,
        )
        retry_bad, retry_reason = has_unlicensed_specifics(base_answer, retry)
        if not retry_bad:
            return retry
        return light_wechat_style(base_answer)
    return styled


def answer_normal(model, tokenizer, prompt: str, history, facts: str) -> str:
    messages = [{"role": "system", "content": NORMAL_SYSTEM + "\n\n个人事实：\n" + facts}]
    messages.extend(recent_history(history, 8))
    messages.append({"role": "user", "content": prompt})
    return generate(
        model,
        tokenizer,
        messages,
        max_new_tokens=96,
        temperature=0.45,
        top_p=0.82,
        adapter=True,
    )


def print_help():
    print(
        "命令：\n"
        "  /smart   智能路由：事实/写作先走基座，再用 LoRA 调口吻（默认）\n"
        "  /normal  纯闲聊：直接 LoRA\n"
        "  /fact    强制事实/咨询模式\n"
        "  /write   强制写作模式\n"
        "  /rewrite 强制只改写成你的语气\n"
        "  /facts   显示个人事实文件路径\n"
        "  /reset   清空上下文\n"
        "  /exit    退出"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="outputs/qwen3_8b_wechat_lora")
    ap.add_argument("--mode", default="smart", choices=["smart", ROUTE_NORMAL, ROUTE_FACT, ROUTE_WRITE, ROUTE_REWRITE])
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
    mode = args.mode
    facts = read_facts()
    print("加载完成。默认 /smart；输入 /help 看命令。", flush=True)

    while True:
        raw_prompt = read_line()
        prompt = sanitize_input(raw_prompt)
        if not prompt:
            continue
        if prompt in {"/exit", "exit", "quit"}:
            break
        if prompt == "/help":
            print_help()
            continue
        if prompt == "/reset":
            history.clear()
            print("模型> 上下文清空了")
            continue
        if prompt in {"/smart", "/normal", "/fact", "/write", "/rewrite"}:
            mode = prompt[1:]
            print(f"模型> 已切到 /{mode}")
            continue
        if prompt == "/facts":
            print(f"模型> 个人事实文件：{FACTS_PATH.resolve()}")
            print("模型> 改完后输入 /reload_facts 重新加载")
            continue
        if prompt == "/reload_facts":
            facts = read_facts()
            print("模型> 已重新加载个人事实")
            continue

        route = classify(prompt, mode)
        try:
            if route == ROUTE_NORMAL:
                reply = answer_normal(model, tokenizer, prompt, history, facts)
            elif route == ROUTE_REWRITE:
                reply = style_rewrite(model, tokenizer, "把这段话改成我的语气", prompt, history)
            else:
                base_answer = answer_base(model, tokenizer, prompt, history, route, facts)
                if route == ROUTE_FACT:
                    reply = guarded_style_rewrite(model, tokenizer, prompt, base_answer, history)
                else:
                    reply = base_answer
        except Exception as exc:
            print(f"模型> 这句处理时遇到输入/编码问题，请重新打一遍。({type(exc).__name__})")
            continue

        print("模型>", reply)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

