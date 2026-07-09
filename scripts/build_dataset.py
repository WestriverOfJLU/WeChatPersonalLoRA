import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


SYSTEM = (
    "你正在模仿目标用户在微信里的回复风格。"
    "回复要自然、口语化、简短优先，保留他的措辞习惯；"
    "不要解释你在模仿，也不要自称模型。"
)


BAD_PATTERNS = [
    re.compile(r"https?://", re.I),
    re.compile(r"<[^>]+>"),
    re.compile(r"^\s*\[.*?\]\s*$"),
]


def clean_text(text: str) -> str:
    text = (text or "").replace("\u200b", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def ok_target(text: str) -> bool:
    if not (2 <= len(text) <= 500):
        return False
    if any(p.search(text) for p in BAD_PATTERNS):
        return False
    if text.isdigit():
        return False
    return True


def read_context_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "messages" in obj:
                yield obj


def build_style_rows(csv_path: Path, limit_per_talker: int | None):
    rows = []
    talker_counts = Counter()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = clean_text(row.get("content", ""))
            if not ok_target(text):
                continue
            talker = row.get("talker_id", "unknown")
            if limit_per_talker and talker_counts[talker] >= limit_per_talker:
                continue
            talker_counts[talker] += 1
            when = row.get("datetime", "")
            month = row.get("year_month", "")
            prompt = (
                "请生成一条像我本人会在微信里发出的回复。\n"
                f"聊天对象ID: {talker}\n"
                f"时间: {when or month}\n"
                "要求: 只输出回复文本，不要加引号。"
            )
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text},
                    ],
                    "metadata": {
                        "source": "self_message_style",
                        "talker_id": talker,
                        "datetime": when,
                        "year_month": month,
                    },
                }
            )
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path)
    ap.add_argument("--context-jsonl", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260521)
    ap.add_argument("--limit-per-talker", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    context_path = args.context_jsonl
    if not context_path and args.summary_json and args.summary_json.exists():
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        candidate = Path(summary.get("paths", {}).get("all", ""))
        if candidate.exists():
            context_path = candidate

    if context_path and context_path.exists():
        rows = list(read_context_jsonl(context_path))
        source = str(context_path)
    else:
        rows = build_style_rows(
            args.csv,
            limit_per_talker=args.limit_per_talker or None,
        )
        source = str(args.csv)

    random.shuffle(rows)
    val_n = max(100, int(len(rows) * args.val_ratio)) if len(rows) > 1000 else max(1, int(len(rows) * args.val_ratio))
    val = rows[:val_n]
    train = rows[val_n:]

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    stats = {
        "source": source,
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "format": "messages JSONL",
        "note": "self-message style data unless context-jsonl exists",
    }
    (args.out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

