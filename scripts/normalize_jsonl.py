import argparse
import json
from pathlib import Path


def repair_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return text
    try:
        repaired = text.encode("gbk").decode("utf-8")
    except UnicodeError:
        return text
    # Keep the repaired version only when it clearly improves common mojibake.
    bad_before = sum(text.count(ch) for ch in "浣犳槸鐨勬垜瀵规柟")
    bad_after = sum(repaired.count(ch) for ch in "浣犳槸鐨勬垜瀵规柟")
    return repaired if bad_after < bad_before else text


def normalize_obj(obj):
    for message in obj.get("messages", []):
        message["content"] = repair_mojibake(message.get("content", ""))
    return obj


def convert(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = normalize_obj(json.loads(line))
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    train_n = convert(args.train, args.out_dir / "train_context.jsonl")
    val_n = convert(args.val, args.out_dir / "val_context.jsonl")
    stats = {
        "train": train_n,
        "val": val_n,
        "train_path": str(args.out_dir / "train_context.jsonl"),
        "val_path": str(args.out_dir / "val_context.jsonl"),
        "note": "Mojibake repaired with gbk->utf-8 roundtrip where beneficial.",
    }
    (args.out_dir / "context_dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
