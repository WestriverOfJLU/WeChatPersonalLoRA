# WeChat Personal LoRA

Local toolkit for fine-tuning a Qwen chat model on personal chat-style data, then running it through a safer router:

- LoRA adapter: learns personal reply tone.
- Base model: handles factual consulting, technical questions, and writing.
- Router chat: decides when to use the base model, when to use LoRA, and when to rewrite reliable content into the personal style.

This repository intentionally does **not** include private chat logs, generated datasets, model checkpoints, or LoRA weights.

## Why Router Chat

A style LoRA is good at sounding like a person, but bad at knowing facts. If you ask it about schools, teachers, majors, technical concepts, or writing tasks, a raw LoRA can hallucinate.

The recommended flow is:

```text
user message
  -> classify route
  -> factual / consulting / writing: base Qwen first
  -> optional guarded style rewrite with LoRA
  -> final answer
```

The guarded rewrite checks for risky changes such as newly invented people, schools, institutions, rankings, or overconfident claims.

## Features

- QLoRA training script for Qwen-style chat models.
- Contextual SFT JSONL normalizer.
- Pure LoRA chat loop for style testing.
- Router chat for practical use.
- Windows PowerShell launchers for WSL.
- Input sanitization for PowerShell/WSL terminal edge cases.
- Personal facts file to reduce identity and background hallucinations.

## Repository Layout

```text
scripts/
  train_lora.py        Train a 4-bit QLoRA adapter
  router_chat.py       Recommended router chat
  chat_loop.py         Pure LoRA chat loop
  chat_with_lora.py    Single-prompt LoRA inference
  normalize_jsonl.py   Normalize contextual SFT JSONL
  build_dataset.py     Build simple style-only data from CSV

data/
  personal_facts.example.md

requirements-wsl.txt
start_router_chat.ps1
start_chat.ps1
run_samples.ps1
status_train.ps1
```

Ignored local-only paths include `data/*.jsonl`, `outputs/`, `logs/`, `.venv/`, and model weight files.

## Setup

The scripts are intended for WSL with an NVIDIA GPU.

```bash
cd /path/to/wechat_personal_lora
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements-wsl.txt
```

Test CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Data Format

Recommended contextual SFT format:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Put your private JSONL files under `data/`. They are ignored by Git.

If you already have train/validation JSONL:

```bash
python scripts/normalize_jsonl.py \
  --train /path/to/train.jsonl \
  --val /path/to/val.jsonl \
  --out-dir data
```

## Train

Example for Qwen3-8B:

```bash
python scripts/train_lora.py \
  --model Qwen/Qwen3-8B \
  --train data/train_context.jsonl \
  --val data/val_context.jsonl \
  --output outputs/qwen3_8b_wechat_lora \
  --max-seq-length 1024 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --eval-max-samples 256 \
  --eval-steps 500 \
  --save-steps 500 \
  --logging-steps 10
```

For lower VRAM pressure, reduce `--max-seq-length` to `768`.

## Run

Copy the example facts file and edit it:

```bash
cp data/personal_facts.example.md data/personal_facts.md
```

Recommended router chat:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_router_chat.ps1
```

Pure LoRA style test:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_chat.ps1
```

Router commands:

```text
/smart   default router mode
/normal  pure LoRA chat
/fact    force factual / consulting mode
/write   force writing mode
/rewrite rewrite input into the personal tone
/facts   show the personal facts file path
/reset   clear context
/exit    quit
```

## Publish Adapter Weights

If you want to publish your trained adapter, publish it separately from this code repository. Adapter-only publishing is safer than publishing a merged full model:

```bash
hf auth login
hf repo create <user>/<repo-name> --type model --private
hf upload <user>/<repo-name> outputs/qwen3_8b_wechat_lora . --repo-type model
```

Review the adapter model card carefully before making it public.

## Privacy Notes

Do not commit:

- Raw chat exports.
- Generated SFT datasets.
- `personal_facts.md` with private details.
- LoRA checkpoints or merged model weights.
- Debug files that may contain user input.
- Tokens or credentials.

This project is for local experimentation and personal assistant research. Do not use it to impersonate a real person without consent.

## License

MIT. See [LICENSE](LICENSE).
