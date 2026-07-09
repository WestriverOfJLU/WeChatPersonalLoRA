# WeChatPersonalLoRA

一个用于本地训练和运行“个人聊天风格 LoRA”的工具箱。项目以 Qwen 系列聊天模型为基座，支持用个人聊天风格数据训练 LoRA，并通过路由聊天脚本把“事实可靠性”和“个人语气”分开处理。

本仓库只包含代码，不包含私人聊天记录、生成的数据集、模型 checkpoint 或 LoRA 权重。

## 为什么需要路由聊天

风格 LoRA 很擅长“说话像某个人”，但不适合独自承担事实问答。直接让 LoRA 回答学校、导师、专业、技术概念或写作任务时，很容易出现幻觉。

推荐流程是：

```text
用户消息
  -> 判断问题类型
  -> 事实 / 咨询 / 写作：先由基座模型生成可靠内容
  -> 需要个人口吻时，再由 LoRA 做受约束改写
  -> 输出最终回答
```

`router_chat.py` 会检查 LoRA 改写是否新增了可疑的人名、学校、机构、排名或过度确定的判断，尽量避免“为了像你而把事实改歪”。

## 功能

- 使用 4-bit QLoRA 训练 Qwen 风格模型。
- 支持上下文 SFT JSONL 数据格式。
- 提供纯 LoRA 聊天脚本，用于测试个人语气。
- 提供推荐的 router 聊天脚本，用于更实用的日常对话。
- 支持 Windows PowerShell + WSL 启动。
- 针对 PowerShell / WSL 终端输入中的退格、异常空格、非法 Unicode 做了净化。
- 支持本地 `personal_facts.md`，减少身份和背景信息幻觉。

## 目录结构

```text
scripts/
  train_lora.py        训练 4-bit QLoRA adapter
  router_chat.py       推荐使用的路由聊天脚本
  chat_loop.py         纯 LoRA 聊天脚本
  chat_with_lora.py    单轮 LoRA 推理
  normalize_jsonl.py   标准化上下文 SFT JSONL
  build_dataset.py     从 CSV 构建简单风格数据

data/
  personal_facts.example.md

requirements-wsl.txt
start_router_chat.ps1
start_chat.ps1
run_samples.ps1
status_train.ps1
```

以下内容默认被 `.gitignore` 排除：

```text
data/*.jsonl
data/personal_facts.md
outputs/
logs/
.venv/
*.safetensors
*.gguf
```

## 环境准备

建议在 WSL 中使用 NVIDIA GPU 运行。

```bash
cd /path/to/WeChatPersonalLoRA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements-wsl.txt
```

检查 CUDA：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 数据格式

推荐使用上下文 SFT JSONL，每行一条样本：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

私人数据请放到 `data/` 目录下。该目录里的真实数据默认不会进入 Git。

如果你已经有训练集和验证集：

```bash
python scripts/normalize_jsonl.py \
  --train /path/to/train.jsonl \
  --val /path/to/val.jsonl \
  --out-dir data
```

## 训练

以 `Qwen/Qwen3-8B` 为例：

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

如果显存压力较大，可以把 `--max-seq-length` 降到 `768`。

## 运行

先复制个人事实模板：

```bash
cp data/personal_facts.example.md data/personal_facts.md
```

然后按需填写 `data/personal_facts.md`。这里只应该写确定无误、可以被本地助手使用的事实。

推荐使用 router 聊天：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_router_chat.ps1
```

纯 LoRA 风格测试：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_chat.ps1
```

router 内置命令：

```text
/smart   默认智能路由
/normal  纯 LoRA 闲聊
/fact    强制事实 / 咨询模式
/write   强制写作模式
/rewrite 把输入改写成个人口吻
/facts   显示个人事实文件路径
/reset   清空上下文
/exit    退出
```

## 发布 LoRA 权重

如果要发布训练好的 adapter，建议单独发布到 Hugging Face，而不是放进 GitHub 仓库。adapter-only 比合并完整模型更安全，也更适合分发。

```bash
hf auth login
hf repo create <user>/<repo-name> --type model --private
hf upload <user>/<repo-name> outputs/qwen3_8b_wechat_lora . --repo-type model
```

公开前请认真检查 model card，确认不包含私人聊天记录、个人身份信息或不适合公开的示例。

## 隐私提醒

请不要提交：

- 原始聊天记录。
- 生成后的 SFT 数据集。
- 带私人信息的 `personal_facts.md`。
- LoRA checkpoint 或合并模型权重。
- tokenizer/debug 日志。
- token、cookie、密钥或其他凭据。

本项目用于本地实验和个人助手研究。请勿在未经同意的情况下用它冒充真实个人。

## 许可证

MIT，见 [LICENSE](LICENSE)。
