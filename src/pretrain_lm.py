"""
MiniLLM 语言预训练（Stage 0）：在 MiniMind 通用中文语料上做 next-token 预测，
给随机初始化的自制 LLM 一个语言先验，再进入 VLM 对齐/精调。

- 直接用裸 MiniLLM（不带 vision），更轻。
- tokenizer 用本项目自训练的 ~12k BPE（tokenizers/receipt-bpe/）。
- 把语料 tokenize 后打包成定长 block 做 LM。
- 产物：checkpoints/route_c/llm_pretrained.pt
    {'llm_state_dict', 'llm_config', 'tokenizer_dir', 'vocab_size'}

用法：
  # smoke
  python -m src.pretrain_lm --max-lines 2000 --epochs 1 --block-size 512 --batch-size 8
  # full
  python -m src.pretrain_lm --max-lines 300000 --epochs 2 --block-size 512 --batch-size 24
"""
import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer
from src.model.llm import MiniLLM, LLMConfig


class PackedLMDataset(Dataset):
    """把语料 tokenize 后拼接成 token 流，再切成定长 block。"""

    def __init__(self, corpus_path, tokenizer, block_size, max_lines):
        self.block_size = block_size
        eos = tokenizer.eos_token_id
        ids_stream = []
        n = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="tokenizing", total=max_lines):
                if max_lines and n >= max_lines:
                    break
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text:
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                ids_stream.extend(ids)
                ids_stream.append(eos)
                n += 1
        # 切块
        total = (len(ids_stream) // block_size) * block_size
        self.data = torch.tensor(ids_stream[:total], dtype=torch.long).view(-1, block_size)
        print(f"✓ 打包完成: {n} 行 -> {len(ids_stream):,} tokens -> {self.data.shape[0]:,} 个 block(block_size={block_size})")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def lr_at(step, total, warmup, base):
    if warmup > 0 and step < warmup:
        return base * step / warmup
    if total <= warmup:
        return base
    prog = (step - warmup) / (total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * min(prog, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/pretrain_t2t_mini.jsonl")
    ap.add_argument("--tokenizer", default="tokenizers/receipt-bpe")
    ap.add_argument("--out", default="checkpoints/route_c/llm_pretrained.pt")
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--intermediate-size", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--max-lines", type=int, default=300000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-every-steps", type=int, default=2000)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"device={device} amp={use_amp} | tokenizer={args.tokenizer} vocab={vocab_size}")

    cfg = LLMConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.block_size, 2048),
    )
    model = MiniLLM(cfg).to(device)
    print(f"MiniLLM 参数量: {model.num_parameters/1e6:.1f}M")

    ds = PackedLMDataset(args.corpus, tokenizer, args.block_size, args.max_lines)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    loss_fct = torch.nn.CrossEntropyLoss()

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(tag):
        torch.save({
            "llm_state_dict": model.state_dict(),
            "llm_config": asdict(cfg),
            "tokenizer_dir": args.tokenizer,
            "vocab_size": vocab_size,
        }, out_path)
        print(f"  ✓ 保存 LLM 预训练权重: {out_path} ({tag})")

    gstep = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        optim.zero_grad()
        for i, input_ids in enumerate(pbar):
            input_ids = input_ids.to(device)
            with torch.set_grad_enabled(True):
                # 纯 causal LM（attention_mask=None → SDPA is_causal）
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(input_ids)
                else:
                    logits = model(input_ids)
                # loss 强制 fp32，避免 bf16 数值问题污染梯度
                loss = loss_fct(logits[:, :-1].float().reshape(-1, vocab_size),
                                input_ids[:, 1:].reshape(-1))
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                for g in optim.param_groups:
                    g["lr"] = lr_at(gstep, total_steps, args.warmup, args.lr)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                optim.zero_grad()
                gstep += 1
                if args.save_every_steps and gstep % args.save_every_steps == 0:
                    save(f"step {gstep}")
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ppl": f"{math.exp(min(loss.item(),20)):.1f}"})
        save(f"epoch {epoch}")

    print(f"\n预训练完成 -> {out_path}")


if __name__ == "__main__":
    main()
