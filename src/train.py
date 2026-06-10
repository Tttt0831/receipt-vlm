"""
训练脚本（config 驱动，真实两阶段训练）

- Stage A (configs/stage1.yaml): 英文/CORD/SROIE 预训练，学会模态对齐 + JSON 格式。
- Stage B (configs/stage2.yaml): 加载 Stage A 权重，在中文票据上精调。

与旧版的区别：使用真实 tokenizer、真实图片、对齐的 causal-LM 标签
（损失只作用在 JSON 答案上），而非随机张量。

用法：
    python -m src.train --config configs/stage1.yaml
    # CPU 小样本 smoke：
    python -m src.train --config configs/stage1.yaml \
        --data data/synthetic/train.jsonl --vision fallback \
        --max-samples 50 --epochs 1 --batch-size 4
"""
import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.vlm import ReceiptVLM, VLMConfig
from src.model.tokenizer import get_tokenizer
from src.data.dataset import ReceiptDataset, build_training_collate


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg, tokenizer, vision_model_name, device):
    """根据配置 + tokenizer 构建模型。vocab_size 由 tokenizer 决定。"""
    seq = cfg.get("sequence", {})
    freezing = cfg.get("freezing", {})
    model_cfg = cfg.get("model", {})
    strategy = freezing.get("freeze_strategy",
        "projection_llm_ends" if freezing.get("trainable_llm_layers") else "projection_only")

    vlm_config = VLMConfig(
        vision_model_name=vision_model_name,
        freeze_vision=freezing.get("freeze_vision", True),
        max_num_patches=seq.get("max_num_patches", 256),
        llm_type=model_cfg.get("llm_type", "mini"),
        llm_vocab_size=tokenizer.vocab_size,
        llm_hidden_size=model_cfg.get("llm_hidden_size", 512),
        llm_num_layers=model_cfg.get("llm_num_layers", 6),
        llm_num_heads=8,
        llm_intermediate_size=model_cfg.get("llm_intermediate_size", 2048),
        hf_model_name=model_cfg.get("hf_model_name", "Qwen/Qwen2-1.5B"),
        hf_lora_r=model_cfg.get("hf_lora_r", 16),
        hf_lora_alpha=model_cfg.get("hf_lora_alpha", 32),
        hf_lora_dropout=model_cfg.get("hf_lora_dropout", 0.05),
        projection_intermediate_dim=model_cfg.get("projection_intermediate_dim", 2048),
        max_sequence_length=seq.get("max_sequence_length", 1024),
        gradient_checkpointing=cfg.get("training", {}).get("gradient_checkpointing", False),
        freeze_strategy=strategy,
        pad_token_id=tokenizer.pad_token_id,
        image_token_id=tokenizer.image_token_id,
        boa_token_id=tokenizer.boa_token_id,
        eoa_token_id=tokenizer.eoa_token_id,
    )
    model = ReceiptVLM(vlm_config).to(device)
    return model


def make_loaders(train_path, val_path, max_samples, batch_size, collate):
    train_ds = ReceiptDataset(str(train_path), max_samples=max_samples, split="train")
    if val_path and Path(val_path).exists():
        val_ds = ReceiptDataset(str(val_path), split="val")
    else:
        # 没有独立验证集 → 从训练集切 10%
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(train_ds, [n_train, n_val])
        print(f"未找到验证集，从训练集切分: train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    return train_loader, val_loader


def lr_at(step, total_steps, warmup_steps, base_lr):
    """线性 warmup + 余弦衰减。"""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_epoch(model, loader, optimizer, device, train, epoch, tcfg, total_steps, step_offset, use_amp):
    model.train() if train else model.eval()
    desc = f"Epoch {epoch} [{'train' if train else 'val'}]"
    total_loss, n = 0.0, 0
    accum = tcfg.get("gradient_accumulation_steps", 1)
    base_lr = float(tcfg.get("learning_rate", 1e-4))
    warmup = tcfg.get("warmup_steps", 0)
    max_norm = tcfg.get("max_grad_norm", 1.0)

    pbar = tqdm(loader, desc=desc)
    for i, batch in enumerate(pbar):
        vision_inputs = batch["vision_inputs"]
        if isinstance(vision_inputs, dict):
            vision_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vision_inputs.items()}
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(vision_inputs, input_ids, attn, labels)
                    loss = out["loss"]
            else:
                out = model(vision_inputs, input_ids, attn, labels)
                loss = out["loss"]

        if train:
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step_offset + i, total_steps, warmup, base_lr)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Receipt-VLM 训练")
    parser.add_argument("--config", required=True, help="stage 配置 yaml")
    parser.add_argument("--data", default=None, help="覆盖训练数据 jsonl 路径")
    parser.add_argument("--val-data", default=None, help="覆盖验证数据 jsonl 路径")
    parser.add_argument("--vision", default=None, help="覆盖视觉编码器 (如 fallback)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-from", default=None, help="初始化权重（Stage B 加载 Stage A）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg.get("training", {})
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda") and (tcfg.get("mixed_precision") in ("bf16", "fp16"))
    print(f"Stage: {cfg.get('stage', {}).get('name')} | device: {device} | amp: {use_amp}")

    # 视觉编码器：CLI > config 默认（README 默认真 SigLIP2）
    vision_model_name = args.vision or cfg.get("model", {}).get("vision_model_name") \
        or "google/siglip2-base-patch16-naflex"

    tokenizer = get_tokenizer()
    model = build_model(cfg, tokenizer, vision_model_name, device)
    model.print_trainable_parameters()

    # 初始化权重（Stage B）
    init_from = args.init_from or cfg.get("checkpointing", {}).get("resume_from_checkpoint")
    if init_from and Path(init_from).exists():
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ 已从 {init_from} 初始化权重")

    # 数据
    data_cfg = cfg.get("data", {})
    train_path = args.data or data_cfg.get("train_path")
    val_path = args.val_data or data_cfg.get("val_path")
    if not train_path or not Path(train_path).exists():
        raise SystemExit(f"训练数据不存在: {train_path}\n请用 --data 指定 jsonl，或先补齐 {data_cfg.get('train_path')}")

    collate = build_training_collate(
        tokenizer, model.vision_encoder.process_images,
        max_length=cfg.get("sequence", {}).get("max_sequence_length", 1024),
    )
    train_loader, val_loader = make_loaders(
        train_path, val_path, args.max_samples, tcfg.get("batch_size", 4), collate)

    # 优化器（只优化可训练参数）
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(tcfg.get("learning_rate", 1e-4)),
                                  weight_decay=float(tcfg.get("weight_decay", 0.01)))

    epochs = tcfg.get("epochs", 3)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs

    out_dir = Path(cfg.get("checkpointing", {}).get("output_dir", "checkpoints/stage"))
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    # 定期保存配置
    save_every_n = cfg.get("checkpointing", {}).get("save_every_n_epochs", 0)
    save_from_epoch = cfg.get("checkpointing", {}).get("save_from_epoch", 0)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True, epoch,
                               tcfg, total_steps, (epoch - 1) * steps_per_epoch, use_amp)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, device, False, epoch,
                                 tcfg, total_steps, 0, use_amp)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # 定期保存 checkpoint（仅在达到 save_from_epoch 后）
        if save_every_n > 0 and epoch % save_every_n == 0 and epoch >= save_from_epoch:
            epoch_path = out_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": tokenizer.vocab_size,
                "model_config": asdict(model.config),
                "vision_model_name": vision_model_name,
                "stage": cfg.get("stage", {}).get("name"),
            }, epoch_path)
            print(f"  ✓ 保存定期 checkpoint: {epoch_path} (epoch {epoch})")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": tokenizer.vocab_size,
                "model_config": asdict(model.config),
                "vision_model_name": vision_model_name,
                "stage": cfg.get("stage", {}).get("name"),
            }, best_path)
            print(f"  ✓ 保存最优 checkpoint: {best_path} (val={val_loss:.4f})")

    print(f"\n训练完成。最优 val loss: {best_val:.4f}")
    print(f"checkpoint: {best_path}")


if __name__ == "__main__":
    main()
