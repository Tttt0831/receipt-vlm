# 已知问题与待办（Known Issues）

> 记录尚未在 GPU 环境验证/修复的技术债，避免遗忘。

---

## 1. 路线 A (HF LoRA) 的 checkpoint 保存/加载方式低效且可能 key 不匹配

**严重度**: 中（不影响 CPU 冒烟测试，但会影响 GPU 真实训练后的复现/推理）

**状态**: 待 GPU 环境验证后修复

### 现象

当前 [`src/train.py`](../src/train.py) 的 checkpoint 保存逻辑对所有模型统一使用：

```python
torch.save({
    "model_state_dict": model.state_dict(),   # ← 问题所在
    ...
}, best_path)
```

对路线 A（`llm_type="hf_lora"`，即 SigLIP2 + Projection + Qwen2-1.5B LoRA）这会带来两个问题：

### 问题 1：checkpoint 体积浪费

`model.state_dict()` 包含**完整冻结的 Qwen2-1.5B 基座（~3GB）**，而真正训练的只有：
- LoRA 适配器（~15M 参数）
- MLP Projection（~1-4M 参数）

每个 epoch 存一份 = 每个 checkpoint ~3GB，纯属浪费磁盘。

### 问题 2：加载时 PEFT key 可能对不上

[`src/infer.py`](../src/infer.py) 的 `build_vlm()` 在加载时会：
1. 调用 `create_vlm_from_config()` 重建模型
   → 重新从 HuggingFace 下载 Qwen2-1.5B
   → 重新 `get_peft_model()` 包一层**全新的** LoRA
2. 再 `load_state_dict(checkpoint["model_state_dict"])`

PEFT 包装后参数名会带 `base_model.model.` 前缀和 `lora_A/lora_B` 子模块。
重建出的 PEFT 结构与保存时**理论上一致**，但以下情况会导致 key 不匹配：
- PEFT / transformers 版本不同
- LoRA target_modules 配置在 config 里没完整持久化
- `strict=True`（默认）加载时任何缺失/多余 key 都会报错

### 复现条件

需要 GPU 环境真实跑一次路线 A 训练 → 保存 → 用 `src/run_eval.py` 加载评估。
CPU 环境无法触发（不会真正加载 Qwen2-1.5B）。

---

## 推荐修复方案

对 HF LoRA 模式，**分别保存可训练组件**，而非整个 `state_dict()`：

```python
if model.config.llm_type == "hf_lora":
    # 1. 只存 LoRA 适配器（PEFT 原生格式）
    model.llm.model.save_pretrained(out_dir / "lora_adapter")
    # 2. 单独存 projection 权重
    torch.save(model.projection.state_dict(), out_dir / "projection.pt")
    # 3. 存 meta（config / 特殊 token / vision 名称）
    torch.save({
        "model_config": asdict(model.config),
        "vision_model_name": vision_model_name,
        "vocab_size": tokenizer.vocab_size,
        "val_loss": val_loss,
        "stage": cfg.get("stage", {}).get("name"),
    }, out_dir / "meta.pt")
else:
    # MiniLLM 维持原样
    torch.save({"model_state_dict": model.state_dict(), ...}, best_path)
```

对应地，`src/infer.py` 的 `build_vlm()` 加载时：

```python
if cfg.llm_type == "hf_lora":
    model = create_vlm_from_config(meta["model_config"])         # 重建（含 fresh LoRA）
    model.llm.model.load_adapter(out_dir / "lora_adapter")       # 加载训练好的 LoRA
    model.projection.load_state_dict(torch.load(out_dir / "projection.pt"))
```

好处：
- checkpoint 从 ~3GB 降到 ~20MB
- 用 PEFT 原生 `save_pretrained` / `load_adapter`，规避手动 key 对齐
- 基座权重始终从 HF 拉取，保证版本一致

> 路线 B（`src/train_qwen2vl_lora.py`）已经用了 `model.save_pretrained()` 的正确做法，路线 A 应对齐。

---

## 修复检查清单

- [ ] 改 `src/train.py` 的保存分支（区分 hf_lora / mini）
- [ ] 改 `src/infer.py` 的 `build_vlm()` 加载分支
- [ ] GPU 上跑一次路线 A：训练 1 epoch → 保存 → run_eval 加载，确认无 key 报错
- [ ] 确认 checkpoint 体积 < 50MB
- [ ] README 更新 checkpoint 目录结构说明
