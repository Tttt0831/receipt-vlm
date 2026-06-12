# 已知问题与待办（Known Issues）

> 记录技术债与待修问题，避免遗忘。

---

## 0. 路线 C：全量语言预训练发散（产物为随机模型）⛔ 阻塞路线 C

**严重度**: 高（路线 C 当前不可用）

**状态**: 待修复

### 现象
- `src/pretrain_lm.py` 的 **smoke（80 step）正常**：loss 9.6 → 7.85，稳步下降。
- **全量（102k step, bs=16）异常**：loss 从 9.62 起，约 4000 step 后塌成假的 `0.0000`（ppl=1.0），其后一直为 0。
- 核验保存的 `checkpoints/route_c/llm_pretrained.pt`：在**训练数据**上 loss ≈ `ln(12000)=9.4`、next-token acc ≈ 8%、生成 `再再再再` 死循环 → **就是个随机模型**。权重无 NaN/爆炸，仅 `ln_f` 增益从 1.0 漂到 2.26，说明梯度信号几乎为 0。

### 判断
损失"假塌成 0"把梯度清零，模型根本没学到东西。语料已确认 100% 不重复（200000/200000），排除数据退化。疑因：
- 手写 attention（`src/model/llm.py`）用 `float('-inf')` 掩码，在 **bf16 autocast** 下 softmax 可能产生 NaN/饱和，污染 loss。
- 学习率 3e-4 对从零训练偏高，叠加上面的不稳定。

### 修复方向
- loss 计算**强制 fp32**（autocast 内显式 `.float()` 再算 CrossEntropy），或干脆把 loss 移出 autocast。
- attention 改用 `torch.nn.functional.scaled_dot_product_attention`（数值稳定、自带 causal）替换手写 `-inf` 掩码；顺带解决长序列 O(seq²) 显存问题。
- 降 LR（如 1e-4）+ 更长 warmup + 确认 grad clip 生效；加 loss/grad 的 NaN 守卫。
- 修好后用"能否生成连贯中文 + held-out next-token acc"做验收，再进 Stage 1/2。

---

## 1. 路线 A (HF LoRA) 的 checkpoint 保存/加载方式低效且可能 key 不匹配

**严重度**: 中（已在 GPU 上确认体积问题）

**状态**: 已确认（best_model.pt 实测 4.6GB），待修复

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
