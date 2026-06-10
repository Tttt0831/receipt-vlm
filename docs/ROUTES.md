# Receipt-VLM 训练路线对比

本文档对比两种训练路线的技术方案、配置和使用方法。

---

## 📋 路线概览

| 路线 | 架构 | 模型规模 | 显存需求 | 配置/脚本 |
|------|------|----------|----------|-----------|
| **路线A** | SigLIP2 + Projection + Qwen2-1.5B LoRA | ~1.5B (base) + LoRA | ~16-18GB | `configs/route_a_synthetic_only.yaml` |
| **路线B** | Qwen2-VL-2B-Instruct LoRA | ~2B (base) + LoRA | ~20-22GB | `src/train_qwen2vl_lora.py` |

---

## 路线 A: 自建 VLM (SigLIP2 + Qwen2-1.5B)

### 架构图

```
输入图片
    ↓
SigLIP2 NaFlex 视觉编码器 (冻结)
    ↓
MLP Projection (可训练)
    ↓
Qwen2-1.5B 语言模型 + LoRA (部分可训练)
    ↓
结构化 JSON 输出
```

### 技术特点

- **视觉编码**: SigLIP2 NaFlex，支持动态分辨率
- **投影层**: 2层 MLP，768 → 3072 → 1536
- **语言模型**: Qwen2-1.5B，通过 LoRA 微调
- **冻结策略**: vision 冻结，LLM 中间层冻结

### 配置参数

| 参数 | 值 |
|------|------|
| vision_model | google/siglip2-base-patch16-naflex |
| llm_type | hf_lora |
| hf_model | Qwen/Qwen2-1.5B |
| lora_r | 16 |
| lora_alpha | 32 |
| llm_hidden_size | 1536 |
| max_num_patches | 1024 |
| batch_size | 4 |
| gradient_accumulation_steps | 4 |
| learning_rate | 5e-5 |
| epochs | 10 |

### 训练命令

```bash
cd /root/autodl-tmp/receipt-vlm

# 完整训练
python -m src.train --config configs/route_a_synthetic_only.yaml

# 自定义参数
python -m src.train \
  --config configs/route_a_synthetic_only.yaml \
  --epochs 10 \
  --batch-size 4
```

### 优缺点

**✅ 优点**
- 轻量级架构，推理速度快
- 模块化设计，易于调试
- 显存需求较低（~16-18GB）
- 支持动态分辨率输入

**❌ 缺点**
- 需要自己实现多模态融合
- LoRA checkpoint 保存/加载较复杂
- 效果可能不如原生VLM

---

## 路线 B: Qwen2-VL-2B 直接微调

### 架构图

```
输入图片 + 文本提示
    ↓
Qwen2-VL-2B-Instruct (原生VLM)
    ↓
LoRA 适配器 (可训练)
    ↓
结构化 JSON 输出
```

### 技术特点

- **模型**: Qwen2-VL-2B-Instruct（现成VLM）
- **微调方式**: LoRA
- **视觉处理**: 原生 ViT + 动态分辨率
- **训练效率**: 直接微调，无需额外组件

### 配置参数

| 参数 | 值 |
|------|------|
| model | Qwen/Qwen2-VL-2B-Instruct |
| lora_r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| target_modules | all-linear |
| batch_size | 2 |
| gradient_accumulation_steps | 8 |
| learning_rate | 2e-4 |
| epochs | 5 |

### 训练命令

```bash
cd /root/autodl-tmp/receipt-vlm

# 完整训练
python src/train_qwen2vl_lora.py \
  --data data/synthetic/train.jsonl \
  --val-data data/data/synthetic/val.jsonl

# 快速测试
python src/train_qwen2vl_lora.py \
  --max-samples 50 \
  --epochs 1 \
  --batch-size 1

# 自定义参数
python src/train_qwen2vl_lora.py \
  --data data/synthetic/train.jsonl \
  --val-data data/data/synthetic/val.jsonl \
  --epochs 10 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 2e-4 \
  --output-dir checkpoints/route_b
```

### 优缺点

**✅ 优点**
- 使用成熟的原生VLM，效果更好
- LoRA checkpoint 保存/加载简单
- 代码实现更简洁
- 预训练能力强，泛化性好

**❌ 缺点**
- 显存需求较高（~20-22GB）
- 模型较大，推理速度较慢
- 依赖第三方模型（更新可能影响兼容性）

---

## 🔄 路线对比总结

| 维度 | 路线A | 路线B |
|------|-------|-------|
| **模型大小** | 较小 | 较大 |
| **显存需求** | 16-18GB | 20-22GB |
| **推理速度** | 快 | 中等 |
| **训练难度** | 中等 | 简单 |
| **效果预期** | 基础 | 更好 |
| **可维护性** | 高 | 中等 |
| **部署成本** | 低 | 中 |

---

## 📊 数据划分

所有路线使用相同的合成数据（5000/500/500划分）：

| 用途 | 路径 | 样本数 |
|------|------|--------|
| 训练集 | `data/synthetic/train.jsonl` | 5000 |
| 验证集 | `data/data/synthetic/val.jsonl` | 500 |
| 测试集 | `data/data/synthetic/test.jsonl` | 500 |

---

## 🎯 推荐选择

### 选择路线A如果：
- 显存有限（<20GB）
- 需要快速推理
- 希望完全控制模型架构
- 追求轻量化部署

### 选择路线B如果：
- 有足够显存（≥20GB）
- 追求最佳效果
- 希望快速实现
- 依赖成熟VLM的泛化能力

---

## 📝 训练检查清单

### 路线A训练前检查

- [x] 数据路径正确
- [ ] Qwen/Qwen2-1.5B 模型已下载
- [ ] LoRA 配置正确
- [ ] 显存充足

### 路线B训练前检查

- [x] 数据路径正确
- [ ] Qwen/Qwen2-VL-2B-Instruct 模型已下载
- [ ] transformers 版本兼容
- [ ] 显存充足（≥20GB）

---

## 🔗 相关文档

- [项目README](../README.md)
- [已知问题](KNOWN_ISSUES.md)
- [数据指南](DATA_GUIDE.md)
