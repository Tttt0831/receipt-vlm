# Receipt-VLM 项目计划 (最终状态)

**最后更新**: 2026-06-10  
**项目状态**: ✅ 路线A完成评估，路线B发现训练Bug

---

## 📊 最终结果总结

### 项目背景
- **目标**: 构建轻量级中英文票据VLM，端到端抽取6个关键字段
- **架构**: SigLIP2 + Projection + Qwen2-1.5B LoRA
- **数据集**: 5000合成训练样本，500验证样本，500测试样本

---

## 🎯 路线A执行结果

### ✅ 已完成
1. **数据准备**: 合成数据集已构建 (5000/500/500划分)
2. **模型搭建**: SigLIP2 + Qwen2-1.5B + LoRA架构就绪
3. **训练完成**: 成功完成8个epoch训练
4. **性能测评**: 完成best_model.pt (epoch 4)测评

### 📈 训练指标

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|----------|------|
| 1 | 1.3671 | 1.3949 | baseline |
| 2 | 1.3018 | 1.2732 | ✅ 改善 |
| 3 | 1.1755 | 1.2047 | ✅ 改善 |
| 4 | 1.1009 | 1.1859 | 🏆 最佳 |
| 5 | 1.0168 | 1.1947 | ⚠️ val上升 |
| 6 | 0.9229 | 1.2421 | ⚠️ 过拟合 |
| 7 | 0.8287 | 1.2848 | ⚠️ 过拟合 |
| 8 | 0.7489 | 1.3270 | ❌ 严重过拟合 |

### 📊 评估结果 (Epoch 4, 500样本)

| 字段 | 准确率 | 状态 |
|------|--------|------|
| merchant_name | 0.00% | ❌ 完全失败 |
| date | 0.00% | ❌ 完全失败 |
| total_amount | 0.00% | ❌ 完全失败 |
| tax_amount | 22.00% | ⚠️ 很低 |
| tax_id | 29.20% | ⚠️ 很低 |
| invoice_no | 22.60% | ⚠️ 很低 |

### 总体指标
- **Value Match Rate**: 12.30%
- **JSON Validity Rate**: 100%
- **Hallucination Rate**: 0%

---

## 🔍 核心问题分析

### 问题: 模型输出缺少开头的 `'{"'` token

**现象**: 
```
正确: {"merchant_name":"...","date":"..."}
实际: merchant_name":"...","date":"..."}
       ↑缺少 '{"' token
```

**原因分析**:
- BPE tokenizer将 `'{"'` 合并为一个token (ID: 179)
- 模型学会了从 `'merchant'` (ID: 158) 开始
- 训练数据格式正确，但模型未能学会生成开头的 `'{"'`

**影响**:
- JSON格式不完整，解析失败
- 所有字段准确率为0%
- Value Match Rate仅12.3%

---

## 🚨 路线B执行结果

### ✅ 已完成
1. **训练完成**: 5个epoch训练
2. **LoRA保存**: 17MB adapter权重

### ❌ 发现严重Bug

**Bug位置**: `src/train_qwen2vl_lora.py` 第136-140行

```python
# 构造 labels — 训练时目标为同一份 token ids
# 但我们需要 labels 只作用在答案部分。简化处理：对整个序列计算 loss
labels = inputs["input_ids"].clone()
# 将 padding token 设为 -100
labels[labels == processor.tokenizer.pad_token_id] = -100
```

**问题**:
- 模型被训练预测**整个序列**（包括用户提示）
- 而不是只预测**助手回复**
- 导致模型学会复制提示词，而不是生成JSON

**结果**:
- 模型输出拒绝响应："很抱歉，由于当前我只能基于文字描述..."
- 无法生成有效的JSON

**修复方案**:
需要正确设置labels，只对助手回复部分计算loss：
```python
# 正确的做法
labels = inputs["input_ids"].clone()
# 将用户部分设为 -100
labels[:assistant_start] = -100
```

### 📈 训练指标 (路线B)

| Epoch | Train Loss | Val Loss | 状态 |
|-------|-----------|----------|------|
| 1 | 7.0183 | 6.5323 | baseline |
| 2 | 6.5470 | 6.5309 | ✅ 改善 |
| 3 | 6.5444 | 6.5307 | ✅ 稳定 |
| 4 | 6.5442 | 6.5310 | → 稳定 |
| 5 | 6.5400 | 6.5307 | ✅ 最佳 |

---

## 📋 文件清单

### 新增文件
- `PLAN.md` - 项目计划文档
- `configs/route_a_synthetic_only.yaml` - 路线A配置
- `docs/ROUTES.md` - 路线对比文档
- `evaluation_results/route_a_epoch4.json/` - 评估结果
- `checkpoints/route_a_synthetic/` - 模型checkpoint
- `checkpoints/route_b/best_lora/` - 路线B LoRA adapter

### 训练日志
- `training_route_a.log` - 路线A训练日志
- `training_route_b.log` - 路线B训练日志

### 数据文件
- `data/synthetic/train.jsonl` - 5000训练样本
- `data/data/synthetic/val.jsonl` - 500验证样本
- `data/data/synthetic/test.jsonl` - 500测试样本

---

## 🎯 结论

### 路线A
- ✅ 训练成功完成
- ✅ 模型可以运行
- ❌ 核心指标未达标（0%准确率）
- ❌ 存在JSON格式问题

### 路线B
- ✅ 训练成功完成
- ❌ 发现训练脚本Bug
- ❌ 模型无法生成JSON
- ⚠️ 需要修复后重新训练

### 建议
1. 修复路线B训练脚本中的labels设置
2. 重新训练路线B
3. 或者改进路线A的tokenizer来解决 `'{"'` 问题

---

## 🔗 相关资源

- [项目README](README.md)
- [路线对比](docs/ROUTES.md)
