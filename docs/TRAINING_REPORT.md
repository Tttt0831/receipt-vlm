# 训练实验报告

## 实验配置

### 模型架构
- **Vision Encoder**: SigLIP2-base-patch16-naflex (375M参数, 冻结)
- **LLM**: 自定义12层1024维模型 (35.7M可训练参数)
- **Projection**: MLP层 (3.7M参数)
- **总参数**: 537M (6.66%可训练)

### 训练配置
- **数据集**: 中文Synthetic收据数据
  - 训练集: 3,998样本
  - 验证集: 497样本
  - 测试集: 505样本
- **配置文件**: configs/stage1_chinese.yaml
- **训练轮数**: 5 epochs
- **批次大小**: 4
- **梯度累积**: 4 steps
- **学习率**: 5e-5
- **混合精度**: BF16

## 训练结果

### 损失曲线
| Epoch | 训练损失 | 验证损失 | 改进 |
|-------|---------|---------|------|
| 1 | 3.6748 | 1.7289 | 基准 |
| 2 | 1.5749 | 1.4383 | ✅ 16.8% |
| 3 | 1.4020 | 1.3514 | ✅ 6.0% |
| 4 | 1.3312 | 1.3119 | ✅ 3.0% |
| 5 | 1.3002 | 1.3033 | ✅ 0.6% |

**改进总结**:
- 训练损失降低: 65% (3.6748 → 1.3002)
- 验证损失降低: 25% (1.7289 → 1.3033)

### 模型保存
所有checkpoint保存在: `checkpoints/stage_a_chinese/`
- best_model.pt (最优模型, val_loss=1.3033)
- epoch_1.pt 到 epoch_5.pt

## 评估结果

### 测试配置
- **测试集**: 中文Synthetic测试数据 (20样本)
- **评估脚本**: src/run_eval.py
- **模型**: checkpoints/stage_a_chinese/best_model.pt

### 指标表现
| 指标 | 结果 | 分析 |
|------|------|------|
| **JSON有效性** | 100.0% | ✅ 格式完全正确 |
| **幻觉率** | 0.0% | ✅ 无幻觉问题 |
| **值匹配率** | 17.5% | ❌ 需要改进 |

### 字段级别准确率
| 字段 | 准确率 | 状态 |
|------|--------|------|
| merchant_name | 0.0% | ❌ 需要改进 |
| date | 0.0% | ❌ 需要改进 |
| total_amount | 0.0% | ❌ 需要改进 |
| tax_amount | 20.0% | ⚠️ 部分成功 |
| tax_id | 40.0% | ⚠️ 中等表现 |
| invoice_no | 45.0% | ⚠️ 中等表现 |

### 分析与发现

**成功之处**:
1. ✅ 模型成功学会了JSON输出格式
2. ✅ 没有产生幻觉输出
3. ✅ 损失稳定下降，训练过程稳定
4. ✅ 部分结构化字段有基础识别能力

**问题分析**:
1. ❌ 关键业务字段(商户名、金额)识别失败
2. ❌ 训练轮数可能不足(仅5 epochs)
3. ❌ 可能需要更多训练数据或更长训练时间
4. ❌ 学习率和训练策略可能需要调优

## 改进建议

### 短期改进
1. **增加训练时间**: 将epochs从5增加到10-15
2. **调整学习率**: 尝试更高的学习率或更长的warmup
3. **数据增强**: 检查训练数据质量和多样性

### 中期改进
1. **两阶段训练**:
   - Stage A: 在大规模合成数据上预训练
   - Stage B: 在真实数据上微调
2. ** curriculum learning**: 从简单样本开始，逐步增加难度
3. **损失函数调优**: 针对不同字段设置不同权重

### 长期改进
1. **数据质量提升**: 收集更多真实标注数据
2. **模型架构优化**: 考虑更大的LLM或更好的对齐策略
3. **预训练策略**: 探索更好的多模态预训练方法

## 数据分布

### 中文数据集 (Synthetic)
```
data/processed/synthetic/
├── train.jsonl: 3,998样本
├── val.jsonl: 497样本
└── test.jsonl: 505样本
```

**模板分布**:
- receipt_thermal: ~35%
- vat_general: ~35%
- vat_special: ~15%
- receipt_english: ~15%

### 英文数据集 (CORD)
```
data/processed/cord/
├── train.jsonl: 100样本
├── val.jsonl: 100样本
└── test.jsonl: 100样本
```

## 训练命令

### 训练命令
```bash
python src/train.py \
    --config configs/stage1_chinese.yaml \
    --data data/processed/train_synthetic.jsonl \
    --val-data data/processed/val_synthetic.jsonl
```

### 评估命令
```bash
python src/run_eval.py \
    --checkpoint checkpoints/stage_a_chinese/best_model.pt \
    --data data/processed/synthetic/test.jsonl \
    --max-samples 20 \
    --name "chinese_test" \
    --output evaluation_results/stage_a_chinese
```

## 结论

本次训练实验成功验证了训练流程的可行性，模型能够学习JSON输出格式，但关键信息抽取能力需要进一步改进。建议通过增加训练时间、调优训练策略和提升数据质量来改善模型性能。

---

**实验日期**: 2026-06-06
**实验环境**: AutoDL GPU实例
**模型版本**: Stage A Chinese v1
