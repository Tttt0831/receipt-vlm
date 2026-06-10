# 数据分布与使用指南

## 📊 当前数据分布

### 两套独立的数据集

#### 🇨🇳 中文数据集 (Synthetic)
```
data/processed/synthetic/
├── train.jsonl    3,998 样本 (80%)
├── val.jsonl        497 样本 (10%)
└── test.jsonl       505 样本 (10%)
```

**特点：**
- 合成数据，量较大
- 中文收据（公司名称、地址等均为中文）
- 4种模板分布均衡：receipt_thermal, vat_general, vat_special, receipt_english
- 字段完整度高：merchant_name, date, total_amount 100%
- 可选字段：tax_amount (80%), tax_id (70%), invoice_no (76%)

#### 🇬🇧 英文数据集 (CORD)
```
data/processed/cord/
├── train.jsonl     100 样本 (33%)
├── val.jsonl       100 样本 (33%)
└── test.jsonl      100 样本 (33%)
```

**特点：**
- 真实公开数据集
- 英文收据
- 数据量小但质量高
- 适合评估模型在小数据上的学习能力

## 🎯 训练配置选择

### 方案一：中文训练
```bash
# 配置文件: configs/stage1_chinese.yaml
python -m src.train --config configs/stage1_chinese.yaml
```

**优势：**
- 训练数据充足 (3,998 样本)
- 语言一致性（训练/验证/测试都是中文）
- 适合中文收据理解任务

### 方案二：英文训练
```bash
# 配置文件: configs/stage1_english.yaml
python -m src.train --config configs/stage1_english.yaml
```

**优势：**
- 真实数据质量高
- 国际化场景适用
- 可作为基线对比

## ⚠️ 重要提示

**不要混用数据！**

❌ **错误做法：**
```yaml
# 这样会导致语言不匹配
train_path: "data/processed/synthetic/train.jsonl"  # 中文
val_path: "data/processed/cord/val.jsonl"            # 英文
```

✅ **正确做法：**
```yaml
# 方案1：全中文
train_path: "data/processed/synthetic/train.jsonl"
val_path: "data/processed/synthetic/val.jsonl"

# 方案2：全英文
train_path: "data/processed/cord/train.jsonl"
val_path: "data/processed/cord/val.jsonl"
```

## 📁 旧数据处理

```
data/processed/
├── train.jsonl    # 旧文件 (5,100 行 = synthetic 5000 + cord 100)
├── val.jsonl      # 旧文件 (100 行 = cord val)
└── test.jsonl     # 旧文件 (100 行 = cord test)
```

这些旧文件可以删除或保留作为备份：
```bash
# 备份旧文件
mv data/processed/train.jsonl data/processed/train.jsonl.bak
mv data/processed/val.jsonl data/processed/val.jsonl.bak
mv data/processed/test.jsonl data/processed/test.jsonl.bak
```

## 🔧 数据处理脚本

- `scripts/split_synthetic_data.py` - 分割合成数据为 train/val/test
- `scripts/create_cord_splits.py` - 创建 CORD 数据集分割
- `scripts/merge_datasets.py` - 旧的混合脚本（不推荐使用）
