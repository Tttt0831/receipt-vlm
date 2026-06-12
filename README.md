# 轻量级中英文票据 VLM

> 从票据图片端到端抽取结构化字段（JSON），无需外部 OCR。围绕三条并行训练路线对比"自建轻量 VLM" vs "微调现成 VLM"。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目概述

从中英文票据/发票图片中直接输出 6 个关键字段的结构化 JSON。项目以**三条并行路线**对比不同的实现方式，使用同一套合成数据与同一套评估指标，便于横向比较。

抽取字段 schema：

```json
{
  "merchant_name": "string | null",    // 商户/销售方名称
  "date":          "YYYY-MM-DD | null",// 开票日期（归一化）
  "total_amount":  "number | null",    // 价税合计/总金额
  "tax_amount":    "number | null",    // 税额
  "tax_id":        "string | null",    // 纳税人识别号 / Tax ID
  "invoice_no":    "string | null"     // 发票号码 / Receipt No.
}
```

---

## 🧭 三条路线

| 路线 | 架构 | 可训练部分 | 配置 / 脚本 | 状态 |
|------|------|-----------|-------------|------|
| **A** | SigLIP2(冻结) + MLP Projection + **Qwen2-1.5B** LoRA | projection + LoRA + 端层 | `configs/route_a.yaml` · `src/train.py` | ✅ 训练+评估完成 |
| **B** | **Qwen2-VL-2B-Instruct**(原生 VLM) + LoRA | LoRA 适配器 | `configs/route_b.yaml` · `src/train_qwen2vl_lora.py` | ✅ 训练+评估完成（效果最好） |
| **C** | SigLIP2(冻结) + MLP Projection + **自制 MiniLLM 214M** | LM 预训练 + projection + 端层 | `configs/route_c.yaml` · `src/pretrain_lm.py` + `src/train.py` | 🚧 WIP（见下方说明） |

三条路线的多模态融合都走 **LLaVA 风格**：文本序列里放 1 个 `<image>` 占位 token，前向时展开成 N 个视觉 patch embedding（路线 A/C 在 [`src/model/vlm.py`](src/model/vlm.py) 手工实现；路线 B 用 Qwen2-VL 原生处理）。

### 路线 A：SigLIP2 + Qwen2-1.5B（自建 VLM）
- 视觉：SigLIP2 NaFlex（冻结），动态分辨率
- 投影：2 层 MLP，`768 → 3072 → 1536`
- 语言：Qwen2-1.5B，LoRA(r=16) + 解冻 embedding/首末层
- tokenizer：Qwen2（vocab 151649）

### 路线 B：Qwen2-VL-2B + LoRA（微调现成 VLM）
- 直接在原生 Qwen2-VL-2B-Instruct 上加 LoRA(r=16)，无需自建视觉融合
- 训练/推理最简单，泛化最好

### 路线 C：SigLIP2 + 自制 MiniLLM 214M（含语言预训练）
这是"从零造一个小语言模型再做 VLM"的探索：
- 语言模型 `MiniLLM`（[`src/model/llm.py`](src/model/llm.py)）：decoder-only，RoPE + MHA + GELU-FFN + Pre-LN + 权重绑定，配置 `h1024 / L16 / heads16 / inter4096 ≈ 214M`
- 自训练 ~12k BPE tokenizer（[`src/train_tokenizer.py`](src/train_tokenizer.py) → `tokenizers/receipt-bpe/`），让小模型 embedding 不爆
- **Stage 0 语言预训练**（[`src/pretrain_lm.py`](src/pretrain_lm.py)）：在 MiniMind 通用中文语料上做 next-token，给随机初始化的 LLM 一个语言先验
- **Stage 1/2**：再用 `src/train.py --init-llm` 载入预训练权重，做 VLM 对齐 + 票据精调

> ⚠️ **当前状态**：tokenizer、配置、预训练/对齐/精调的完整管线已搭好并能跑通，但**全量语言预训练目前不稳定**（损失会塌成一个假的 0、产物退化为随机模型，疑似手写 attention 的 `-inf` 掩码在 bf16 下不稳定 + 学习率偏高）。因此路线 C **暂无可用下游结果**，正在修复。详见 [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)。

---

## 📊 评估结果（500 合成测试集，同一指标口径）

指标定义见 [`src/eval.py`](src/eval.py)：字段 exact-match 准确率、整体 Value Match、字段级 P/R/F1、merchant_name 的 1-NED、JSON 合法率、幻觉率（真值 null 却填值）。

### 字段级准确率

| 字段 | 路线 A | 路线 B |
|------|:-----:|:-----:|
| merchant_name | 2.80% | 23.40% |
| date | 0.00% | 36.60% |
| total_amount | 0.00% | 67.20% |
| tax_amount | 20.60% | 74.60% |
| tax_id | 33.40% | 41.60% |
| invoice_no | 22.20% | 41.60% |

### 总体指标

| 指标 | 路线 A | 路线 B |
|------|:-----:|:-----:|
| Value Match Rate | 13.17% | **47.50%** |
| Precision | 72.22% | 94.70% |
| Recall | 0.99% | **42.27%** |
| F1 Score | 1.96% | **58.45%** |
| 1-NED (merchant_name) | 0.7969 | 0.6555 |
| JSON 合法率 | 100% | 100% |
| 幻觉率 | 2.64% | 16.36% |

> 结果文件：[`evaluation_results/route_a/`](evaluation_results/route_a) · [`evaluation_results/route_b/`](evaluation_results/route_b)

**结论**：
- **路线 B 明显领先**，金额字段（total/tax）表现好；merchant_name 仍是短板。
- **路线 A 效果很差**：JSON 格式 100% 合法，但内容基本是**幻觉**——输出一张"看起来像"的票据而非真正读图（缺乏视觉接地）。Recall≈1% 主要是模型不输出 `<eoa>`、生成停不下来导致；[`src/infer.py`](src/infer.py) 已加入"取第一个配平 JSON 对象"的容错提取。这是自建轻量 VLM 在小数据下的固有难点。

---

## 📈 训练曲线

### 路线 A（`configs/route_a.yaml`，5 epoch，bs=2，单卡 4090）

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.0185 | 0.8931 |
| 2 | 0.8939 | 0.8820 |
| 3 | 0.8825 | 0.8763 |
| 4 | 0.8751 | 0.8729 |
| 5 | 0.8687 | 0.8727 |

val 在第 4–5 epoch 基本收敛（边际收益递减）。注意：低 train/val loss ≠ 高抽取准确率——见上文"缺乏视觉接地"。

### 路线 B（`configs/route_b.yaml`，5 epoch，bs=2，grad_accum=8）
训练 5 epoch 后达到上表评估指标（per-epoch 曲线未单独留存）。

### 路线 C（语言预训练，214M MiniLLM）
- smoke（80 step）正常：loss 9.6 → 7.85
- 全量（102k step）**异常**：loss 9.62 起，~4000 step 后塌成假 0，产物为随机模型 → 待修复

---

## 🗂️ 项目结构

```
receipt-vlm/
├── README.md
├── PLAN.md                      # 当前进度与结论
├── requirements.txt
├── configs/
│   ├── route_a.yaml             # 路线A: SigLIP2 + Qwen2-1.5B LoRA
│   ├── route_b.yaml             # 路线B: Qwen2-VL-2B LoRA
│   └── route_c.yaml             # 路线C: SigLIP2 + 自制 MiniLLM 214M
├── src/
│   ├── train.py                 # 路线 A/C 训练（config 驱动，--init-llm 载入预训练 LLM）
│   ├── train_qwen2vl_lora.py    # 路线 B 训练（支持 --config）
│   ├── pretrain_lm.py           # 路线 C: MiniLLM 语言预训练（Stage 0）
│   ├── train_tokenizer.py       # 路线 C: 训练 ~12k BPE tokenizer
│   ├── run_eval.py / eval.py    # 真实推理评估 + 指标
│   ├── infer.py                 # 推理引擎（自动识别路线 A/B/C）
│   ├── model/                   # vision.py / projection.py / llm.py / vlm.py / llm_hf.py / tokenizer.py
│   ├── data/                    # synth.py（合成票据）/ dataset.py
│   └── utils/                   # normalize.py / preprocessing.py
├── docs/
│   ├── ROUTES.md                # 三路线详解
│   └── KNOWN_ISSUES.md          # 已知问题（含路线A ckpt 膨胀、路线C 预训练不稳定）
├── tokenizers/receipt-bpe/      # 路线C 自训练 BPE（已入库）
└── evaluation_results/          # 路线 A/B 评估指标（小文件，已入库）
```

> 数据集与 checkpoint 体积大，**不入库**（见 `.gitignore`）：`data/`、`checkpoints/`、`*.pt`、`*.log`。

---

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 0) 生成合成数据（3000/500/500）
python -m src.data.synth

# 路线 A：训练 + 评估
python -m src.train --config configs/route_a.yaml --epochs 5 --batch-size 2
python -m src.run_eval --checkpoint checkpoints/route_a/best_model.pt \
    --data data/synthetic/test/test.jsonl --tokenizer Qwen/Qwen2-1.5B \
    --output evaluation_results/route_a

# 路线 B：训练（读 yaml）+ 评估
python src/train_qwen2vl_lora.py --config configs/route_b.yaml
python -m src.run_eval --checkpoint checkpoints/route_b/best_lora \
    --data data/synthetic/test/test.jsonl --output evaluation_results/route_b

# 路线 C：① 训 tokenizer ② 语言预训练 ③ VLM 对齐/精调
python -m src.train_tokenizer --vocab-size 12000
python -m src.pretrain_lm --epochs 3 --batch-size 16          # 注意：当前不稳定，待修
python -m src.train --config configs/route_c.yaml \
    --init-llm checkpoints/route_c/llm_pretrained.pt
```

环境：Python ≥ 3.10，PyTorch 2.x（bf16），单卡 RTX 4090 24GB 即可。

---

## 🧪 评估指标体系

| 指标 | 说明 |
|------|------|
| Field Accuracy | 6 字段各自的 exact-match 准确率 |
| Value Match Rate | 所有字段值的整体匹配率 |
| Precision / Recall / F1 | 字段级，反映漏抽 vs 幻觉 |
| 1-NED | 归一化编辑距离（merchant_name） |
| JSON Validity Rate | 输出通过 schema 校验的比例 |
| Hallucination Rate | 真值为 null 却被模型填值的比例 |

归一化规则（[`src/utils/normalize.py`](src/utils/normalize.py)）：日期统一 `YYYY-MM-DD`；金额去符号/千分位、保留两位；字符串 strip + 全角转半角。

---

## 🙏 致谢

- [SigLIP](https://github.com/google-research/big_vision) — 视觉编码器
- [MiniMind](https://github.com/jingyaogong/minimind) — 自制 LLM 架构参考 + 语言预训练语料
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — 路线 B 基座
- [LLaVA](https://github.com/haotian-liu/LLaVA) — Projection 融合设计

## 📄 许可证

MIT License
