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
| **A** | SigLIP2(冻结→解冻消融) + MLP Projection + **Qwen2-1.5B** LoRA | projection + LoRA + 解冻视觉顶层 | `configs/route_a*.yaml` · `src/train.py` | ✅ 完成（含2层/4层解冻消融） |
| **B** | **Qwen2-VL-2B-Instruct**(原生 VLM) + LoRA | LoRA 适配器 | `configs/route_b.yaml` · `src/train_qwen2vl_lora.py` | ✅ 完成（效果最好） |
| **C** | SigLIP2(冻结) + MLP Projection + **自制 MiniLLM 214M** | LM 预训练 + projection + 端层 | `configs/route_c.yaml` · `src/pretrain_lm.py` + `src/train.py` | ✅ 完成（35M-token预训练评估见 PLAN.md） |

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
- 语言模型 `MiniLLM`（[`src/model/llm.py`](src/model/llm.py)）：decoder-only，**对齐 MiniMind**——RoPE + SDPA 注意力 + RMSNorm + SwiGLU + 权重绑定，配置 `h1024 / L16 / heads16 / inter2752 ≈ 214M`
- 自训练 ~12k BPE tokenizer（[`src/train_tokenizer.py`](src/train_tokenizer.py) → `tokenizers/receipt-bpe/`），让小模型 embedding 不爆
- **Stage 0 语言预训练**（[`src/pretrain_lm.py`](src/pretrain_lm.py)）：在 MiniMind 通用中文语料上做 next-token，给随机初始化的 LLM 一个语言先验
- **Stage 1/2**：再用 `src/train.py --init-llm` 载入预训练权重，做 VLM 对齐 + 票据精调

> ✅ **路线 C 已完成**：214M MiniLLM 经 35M-token 中文语料预训练 + 8 epoch VLM 微调，F1=45.8%，超越路线 A 全部变体。详细训练过程、续写测试、三路线对比见 [PLAN.md](PLAN.md)。

---

## 📊 评估结果（500 合成测试集，同一指标口径）

指标定义见 [`src/eval.py`](src/eval.py)：字段 exact-match 准确率、整体 Value Match、字段级 P/R/F1、merchant_name 的 1-NED、JSON 合法率、幻觉率（真值 null 却填值）。

### 三路线 + Route A 消融 完整对比

| 字段 | 路线A(冻结) | A(2层JFT) | **A(4层JFT)** | 路线B | **路线C** |
|------|:----------:|:---------:|:------------:|:-----:|:--------:|
| merchant_name | 2.80% | 38.40% | **41.00%** | 23.40% | **41.60%** |
| date | 0.00% | 0.40% | **0.60%** | 36.60% | **52.40%** |
| total_amount | 0.00% | 0.00% | 0.00% | **67.20%** | 43.80% |
| tax_amount | 20.60% | 20.60% | **21.60%** | **74.60%** | 41.20% |
| tax_id | 33.40% | 33.80% | **35.00%** | **41.60%** | 34.60% |
| invoice_no | 22.20% | 12.00% | **13.00%** | **41.60%** | 9.80% |

| 总体 | 路线A(冻结) | A(2层JFT) | **A(4层JFT)** | 路线B | **路线C** |
|------|:----------:|:---------:|:------------:|:-----:|:--------:|
| Value Match | 13.17% | 17.53% | **18.53%** | **47.50%** | 37.23% |
| Precision | 72.22% | 77.32% | 75.09% | 94.70% | **89.95%** |
| Recall | 0.99% | 7.94% | **8.49%** | **42.27%** | 30.76% |
| **F1** | 1.96% | 14.39% | **15.26%** | **58.45%** | 45.84% |
| 1-NED(merchant)↓ | 0.7969 | 0.4746 | **0.4493** | 0.6555 | **0.4435** |
| JSON 合法率 | 100% | 100% | 100% | 100% | 100% |
| 幻觉率 | 2.64% | 16.09% | 17.85% | 16.36% | 21.76% |

> 结果文件：[`evaluation_results/route_a/`](evaluation_results/route_a) · [`evaluation_results/route_a_jointft/`](evaluation_results/route_a_jointft) · [`evaluation_results/route_a_jointft_4l/`](evaluation_results/route_a_jointft_4l) · [`evaluation_results/route_b/`](evaluation_results/route_b) · [`evaluation_results/route_c/`](evaluation_results/route_c)

**核心结论**：
- **路线 B 总体领先**（F1 58.5%），金额字段优势明显；merchant_name 是短板
- **路线 C 性价比最优**：214M 小模型 F1 达 45.8%，文本字段（merchant_name/date）甚至超过路线 B
- **路线 A 需要视觉解冻**：冻结视觉 → 完全幻觉（F1 1.96%），解冻2层 → F1 14.4%，4层 → F1 15.3%
- **路线 A 的瓶颈不在视觉层数**：4层 vs 2层仅 +0.9pp F1，total_amount 始终为 0%

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

**路线 A 联合微调 2层**（解冻视觉顶部 2 层，[`configs/route_a_jointft.yaml`](configs/route_a_jointft.yaml)，5 epoch）：

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.0670 | 0.8944 |
| 2 | 0.8863 | 0.8704 |
| 3 | 0.8660 | 0.8517 |
| 4 | 0.8513 | 0.8471 |
| 5 | 0.8452 | **0.8469** |

解冻视觉后 val loss（0.847）低于基线（0.873），且抽取指标显著改善（见上方消融表）。

**路线 A 联合微调 4层**（解冻视觉顶部 4 层，[`configs/route_a_jointft_4l.yaml`](configs/route_a_jointft_4l.yaml)，5 epoch，从 epoch 2 checkpoint 恢复）：

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.0607 | 0.8897 |
| 2 | 0.8918 | 0.8656 |
| 3 | 0.8744 | 0.8584 |
| 4 | 0.8699 | **0.8574** |
| 5 | 0.8688 | 0.8576 |

4层 val loss（0.8574）略高于 2 层版本（0.8469），但抽取指标仍有小幅提升（F1 14.4% → 15.3%）。视觉层数增加带来的边际收益递减。

### 路线 B（`configs/route_b.yaml`，5 epoch，bs=2，grad_accum=8）
训练 5 epoch 后达到上表评估指标（per-epoch 曲线未单独留存）。

### 路线 C（214M MiniLLM，35M-token 中文预训练 + 8 epoch VLM）

**语言预训练**（2 epoch, MiniMind 中文语料）：
- Loss: 9.60 → 2.08（ppl 14707 → 8.0），无 NaN/发散
- 预训练后 next-token 准确率 42.9%，能生成连贯中文，无重复

**VLM 训练**（8 epoch, 3000/500 合成票据）：

| Epoch | Train Loss | Val Loss |
|:-----:|:----------:|:--------:|
| 1 | 1.5156 | 1.0205 |
| 2 | 1.0172 | 0.9826 |
| 3 | 0.9826 | 0.9548 |
| 4 | 0.9191 | 0.8848 |
| 5 | 0.7947 | 0.7424 |
| 6 | 0.6645 | 0.6807 |
| 7 | 0.5905 | **0.6763** |
| 8 | 0.5567 | 0.6789 |

最佳 val loss 0.6763（Epoch 7），远优于路线 A 基线 0.873。仅 214M 参数达到 F1=45.8%，超越路线 A 的 1.5B 参数方案。详见 [PLAN.md](PLAN.md)。

---

## 🗂️ 项目结构

```
receipt-vlm/
├── README.md
├── PLAN.md                      # 当前进度与结论
├── requirements.txt
├── configs/
│   ├── route_a.yaml               # 路线A: SigLIP2(冻结) + Qwen2-1.5B LoRA
│   ├── route_a_jointft.yaml        # 路线A消融: 视觉解冻2层 + 联合微调
│   ├── route_a_jointft_4l.yaml     # 路线A消融: 视觉解冻4层 + 联合微调
│   ├── route_b.yaml                # 路线B: Qwen2-VL-2B LoRA
│   └── route_c.yaml                # 路线C: SigLIP2 + 自制 MiniLLM 214M
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
└── evaluation_results/          # 路线 A/B/C 评估指标（小文件，已入库）
    ├── route_a/ route_a_jointft/ route_a_jointft_4l/
    ├── route_b/
    └── route_c/
```

> 数据集与 checkpoint 体积大，**不入库**（见 `.gitignore`）：`data/`、`checkpoints/`、`*.pt`、`*.log`。

---

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 0) 生成合成数据（3000/500/500）
python -m src.data.synth

# 路线 A：训练 + 评估（含联合微调消融）
python -m src.train --config configs/route_a.yaml --epochs 5 --batch-size 2
python -m src.train --config configs/route_a_jointft.yaml --epochs 5 --batch-size 2
python -m src.train --config configs/route_a_jointft_4l.yaml --epochs 5 --batch-size 2
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
