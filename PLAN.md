# Receipt-VLM 项目进度

**最后更新**: 2026-06-12

端到端中英文票据字段抽取（6 字段 → JSON）。三条并行路线对比，统一合成数据（3000/500/500）与统一评估口径。

---

## 路线总览

| 路线 | 架构 | 状态 | 关键指标（500 测试集） |
|------|------|------|------------------------|
| **A** | SigLIP2(冻结) + MLP + Qwen2-1.5B LoRA | ✅ 完成 | Value Match 13.2% / F1 2.0%（差，缺视觉接地） |
| **B** | Qwen2-VL-2B-Instruct + LoRA | ✅ 完成 | Value Match 47.5% / F1 58.5%（最好） |
| **C** | SigLIP2(冻结) + MLP + 自制 MiniLLM 214M | 🚧 WIP | 语言预训练不稳定，下游未做 |

---

## 路线 A 结果（5 epoch, val 0.873）

| 字段 | 准确率 | | 总体 | 得分 |
|------|:------:|---|------|:----:|
| merchant_name | 2.80% | | Value Match | 13.17% |
| date | 0.00% | | Precision | 72.22% |
| total_amount | 0.00% | | Recall | 0.99% |
| tax_amount | 20.60% | | F1 | 1.96% |
| tax_id | 33.40% | | 1-NED(merchant) | 0.7969 |
| invoice_no | 22.20% | | JSON 合法率 | 100% |

**核心问题：缺乏视觉接地**。模型学会了输出格式完美的 JSON，但内容是**幻觉**（编一张像样的票），不是真正读图。低 train/val loss 与高准确率脱节。另有"不输出 `<eoa>`、生成停不下来"导致 Recall≈1%，已在 `src/infer.py` 加入配平 JSON 提取容错。

## 路线 B 结果（5 epoch）

| 字段 | 准确率 | | 总体 | 得分 |
|------|:------:|---|------|:----:|
| merchant_name | 23.40% | | Value Match | 47.50% |
| date | 36.60% | | Precision | 94.70% |
| total_amount | 67.20% | | Recall | 42.27% |
| tax_amount | 74.60% | | F1 | 58.45% |
| tax_id | 41.60% | | 1-NED(merchant) | 0.6555 |
| invoice_no | 41.60% | | 幻觉率 | 16.36% |

金额字段明显优于文本字段，merchant_name 仍是主要短板。原生 VLM 的视觉接地能力是路线 A 不具备的。

## 路线 C 进度（自制 MiniLLM 214M）

已完成：
- 自训练 ~12k BPE tokenizer（`tokenizers/receipt-bpe/`，self-check 通过）
- MiniLLM 配置 `h1024/L16/heads16/inter4096 = 213.8M`
- 完整管线：语言预训练（Stage 0）→ `--init-llm` 载入 → VLM 对齐/精调（Stage 1/2），smoke 全程跑通
- 修正 `train.py` 的 `llm_num_heads` 硬编码 bug（原来恒为 8，导致与预训练 head 数不一致）

**未完成 / 阻塞**：全量语言预训练发散——损失 9.62 起、约 4000 step 后塌成假的 0，保存的 LM 经核验是**随机模型**（训练数据上 loss≈ln(12000)、生成死循环）。疑似手写 attention 的 `-inf` 掩码在 bf16 下不稳定 + LR 偏高。修复方向见 [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)。

---

## 下一步

1. **修路线 C 预训练**：loss 用 fp32 计算、排查 attention softmax 在 bf16 的 NaN、降 LR / 加 warmup / 收紧 grad clip，重跑出一个真能生成中文的 LM，再做 Stage 1/2 评估。
2. **路线 A 接地**：尝试解冻部分 vision / 增大 projection / 更长训练，缓解幻觉。
3. 路线 A checkpoint 体积优化（见 KNOWN_ISSUES）。

## 相关文档
- [README](README.md) · [路线详解](docs/ROUTES.md) · [已知问题](docs/KNOWN_ISSUES.md)
