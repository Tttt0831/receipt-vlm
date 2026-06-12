# 训练路线详解

三条并行路线，统一合成数据（3000/500/500）与统一评估口径（`src/eval.py`）。

| 路线 | 架构 | 模型规模 | 显存 | 配置 / 脚本 | 状态 |
|------|------|----------|------|-------------|------|
| **A** | SigLIP2 + MLP + Qwen2-1.5B LoRA | 1.5B base + LoRA | ~20GB(bs2) | `configs/route_a.yaml` · `src/train.py` | ✅ |
| **B** | Qwen2-VL-2B-Instruct LoRA | 2B base + LoRA | ~20-22GB | `configs/route_b.yaml` · `src/train_qwen2vl_lora.py` | ✅ |
| **C** | SigLIP2 + MLP + 自制 MiniLLM 214M | 214M(自训) | <24GB | `configs/route_c.yaml` · `src/pretrain_lm.py`+`src/train.py` | 🚧 WIP |

多模态融合统一走 **LLaVA 风格**：文本里放 1 个 `<image>` 占位 token，前向时展开成 N 个视觉 patch embedding（路线 A/C 见 `src/model/vlm.py:_merge_multimodal`；路线 B 用 Qwen2-VL 原生）。

---

## 路线 A：SigLIP2 + Qwen2-1.5B（自建 VLM）

```
图片 → SigLIP2 NaFlex(冻结) → MLP Projection(可训练) → Qwen2-1.5B + LoRA → JSON
```

| 参数 | 值 |
|------|------|
| vision_model | google/siglip2-base-patch16-naflex |
| hf_model | Qwen/Qwen2-1.5B |
| tokenizer | Qwen2（vocab 151649） |
| llm_type | hf_lora（r=16, alpha=32） |
| projection | 768 → 3072 → 1536 |
| freeze_strategy | projection_llm_ends（projection + LoRA + embedding + 首末层） |
| batch_size | 2（bs=4 在单卡 24GB 会 OOM） |

```bash
python -m src.train --config configs/route_a.yaml --epochs 5 --batch-size 2
python -m src.run_eval --checkpoint checkpoints/route_a/best_model.pt \
    --data data/synthetic/test/test.jsonl --tokenizer Qwen/Qwen2-1.5B \
    --output evaluation_results/route_a
```

**结论**：JSON 格式 100% 合法，但**缺乏视觉接地**，内容多为幻觉 → 字段准确率低（Value Match 13.2%）。

**变体 `configs/route_a_jointft.yaml`（视觉+投影联合微调）**：解冻 SigLIP2 顶部 2 层 + post_layernorm（视觉用 1/10 小 lr），与 projection/LoRA 联合训练。merchant_name 2.8%→**38.4%**、F1 1.96%→**14.39%**、Value Match→17.53%，证明解冻视觉能增强接地（金额/日期仍待解决）。基线配置 `route_a.yaml` 保留作对比。

---

## 路线 B：Qwen2-VL-2B + LoRA（微调现成 VLM）

```
图片 + 文本提示 → Qwen2-VL-2B-Instruct(原生 VLM) → LoRA → JSON
```

| 参数 | 值 |
|------|------|
| model | Qwen/Qwen2-VL-2B-Instruct |
| lora | r=16, alpha=32, dropout=0.05 |
| batch_size / grad_accum | 2 / 8 |
| learning_rate | 2e-4 |
| epochs | 5 |

```bash
python src/train_qwen2vl_lora.py --config configs/route_b.yaml
python -m src.run_eval --checkpoint checkpoints/route_b/best_lora \
    --data data/synthetic/test/test.jsonl --output evaluation_results/route_b
```

**结论**：三条路线里**效果最好**（Value Match 47.5%，F1 58.5%）。原生 VLM 的视觉接地与泛化能力强；代价是模型大、推理慢。

---

## 路线 C：SigLIP2 + 自制 MiniLLM 214M（含语言预训练）

"从零造一个小 LLM 再做 VLM"。三阶段：

```
Stage 0  语言预训练    src/pretrain_lm.py   MiniMind 语料上 next-token，给 LLM 语言先验
Stage 1  模态对齐      src/train.py         冻结 LLM+Vision，只训 projection
Stage 2  下游精调      src/train.py         解冻端层，训票据抽取
```

| 组件 | 配置 |
|------|------|
| MiniLLM | `h1024 / L16 / heads16 / inter2752 ≈ 214.7M`（`src/model/llm.py`） |
| 架构 | **对齐 MiniMind**：RoPE + SDPA 注意力 + RMSNorm + SwiGLU + 权重绑定 |
| tokenizer | 自训练 ~12k BPE（`src/train_tokenizer.py` → `tokenizers/receipt-bpe/`） |
| 预训练语料 | MiniMind `pretrain_t2t_mini.jsonl`（通用中文，~280M token） |

```bash
python -m src.train_tokenizer --vocab-size 12000
python -m src.pretrain_lm --epochs 3 --batch-size 16     # ⚠️ 当前不稳定
python -m src.train --config configs/route_c.yaml \
    --init-llm checkpoints/route_c/llm_pretrained.pt
```

**为什么需要语言预训练**：随机初始化的 LLM 没有任何语言能力；路线 A 能跑是因为 Qwen2 自带预训练先验。要用自制 LLM 就必须先补上这一步。3000 条票据远不够，需 GB 级通用语料（参考 MiniMind 用序列猴子/匠数等约 0.7–4B token）。

**当前状态**：早期全量预训练发散（loss 假塌成 0、产物为随机模型）。已**对齐 MiniMind 重写** `MiniLLM`（SDPA/RMSNorm/SwiGLU + fp32 loss + 降 lr）修复数值稳定性，forward/backward 已验证，**待重训验证**。详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。

---

## 选择建议

- **要能用的结果**：路线 B。
- **要轻量/可控架构**：路线 A（需解决接地）或路线 C（需先修预训练）。
- **数据划分**：三条路线共用 `data/synthetic/{train,val,test}/*.jsonl`（3000/500/500）。
