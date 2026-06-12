"""
训练本项目自有的小词表 BPE tokenizer（~12k），用于自制 MiniLLM 路线。

语料：
  1) MiniMind 通用中文语料 data/corpus/pretrain_t2t_mini.jsonl 的 "text" 字段
  2) 票据域文本：把 data/synthetic/train 的 prompt + 紧凑 JSON 拼进去，
     让 tokenizer 学到 JSON 标点与字段名。

产物（HF 可直接 AutoTokenizer.from_pretrained 加载）：
  tokenizers/receipt-bpe/{tokenizer.json, tokenizer_config.json, special_tokens_map.json}

用法：
  python -m src.train_tokenizer --vocab-size 12000 \
      --corpus data/corpus/pretrain_t2t_mini.jsonl \
      --receipt data/synthetic/train/train.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

SPECIAL = ["<unk>", "<pad>", "</s>", "<image>", "<boa>", "<eoa>"]


def corpus_iter(corpus_path, receipt_path, max_lines):
    """逐条产出训练文本。"""
    n = 0
    if corpus_path and Path(corpus_path).exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_lines and n >= max_lines:
                    break
                try:
                    t = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if t:
                    yield t
                    n += 1
    # 票据域文本：prompt + 紧凑 JSON（多写几遍以提升 JSON 标点/字段名的合并优先级）
    if receipt_path and Path(receipt_path).exists():
        with open(receipt_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = d.get("prompt", "")
                tgt = d.get("target_json")
                js = json.dumps(tgt, ensure_ascii=False, separators=(",", ":")) if tgt is not None else ""
                if prompt or js:
                    yield (prompt + "\n" + js).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=12000)
    ap.add_argument("--corpus", default="data/corpus/pretrain_t2t_mini.jsonl")
    ap.add_argument("--receipt", default="data/synthetic/train/train.jsonl")
    ap.add_argument("--max-lines", type=int, default=300000,
                    help="语料最多取多少行（控制训练时长；mini 约 70 万行）")
    ap.add_argument("--out", default="tokenizers/receipt-bpe")
    args = ap.parse_args()

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"训练 BPE: vocab={args.vocab_size}, corpus={args.corpus} (<= {args.max_lines} 行) + receipt={args.receipt}")
    tok.train_from_iterator(corpus_iter(args.corpus, args.receipt, args.max_lines), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="</s>",
        additional_special_tokens=["<image>", "<boa>", "<eoa>"],
    )
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out))
    print(f"✓ 已保存 tokenizer 到 {out} | 实际 vocab={len(fast)}")

    # 自检
    s = '{"merchant_name":"北京科技有限公司","total_amount":1250.00} <eoa>'
    ids = fast(s)["input_ids"]
    print(f"自检编码长度={len(ids)} 解码={fast.decode(ids)}")


if __name__ == "__main__":
    main()
