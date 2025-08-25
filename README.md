# Enterprise LLM (Reference Project)

## 0) Build env
```bash
docker build -t llm-enterprise .
docker run --gpus all -it --shm-size 16g -p 8000:8000 -v $PWD:/workspace llm-enterprise bash
1) Tokenizer
bash
Copy
Edit
python tokenizer/train_sentencepiece.py --input data/raw.txt \
  --vocab_size 32000 --model_type bpe --model_prefix tokenizer
2) Data
bash
Copy
Edit
python data/scripts/prepare_corpus.py --input data/*.jsonl --output data/clean.txt
python data/scripts/dedup_basic.py --input data/clean.txt --output data/clean_dedup.txt
python data/scripts/pack_tokens.py --input data/clean_dedup.txt \
  --spm_model tokenizer_out/tokenizer.model --seq_len 4096 \
  --output data/packed_pretrain.parquet
3) Pretrain
bash
Copy
Edit
bash scripts/launch_pretrain.sh
4) SFT
Prepare data/sft.jsonl with {"prompt","response"} pairs.

bash
Copy
Edit
bash scripts/launch_sft.sh
5) DPO
Prepare data/dpo.jsonl with {"prompt","chosen","rejected"}.

bash
Copy
Edit
bash scripts/launch_dpo.sh
6) Eval
bash
Copy
Edit
python eval/run_lmeval.py --model checkpoints/dpo --tasks hellaswag,arc_easy,boolq
7) Serve
bash
Copy
Edit
bash serving/run_vllm.sh checkpoints/dpo
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key" -H "Content-Type: application/json" \
  -d '{"model":"checkpoints/dpo","messages":[{"role":"user","content":"Hello!"}]}'
8) Notes on scaling
Prefer DeepSpeed ZeRO-3 or FSDP on clusters.

Use FlashAttention-2 kernels when available (attn_implementation=flash_attention_2).

For long context, tune rope_scaling and train with longer seq_len.

For low-VRAM finetuning, use QLoRA (bitsandbytes 4-bit NF4).

For serving, vLLM (PagedAttention + continuous batching) gives top throughput at scale.

TL;DR build order
1) **Tokenizer** ¿ 2) **Data clean/dedup/pack** ¿ 3) **Pretrain** (FA2 + ZeRO/FSDP, BF16) ¿ 4) **SFT** (QLoRA optional) ¿ 5) **DPO** ¿ 6) **Eval** (lm-eval-harness) ¿ 7) **Serve** (vLLM) ¿ 8) **Monitor/Secure**.

If you paste these files into a fresh repo, you¿ll have a working pipeline you can iterate and scale. When you¿re ready, tell me your **GPU(s)** and **target parameter count** and I¿ll tune the configs (batch sizes, ZeRO/FSDP strategy, seq length, rope scaling) for your hardware.
