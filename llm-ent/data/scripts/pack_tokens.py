"""
Tokenize and pack into fixed-length sequences (efficient packing for pretraining).
Outputs Arrow dataset with columns: input_ids, attention_mask.
"""
import argparse, pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)     # deduped txt (one doc/line)
parser.add_argument("--spm_model", required=True) # tokenizer_out/tokenizer.model
parser.add_argument("--seq_len", type=int, default=4096)
parser.add_argument("--output", required=True)    # out parquet/dir
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=None,
    use_fast=True,
    tokenizer_file=None,
)
# Load SentencePiece directly
tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", use_fast=True)  # stub to get class
tok.backend_tokenizer.model = tok.backend_tokenizer.model.__class__.from_file(args.spm_model)  # low-level swap
tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

seqs = []
with open(args.input, "r", encoding="utf-8") as f:
    buffer = []
    for line in tqdm(f, desc="Tokenizing"):
        ids = tok(line.strip(), add_special_tokens=False)["input_ids"]
        buffer.extend(ids + [tok.eos_token_id])
        # pack
        while len(buffer) >= args.seq_len:
            seq = buffer[:args.seq_len]
            buffer = buffer[args.seq_len:]
            attn = [1]*len(seq)
            seqs.append((seq, attn))
# write parquet
table = pa.table({
    "input_ids": [pa.array(x[0], type=pa.int32()) for x in seqs],
    "attention_mask": [pa.array(x[1], type=pa.int8()) for x in seqs],
})
pq.write_table(table, args.output)
print("Wrote:", args.output, "rows:", table.num_rows)
