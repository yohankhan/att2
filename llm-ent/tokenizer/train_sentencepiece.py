import sentencepiece as spm
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to a large text file (one document per line).")
parser.add_argument("--vocab_size", type=int, default=32000)
parser.add_argument("--model_type", choices=["bpe","unigram"], default="bpe")
parser.add_argument("--model_prefix", default="tokenizer")
parser.add_argument("--character_coverage", type=float, default=0.9995)
args = parser.parse_args()

os.makedirs("tokenizer_out", exist_ok=True)

spm.SentencePieceTrainer.Train(
    input=args.input,
    model_prefix=os.path.join("tokenizer_out", args.model_prefix),
    vocab_size=args.vocab_size,
    model_type=args.model_type,
    character_coverage=args.character_coverage,
    byte_fallback=True,             # robust OOV behavior
    unk_piece="<unk>",
    pad_id=0,
    bos_id=1,
    eos_id=2
)
print("Wrote:", os.listdir("tokenizer_out"))
