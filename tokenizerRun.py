# Install dependencies
!pip install sentencepiece pandas
!pip install tokenizers  transformers datasets

import sentencepiece as spm
import pandas as pd
from google.colab import drive
import shutil
import os
import unicodedata

input_file = "raw.txt"
output_file = "shakespeare_clean.txt"

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Unicode normalization (NFKC)
text = unicodedata.normalize('NFKC', text)

# Keep line breaks and punctuation (important for poetic structure)
# Optional: strip trailing whitespaces
lines = [line.rstrip() for line in text.splitlines() if line.strip()]

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))

# -------------------------------
# PARAMETERS YOU CAN TWEAK
# -------------------------------
INPUT_FILE = "shakespeare_clean.txt"  # path to your text file
MODEL_PREFIX = "sp_unigram_16k"       # model name prefix
VOCAB_SIZE = 16000                     # increase to 32000, 50000 for larger vocab
MODEL_TYPE = "unigram"                # "unigram" or "bpe"
CHARACTER_COVERAGE = 0.9995            # 0.99 for most English text
INPUT_SENTENCE_SIZE = 1000000          # number of sentences to sample for training
SHUFFLE = True                         # shuffle sentences before training
N_EVAL_LINES = 5000                     # number of lines to use for quick evaluation

# -------------------------------
# STEP 1: Train Tokenizer
# -------------------------------
print("Training Unigram tokenizer...")
spm.SentencePieceTrainer.train(
    input=INPUT_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,
    character_coverage=CHARACTER_COVERAGE,
    input_sentence_size=INPUT_SENTENCE_SIZE,
    shuffle_input_sentence=SHUFFLE
)
print(f"Tokenizer training complete! Files: {MODEL_PREFIX}.model / {MODEL_PREFIX}.vocab")

# -------------------------------
# STEP 2: Load Tokenizer
# -------------------------------
sp_tokenizer = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")

# -------------------------------
# STEP 3: Evaluation Function
# -------------------------------
def evaluate_tokenizer(sp_model, text_path, n_lines=None):
    total_tokens, total_words, recon_errors, max_len = 0, 0, 0, 0
    with open(text_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_lines and i >= n_lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            enc = sp_model.encode(line, out_type=int)
            decoded = sp_model.decode(enc).strip()
            if decoded != line.strip():
                recon_errors += 1
            total_tokens += len(enc)
            total_words += len(line.split())
            max_len = max(max_len, len(enc))
    return {
        "avg_tokens_per_word": total_tokens / (total_words + 1e-9),
        "max_tokens_in_line": max_len,
        "reconstruction_rate": 1 - recon_errors / (i+1e-9)
    }

# -------------------------------
# STEP 4: Run Evaluation
# -------------------------------
results = evaluate_tokenizer(sp_tokenizer, INPUT_FILE, n_lines=N_EVAL_LINES)
print("\nEvaluation Results:")
print(f"Average tokens per word: {results['avg_tokens_per_word']:.4f}")
print(f"Max tokens in a line: {results['max_tokens_in_line']}")
print(f"Reconstruction rate: {results['reconstruction_rate']:.4f}")

# -------------------------------
# STEP 5: Save Tokenizer Locally in Colab
# -------------------------------
print(f"\n✅ Tokenizer saved locally in Colab as: {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")
print("You can download them using:")
print(f"files.download('{MODEL_PREFIX}.model')")
print(f"files.download('{MODEL_PREFIX}.vocab')")
# 1. Average Tokens per Word

# What it tells you: How efficiently the tokenizer compresses text into tokens.

# Lower is better → fewer tokens per word means your model will process longer context for the same input size.

# Compare: If your new tokenizer has lower or similar avg tokens/word than 1.3085, it’s more efficient or equally efficient.

# 2. Max Tokens per Line

# What it tells you: The longest tokenized sequence in a single line.

# Lower is generally better → prevents very long sequences that could slow training or require truncation.

# Compare: If your new tokenizer reduces this from 24, it’s better in sequence efficiency.

# 3. Reconstruction Rate

# What it tells you: How accurately the tokenizer can reproduce the original text after encoding → decoding.

# Higher is better → close to 1 (or 100%) means no loss of information.

# Compare: Your current tokenizer has 0.9606 → new tokenizer should ideally be ≥ 0.9606.

# Summary Table for Evaluation
# Metric	Better Condition
# Avg Tokens per Word	Lower than 1.3085
# Max Tokens per Line	Lower than 24
# Reconstruction Rate	Equal to or higher than 0.9606

# ✅ Rule of Thumb:

# If average tokens per word decreases without dropping reconstruction rate, your tokenizer is better.

# Slight increase in max tokens per line is okay if reconstruction improves or avg tokens per word decreases.

# If reconstruction rate drops significantly, the tokenizer is losing fidelity, even if token counts improve.
