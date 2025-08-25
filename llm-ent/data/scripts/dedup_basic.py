"""
Very simple near-duplicate filter via hashing of normalized lines.
For production, consider MinHash/LSH plus fuzzy heuristics.
"""
import argparse, hashlib

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

seen = set()
w = open(args.output, "w", encoding="utf-8")
with open(args.input, "r", encoding="utf-8") as f:
    for line in f:
        h = hashlib.blake2b(line.strip().encode("utf-8"), digest_size=16).hexdigest()
        if h in seen: 
            continue
        seen.add(h)
        w.write(line)
w.close()
