"""
Normalize input text (UTF-8 NFC), strip control chars, collapse whitespace.
Input: one JSONL or TXT per line; adapt loader as needed.
"""
import argparse, json, unicodedata, sys, re
parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs="+", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\t", " ")
    s = re.sub(r"[ \u00A0]+", " ", s)
    s = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", s)
    return s.strip()

with open(args.output, "w", encoding="utf-8") as out:
    for path in args.input:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    text = line
                t = norm(text)
                if len(t) >= 32:   # drop ultra-short
                    out.write(t + "\n")
