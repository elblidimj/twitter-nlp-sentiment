import os

FILES = [
    ("twitter-datasets/train_pos_full.txt", "twitter-datasets/train_pos_full_clean.txt"),
    ("twitter-datasets/train_neg_full.txt", "twitter-datasets/train_neg_full_clean.txt"),
    ("twitter-datasets/train_pos.txt", "twitter-datasets/train_pos_clean.txt"),
    ("twitter-datasets/train_neg.txt", "twitter-datasets/train_neg_clean.txt"),
]

def dedup_file(in_path, out_path):
    seen = set()
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if line not in seen:
                seen.add(line)
                fout.write(line + "\n")

    print(f"Cleaned: {in_path} → {out_path} ({len(seen)} unique tweets)")

def main():
    for infile, outfile in FILES:
        dedup_file(infile, outfile)

if __name__ == "__main__":
    main()
