import os

FILES = [
    ("twitter-datasets/train_pos_full.txt", "twitter-datasets/train_pos_full_ngrams.txt"),
    ("twitter-datasets/train_neg_full.txt", "twitter-datasets/train_neg_full_ngrams.txt"),
    ("twitter-datasets/train_pos.txt", "twitter-datasets/train_pos_ngrams.txt"),
    ("twitter-datasets/train_neg.txt", "twitter-datasets/train_neg_ngrams.txt"),
]

def ngrams(tokens, n):
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def augment_line(line):
    # basic normalization: lowercase, split on whitespace
    toks = line.strip().lower().split()
    out = []
    # keep unigrams
    out.extend(toks)
    # add bigrams
    out.extend(ngrams(toks, 2))
    # add trigrams
    out.extend(ngrams(toks, 3))
    return ' '.join(out)

def main():
    for inp, out in FILES:
        if not os.path.exists(inp):
            print(f"Warning: {inp} not found, skipping.")
            continue
        with open(inp, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                augmented = augment_line(line)
                fout.write(augmented + "\n")
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
