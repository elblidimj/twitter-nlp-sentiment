import os
import re

FILES = [
    ("twitter-datasets/train_pos_full_clean.txt", "twitter-datasets/train_pos_full_processed.txt"),
    ("twitter-datasets/train_neg_full_clean.txt", "twitter-datasets/train_neg_full_processed.txt"),
    ("twitter-datasets/train_pos_clean.txt", "twitter-datasets/train_pos_processed.txt"),
    ("twitter-datasets/train_neg_clean.txt", "twitter-datasets/train_neg_processed.txt"),
    ("twitter-datasets/test_data.txt", "twitter-datasets/test_data_processed.txt"),
]

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not",
    "i'm": "i am", "he's": "he is", "she's": "she is", "it's": "it is",
    "that's": "that is", "what's": "what is", "where's": "where is",
    "there's": "there is", "who's": "who is", "how's": "how is",
    "let's": "let us", "c'mon": "come on",
    "n't": " not", "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am"
}

NEGATIONS = {"not", "no", "never", "nor", "nothing", "nowhere", "cannot", "none", "neither"}


def split_hashtag(match):
    hashtag = match.group(0)[1:] # Remove the #
    if hashtag.isupper(): return hashtag 
    # CamelCase split: #GoodDay -> Good Day
    return " ".join(re.findall(r'[a-zA-Z][^A-Z]*', hashtag))
def handle_negation(tokens):
    """Appends _NEG to the word immediately following a negation term."""
    new_toks = []
    neg_active = False
    for t in tokens:
        if neg_active:
            if not t.startswith("<"): # Don't tag special tokens like <NUM>
                t = t + "_NEG"
            neg_active = False
        new_toks.append(t)
        if t in NEGATIONS or t.endswith("not"):
            neg_active = True
    return new_toks

def handle_word_repetition(tokens):
    """Replaces repeated words: 'very very very' -> 'very <REPEAT_3>'"""
    if not tokens: return []
    new_toks = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        count = 1
        while i + 1 < len(tokens) and tokens[i+1] == word:
            count += 1
            i += 1
        new_toks.append(word)
        if count > 1:
            new_toks.append(f"<REPEAT_{count}>")
        i += 1
    return new_toks

def augment_line(line):
    text = line.strip()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    text = re.sub(r"(?::|;|=)(?:-)?(?:\)|\}|\]|>|D)", " <SMILE> ", text)
    text = re.sub(r"(?::|;|=)(?:-)?(?:\(|\{|\[|<)", " <SAD> ", text)

    #text = re.sub(r"#\w+", split_hashtag, text)

    text = text.replace('...', ' <DOTS> ')
    text = text.replace('!', ' ! ')
    text = text.replace('?', ' ? ')

    def char_rep(match):
        char = match.group(1)
        full_seq = match.group(0)
        return f"{char}{char} <REP_{len(full_seq)}> "
    text = re.sub(r"([a-z])\1{2,}", char_rep, text)

    text = re.sub(r'\d+', ' <NUM> ', text)

    text = re.sub(r'[^\w\s<!>\?]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    toks = text.split()
    toks = handle_word_repetition(toks)
    toks = handle_negation(toks)
    
    return ' '.join(toks)

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