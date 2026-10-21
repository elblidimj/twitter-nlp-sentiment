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


NEG_SCOPE = 2                       # how many words after a negation get _NEG
NEG_STOPS = {"!", "?", "<DOTS>"}    # a negation does not reach past these


def handle_negation(tokens):
    """Appends _NEG to the NEG_SCOPE words that follow a negation term."""
    new_toks = []
    remaining = 0
    for t in tokens:
        base = t
        if base in NEG_STOPS:
            remaining = 0
        elif remaining > 0 and not base.startswith("<"):
            t = t + "_NEG"
            remaining -= 1
        new_toks.append(t)
        if base in NEGATIONS:
            remaining = NEG_SCOPE
    return new_toks


def handle_char_repetition(tokens):
    """'loool' -> 'lool' followed by a <REP_3> tag, so the word stays in one piece."""
    new_toks = []
    for t in tokens:
        runs = [m.group(0) for m in re.finditer(r"([a-z])\1{2,}", t)]
        if runs and not t.startswith("<"):
            new_toks.append(re.sub(r"([a-z])\1{2,}", r"\1\1", t))
            new_toks.append(f"<REP_{max(len(r) for r in runs)}>")
        else:
            new_toks.append(t)
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

    # Heart first, so "<3" is not read as a mouth (":<3") or as a number.
    text = re.sub(r"<3+", " <HEART> ", text)

    # The data is lowercase, so the "big grin" is ":d" (not ":D").
    text = re.sub(r"(?::|;|=)(?:-)?(?:\)|\}|\]|>|d(?!\w))", " <SMILE> ", text)
    text = re.sub(r"(?::|;|=)(?:-)?p(?!\w)", " <TONGUE> ", text)
    # ":/" and ":|" lean negative; "(?!/)" avoids matching the "://" of a URL.
    text = re.sub(r"(?::|;|=)(?:-)?(?:\(|\{|\[|<|\||/(?!/))", " <SAD> ", text)

    text = text.replace('...', ' <DOTS> ')
    text = text.replace('!', ' ! ')
    text = text.replace('?', ' ? ')

    # Numbers are replaced here, before the <REP_n> tags are created (later,
    # on tokens), so the digit inside <REP_n> is never turned into <NUM>.
    text = re.sub(r'\d+', ' <NUM> ', text)

    text = re.sub(r'[^\w\s<!>\?]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    toks = text.split()
    toks = handle_word_repetition(toks)
    toks = handle_char_repetition(toks)
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