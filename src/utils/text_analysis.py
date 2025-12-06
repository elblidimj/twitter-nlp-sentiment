# src/utils/text_analysis.py

import collections
import matplotlib.pyplot as plt

def get_word_frequencies(tweets, min_len=1, stopwords=None):
    counter = collections.Counter()
    for t in tweets:
        words = t.strip().split()
        for w in words:
            if len(w) < min_len:
                continue
            if stopwords is not None and w in stopwords:
                continue
            counter[w] += 1
    return counter


def plot_top_words(counter, top_k=20, title="Top words"):
    most_common = counter.most_common(top_k)
    if not most_common:
        print("No words to plot.")
        return
    words, freqs = zip(*most_common)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(words)), freqs)
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()
