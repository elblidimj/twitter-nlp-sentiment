# Emoji Polarity Classification on Twitter

> Large-scale NLP study comparing lightweight models vs. transformers on 2.5M tweets — EPFL CS-433 Machine Learning Project (2025/2026)

## Key Results

| Model | Accuracy | F1-score | Parameters | Train Time |
|-------|----------|----------|------------|------------|
| Logistic Regression | 59.1% | 0.596 | — | — |
| CNN | 85.24% | 0.850 | 8.4M | 25 min/epoch |
| **BiLSTM** | **85.8%** | **0.861** | **4.2M** | **30 min/epoch** |
| BERTweet | 92.1% | 0.920 | 125M | 95 min/epoch |

**Key finding:** BiLSTM reaches 95% of BERTweet's performance with **30× fewer parameters** and **3× faster training** — demonstrating that domain-specific preprocessing and embedding adaptation can close most of the gap to large transformers.

**Ablation — impact of preprocessing:**

| Configuration | Accuracy | F1-score |
|---------------|----------|----------|
| BiLSTM without preprocessing | 73.1% | 0.755 |
| BiLSTM with preprocessing | 85.8% | 0.861 |

Preprocessing alone accounts for a **+12.7% accuracy gain**.

## Report
Full project report available [here](report.pdf).

## Problem

Binary classification of tweets labeled by emoji polarity (`:)` → positive, `:(` → negative). The emoticons are removed from the text before training, making the task non-trivial due to Twitter's noisy language (slang, sarcasm, creative spelling, negation).

## Stack

Python · PyTorch · HuggingFace Transformers · BERTweet · NumPy

---

## Project Structure

The project is organized into several folders and Python files, each with a specific responsibility:

```text
project/
├── preprocessing/
│   ├── clean_and_dedup.py        # Remove strictly identical tweets (deduplicated *_clean.txt)
│   ├── preprocess.py             # Twitter-specific text normalization and cleaning
│   ├── build_vocab.sh            # Build token frequency list from cleaned data
│   ├── cut_vocab.sh              # Remove rare words (minimum frequency = 3)
│   ├── pickle_vocab.py           # Serialize vocabulary mapping (vocab.pkl)
│   ├── cooc.py                   # Build word co-occurrence matrix (cooc.pkl)
│   ├── glove_pretrained.py       # Load pretrained Twitter GloVe embeddings
│   └── glove_trained.py          # Train GloVe embeddings from scratch on the dataset
│
├── src/
│   ├── datasets/
│   │   ├── twitter.py            # Dataset loading utilities (full / non-full splits)
│   │   └── loader.py             # Shared dataset loading helpers
│   │
│   ├── model/
│   │   ├── bertweet.py           # BERTweet-based model
│   │   ├── cnn.py                # CNN baseline model
│   │   ├── bilstm.py             # BiLSTM model
│   │   └── logreg.py             # Logistic regression baseline
│   │
│   ├── trainer/
│   │   ├── bertweet_train.py     # Training loop for BERTweet
│   │   ├── trainer_cnn.py        # Training loop for CNN
│   │   ├── trainer_bilstm.py     # Training loop for BiLSTM
│   │   ├── validation.py         # Evaluation utilities (accuracy, F1-score)
│   │   └── tuning_base.py        # Hyperparameter tuning utilities
│   │
│   ├── transforms/
│   │   └── text_embeddings.py    # Embedding preparation for word-based models
│   │
│   └── utils/
│       ├── io_utils.py           # File I/O helpers
│       └── text_analysis.py      # Text analysis utilities
│
├── ethics_oracle_eval.py          # Ethics: oracle sentiment vs emoji polarity (binary mapping)
├── ethics_oracle_neutral.py       # Ethics: quantify oracle neutral predictions
├── helpers.py                     # Shared helper functions
├── run.py                         # Main entry point to run training and evaluation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Requirements

```bash
pip install -r requirements.txt
```

## Preprocessing Pipeline

The preprocessing pipeline is composed of several scripts, each responsible for a specific transformation of the raw Twitter data.  
The scripts must be executed in the order listed below.

### **clean_and_dedup.py**
Reads the raw tweet files (`*_full.txt`, `*_pos.txt`, `*_neg.txt`) and removes strictly identical tweets, producing deduplicated text files (`*_clean.txt`).

### **preprocess.py**
Applies Twitter-specific text normalization, including lowercasing, number normalization (`<NUM>`), punctuation and emoticon handling, and noise filtering.

### **build_vocab.sh**
Extracts all tokens from the preprocessed files and computes their corpus-level frequencies.

### **cut_vocab.sh**
Removes rare words using a minimum frequency threshold of **3** to reduce vocabulary sparsity.

### **pickle_vocab.py**
Builds and serializes the vocabulary dictionary (`vocab.pkl`) mapping words to indices.

### **cooc.py**
Constructs a word co-occurrence matrix (`cooc.pkl`) from the cleaned positive and negative datasets.

### **glove_pretrained.py**
Loads pretrained GloVe embeddings trained on large-scale Twitter data and aligns them with the project vocabulary.

or

### **glove_trained.py**
Trains GloVe embeddings from scratch using the task-specific co-occurrence matrix (`cooc.pkl`), generating embeddings learned directly from the project dataset.

---

**Note:**  
This pipeline is used only for word-based models (CNN, BiLSTM, and linear baselines). Transformer-based models such as **BERTweet** rely on their own tokenizer and do not use this preprocessing.

## Prerequisites

Before running the models, ensure your project directory is set up as follows:

1. **Data Folder**: You must have a folder named `twitter-datasets/` in the root directory containing the raw text files:
    * `train_pos_full.txt` & `train_neg_full.txt` (The full 2.5M dataset)
    * `train_pos.txt` & `train_neg.txt` (The smaller 200k dataset)
    * `test_data.txt` (The unlabeled test set)

2. **Preprocessing Outputs**: All preprocessing steps must be completed first to generate:
    * `vocab.pkl`: The processed vocabulary mapping.
    * `embeddings.npy`: The trained word vectors (GloVe) used as embedding weights for BiLSTM and CNN.

You must also manually download GloVe embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip and place the 50d version at the root of the project.

## How to Run

### Best accuracy: BERTweet
```bash
python run.py
```
Finetunes BERTweet by aligning the preprocessing to the one used during BERTweet pretraining.

### BiLSTM
```bash
python run.py --model bilstm
```

### CNN
```bash
python run.py --model cnn
```

For CNN and BiLSTM, `--tuning True` enables grid search (BiLSTM) or cross-validation (CNN).

## Oracle Evaluation Pipeline

To assess the ethical risk of misinterpreting our model as a sentiment classifier, we evaluate the relationship between **emoji-derived polarity labels** and predictions from a strong, independently trained Twitter sentiment model (**oracle**).

### `ethics_oracle_eval.py`
Runs `cardiffnlp/twitter-roberta-base-sentiment` on training tweets and forces its three-class output into a binary decision. Results are compared against emoji polarity labels using accuracy, F1-score, and a confusion matrix.

### `ethics_oracle_neutral.py`
Runs the same oracle **without collapsing the neutral class** to quantify how often text expresses no clear sentiment while still being assigned a binary emoji polarity label — providing evidence that emoji polarity classification is **not equivalent to sentiment analysis**.
