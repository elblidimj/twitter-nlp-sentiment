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
├── twitter_datasets/             # Dataset directory (must be added manually)
│   ├── train_pos.txt
│   ├── train_neg.txt
│   ├── train_pos_full.txt
│   ├── train_neg_full.txt
│   └── test_data.txt
│
├── ethics_oracle_eval.py          # Ethics: oracle sentiment vs emoji polarity (binary mapping)
├── ethics_oracle_neutral.py       # Ethics: quantify oracle neutral predictions
├── helpers.py                     # Shared helper functions
├── run.py                         # Main entry point to run training and evaluation
├── requirements.txt               # Python dependencies
└── README.md                      # This file


```
## Requirements
You can run ``pip install -r requirements.txt``

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

2. **Preprocessing Outputs**: All preprocessing steps (tokenization, co-occurrence building, and GloVe training) must be completed first to generate the following files in your root directory:
    * `vocab.pkl`: The processed vocabulary mapping.
    * `embeddings.npy`: The trained word vectors (GloVe) used as the embedding weights for the BiLSTM and CNN.

You must also manually download GloVe embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip and put them 50d version at the root of the project

## How to run the pipeline

### Best accuracy : BERTweet 
Run `run.py` to get the best accuracy performed and submitted to aicrowd. It finetunes a BERTweet model by aligning the preprocessing existing in the dataset to the one that was used for training BERTweet.

## BiLSTM
Run `run.py --model bilstm` to run the bilstm model on the tuned hyperparameters

## CNN
Run `run.py --model cnn` to run the cnn model on the tuned hyperparameters

For CNN and BiLSTM an argument `--tuning` also exists, which when set to True allows to run a grid search with bilstm and cross validation with CNN.

## Oracle Evaluation Pipeline

To assess the ethical risk of misinterpreting our model as a sentiment classifier, we evaluate the relationship between **emoji-derived polarity labels** and predictions from a strong, independently trained Twitter sentiment model (**oracle**).

### `ethics_oracle_eval.py`

Runs a pretrained Twitter sentiment classifier (`cardiffnlp/twitter-roberta-base-sentiment`) on the training tweets (full or not full) and forces its three-class output (**negative, neutral, positive**) into a binary decision by mapping neutral tweets to a polarity via **positive-vs-negative logits**.  
The resulting predictions are compared against the emoji polarity labels using **accuracy**, **F1-score**, and a **confusion matrix**.  

### `ethics_oracle_neutral.py`

Runs the same oracle **without collapsing the neutral class**.  
It explicitly counts tweets predicted as **neutral** by the oracle, allowing us to quantify how often the text expresses no clear sentiment while still being assigned a binary emoji polarity label.  
This analysis highlights that a large fraction of tweets are neutral according to the oracle, implying that any binary sentiment interpretation in such cases would be inherently **arbitrary**.

Together, these scripts provide **quantitative evidence** that emoji polarity classification is **not equivalent to sentiment analysis**.
