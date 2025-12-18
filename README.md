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

### **glove.py**
Uses the co-occurrence matrix to train GloVe embeddings and outputs the final embedding matrix (`embeddings.npy`).

---

**Note:**  
This pipeline is used only for word-based models (CNN, BiLSTM, and linear baselines). Transformer-based models such as **BERTweet** rely on their own tokenizer and do not use this preprocessing.

## How to run the pipeline

### Best accuracy : BERTweet 
Run `run.py` to get the best accuracy performed and submitted to aicrowd. It finetunes a BERTweet model by aligning the preprocessing existing in the dataset to the one that was used for training BERTweet.

## BiLSTM
Run `run.py --model bilstm` to run the bilstm model on the tuned hyperparameters

## CNN
Run `run.py --model cnn` to run the cnn model on the tuned hyperparameters


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
