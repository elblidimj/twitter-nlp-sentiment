import numpy as np
from helpers import create_csv_submission
from src.utils.io_utils import load_stopwords, load_vocab_and_embeddings
from src.datasets.twitter import load_training_tweets, load_test_tweets
from src.transforms.text_embeddings import tweets_to_matrix
from src.model.logreg import build_logreg
from src.trainer.validation import train_val_split, evaluate_model
from src.trainer.tuning import tune_logreg, tune_svm

def train_and_predict(
    data_dir="twitter-datasets",
    vocab_path="vocab.pkl",
    emb_path="embeddings.npy",
    submission_name="submission.csv"
):
    vocab, embeddings = load_vocab_and_embeddings(vocab_path, emb_path)
    stopwords = load_stopwords("stopwords.pkl")

    tweets_train, y_train = load_training_tweets(
        data_dir=data_dir, use_full=True, stopwords=stopwords, do_plots=False
    )
    X_train = tweets_to_matrix(tweets_train, vocab, embeddings, stopwords)

    X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, val_size=0.2)

    classifier = tune_svm(X_tr, y_tr, cv_folds=5, plot=False)
    classifier.fit(X_tr, y_tr)
    
    val_acc, val_cm = evaluate_model(classifier, X_val, y_val)

    print("Missclassification error score:", 1-val_acc)
    TP = val_cm[1, 1]
    FP = val_cm[0, 1]
    FN = val_cm[1, 0]
    TN = val_cm[0, 0]
    print(f"VAL -> TP:{TP}, FP:{FP}, FN:{FN}, TN:{TN}")

    classifier.fit(X_train, y_train)

    test_ids, test_tweets = load_test_tweets(data_dir=data_dir)
    X_test = tweets_to_matrix(test_tweets, vocab, embeddings, stopwords)
    y_test_pred = classifier.predict(X_test)
    y_test_pred = np.where(y_test_pred >= 0, 1, -1).astype(int)

    create_csv_submission(test_ids, y_test_pred, submission_name)
    print(f"Submission file created: {submission_name}")
