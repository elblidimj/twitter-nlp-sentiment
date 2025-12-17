import numpy as np
from helpers import create_csv_submission
from src.utils.io_utils import load_stopwords, load_vocab_and_embeddings, load_idf_weights
from src.datasets.twitter import load_training_tweets, load_test_tweets
from src.transforms.text_embeddings import tweets_to_matrix
from src.model.logreg import build_logreg
from src.trainer.validation import train_val_split, evaluate_model
from src.trainer.tuning import tune_logreg, tune_svm
from src.model.cnn import build_cnn_model, save_cnn_visualizations
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_and_predict(
    data_dir="twitter-datasets",
    vocab_path="vocab.pkl",
    emb_path="embeddings.npy",
    submission_name="submission.csv"
):
    # 1. Load vocabulary and pre-trained embeddings
    vocab, embeddings = load_vocab_and_embeddings(vocab_path, emb_path)

    # 2. Load and vectorize training data
    tweets_train, y_train = load_training_tweets(
        data_dir=data_dir, use_full=True, stopwords=None, do_plots=False
    )
    X_train = tweets_to_matrix(tweets_train, vocab, embeddings, None)

    # 3. Label preparation for Keras (convert from -1/1 to 0/1)
    y_train_keras = np.where(y_train == 1, 1, 0)

    # 4. Train/Validation Split
    X_tr, X_val, y_tr_keras, y_val_keras = train_val_split(X_train, y_train_keras, val_size=0.1)

    # 5. Build CNN model and define callbacks
    classifier = build_cnn_model(embeddings) 
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,       # Reduce learning rate by 50%
        patience=2,       # After 2 epochs without val_loss improvement
        min_lr=0.00001
    )

    print("--- Starting CNN Training ---")
    
    # 6. Main Training phase
    history = classifier.fit(
        X_tr, y_tr_keras,
        epochs=10, 
        batch_size=512, 
        validation_data=(X_val, y_val_keras), 
        callbacks=[early_stopping, reduce_lr]
    )
    
    # 7. Visualization Generation
    print("Generating and saving visualizations...")
    vocab_inv = {i: w for w, i in vocab.items()}
    sample_indices = X_val[0]
    sample_words = [vocab_inv.get(idx, '<PAD>') for idx in sample_indices]
    
    save_cnn_visualizations(
        model=classifier, 
        history=history, 
        sample_tweet_indices=sample_indices, 
        vocab_inv=sample_words
    )

    # 8. Model Evaluation
    val_loss, val_acc = classifier.evaluate(X_val, y_val_keras, verbose=0)
    print(f"VAL -> Accuracy after CNN: {val_acc:.4f}")
    
    # 9. Final training on the full dataset before submission
    print("Final training on the complete dataset...")
    classifier.fit(X_train, y_train_keras, epochs=1, batch_size=128) # One additional epoch

    # 10. Test Set Prediction
    test_ids, test_tweets = load_test_tweets(data_dir=data_dir)
    # Convert test tweets into sequences of indices
    X_test = tweets_to_matrix(test_tweets, vocab, embeddings, stopwords=None) 
    
    # Prediction (returns probabilities between 0 and 1)
    y_test_pred_probs = classifier.predict(X_test)
    
    # Convert probabilities back to labels (-1 or 1)
    y_test_pred = np.where(y_test_pred_probs > 0.5, 1, -1).astype(int)

    # 11. Create submission file
    create_csv_submission(test_ids, y_test_pred, submission_name)
    print(f"Submission file created: {submission_name}")