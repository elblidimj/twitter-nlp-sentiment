from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def train_val_split(X, y, val_size=0.1, seed=42):
    """
    Split the dataset into training and validation sets with stratification.
    Ensures that both sets keep the same ratio of positive and negative samples.
    """
    return train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )


def evaluate_model(model, X_val, y_val):
    """
    Evaluate classifier performance on validation set.
    """
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    return acc, cm