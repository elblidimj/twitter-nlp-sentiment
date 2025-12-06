from sklearn.linear_model import LogisticRegression

def build_logreg(C=1.0, max_iter=1000):
    return LogisticRegression(C=C, max_iter=max_iter)

