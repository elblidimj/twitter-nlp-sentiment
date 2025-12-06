# src/model/svm.py

from sklearn.svm import LinearSVC

def build_linear_svm(C=1.0):
    return LinearSVC(C=C)
