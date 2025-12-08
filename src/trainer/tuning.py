import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def tune_logreg(X, y, cv_folds=5, plot=True):
    """
    Perform cross-validation to find best C and max_iter for Logistic Regression.
    Produce a heatmap of misclassification error for (lambda, max_iter).
    Highlight the best param combination with a white point.
    """

    param_grid = {
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "max_iter": [500, 1000, 1500, 2000]
    }

    logreg = LogisticRegression()

    grid = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )

    print("Running Logistic Regression cross-validation...")
    grid.fit(X, y)

    best_params = grid.best_params_
    best_C = best_params["C"]
    best_lambda = 1 / best_C
    best_iter = best_params["max_iter"]

    print("\nBest parameters:", best_params)
    print("Best CV accuracy:", grid.best_score_)

    # Extract CV results
    results = grid.cv_results_

    accuracies = np.array(results["mean_test_score"])
    errors = 1 - accuracies

    Cs = np.array([p["C"] for p in results["params"]])
    lambdas = 1 / Cs
    max_iters = np.array([p["max_iter"] for p in results["params"]])

    if plot:
        lambda_values = np.sort(np.unique(lambdas))
        max_iter_values = np.sort(np.unique(max_iters))

        # Create heatmap matrix
        heatmap = np.zeros((len(max_iter_values), len(lambda_values)))

        for i, mi in enumerate(max_iter_values):
            for j, lam in enumerate(lambda_values):
                mask = (max_iters == mi) & (lambdas == lam)
                heatmap[i, j] = errors[mask][0]

        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap, cmap="viridis", aspect="auto")
        plt.colorbar(label="Misclassification error")

        # Tick labels
        plt.xticks(range(len(lambda_values)), [f"{lam:.3f}" for lam in lambda_values])
        plt.yticks(range(len(max_iter_values)), max_iter_values)

        plt.xlabel("lambda = 1/C")
        plt.ylabel("max_iter")
        plt.title("Misclassification error heatmap for Logistic Regression")

        # Highlight best params
        best_x = np.where(lambda_values == best_lambda)[0][0]
        best_y = np.where(max_iter_values == best_iter)[0][0]

        plt.scatter(best_x, best_y, color="white", s=100, edgecolors="black", linewidths=1.5)

        plt.show()

    return grid.best_estimator_



def tune_svm(X, y, cv_folds=5, plot=True):
    """
    Perform cross-validation to find best C and max_iter for SVM.
    Produce a heatmap of misclassification error for (C, max_iter).
    Highlight the best param combination with a white point.
    """

    param_grid = {
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "max_iter": [500, 1000, 1500, 2000]
    }

    svm = LinearSVC()

    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )

    print("Running SVM cross-validation...")
    grid.fit(X, y)

    best_params = grid.best_params_    
    best_C = best_params["C"]
    best_lambda = 1 / best_C
    best_iter = best_params["max_iter"]

    print("\nBest parameters:", best_params)
    print("Best CV accuracy:", grid.best_score_)

    # Extract CV results
    results = grid.cv_results_

    accuracies = np.array(results["mean_test_score"])
    errors = 1 - accuracies

    Cs = np.array([p["C"] for p in results["params"]])
    lambdas = 1 / Cs
    max_iters = np.array([p["max_iter"] for p in results["params"]])

    if plot:
        lambda_values = np.sort(np.unique(lambdas))
        max_iter_values = np.sort(np.unique(max_iters))

        # Create heatmap matrix
        heatmap = np.zeros((len(max_iter_values), len(lambda_values)))

        for i, mi in enumerate(max_iter_values):
            for j, lam in enumerate(lambda_values):
                mask = (max_iters == mi) & (lambdas == lam)
                heatmap[i, j] = errors[mask][0]

        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap, cmap="viridis", aspect="auto")
        plt.colorbar(label="Misclassification error")

        # Tick labels
        plt.xticks(range(len(lambda_values)), [f"{lam:.3f}" for lam in lambda_values])
        plt.yticks(range(len(max_iter_values)), max_iter_values)

        plt.xlabel("lambda = 1/C")
        plt.ylabel("max_iter")
        plt.title("Misclassification error heatmap for SVM")

        # Highlight best params
        best_x = np.where(lambda_values == best_lambda)[0][0]
        best_y = np.where(max_iter_values == best_iter)[0][0]

        plt.scatter(best_x, best_y, color="white", s=100, edgecolors="black", linewidths=1.5)

        plt.show()

    return grid.best_estimator_