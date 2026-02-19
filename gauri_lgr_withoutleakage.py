"""
Bonn EEG – Subject-Safe Logistic Regression
Improved version:
- Reduced overfitting
- Reduced false negatives
- No subject leakage
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1️⃣ Load Data
# --------------------------------------------------
def load_data(csv_path='bonn_eeg_combined.csv', binary=True):

    df = pd.read_csv(csv_path)

    X = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values

    subjects = []

    for set_idx in range(5):
        for i in range(100):
            subject_within_set = i // 20
            if set_idx < 2:
                subject_id = subject_within_set
            else:
                subject_id = subject_within_set + 5
            subjects.append(subject_id)

    subjects = np.array(subjects)

    if binary:
        y = (y_original == 'E').astype(int)
    else:
        y = y_original

    return X, y, subjects


# --------------------------------------------------
# 2️⃣ Subject-Safe Split
# --------------------------------------------------
def split_data(X, y, subjects):

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        subjects[train_idx]
    )


# --------------------------------------------------
# 3️⃣ Train Model (Reduced Overfitting)
# --------------------------------------------------
def train_model(X_train, y_train, subjects_train):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('clf', LogisticRegression(
            solver='saga',
            max_iter=5000,
            random_state=42
        ))
    ])

    param_grid = {
        'pca__n_components': [20, 40, 60, 80],
        'clf__C': [0.1, 0.05, 0.01, 0.005],
        'clf__penalty': ['l1', 'l2'],
        'clf__class_weight': ['balanced']
    }

    gkf = GroupKFold(n_splits=5)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf.split(X_train, y_train, groups=subjects_train),
        scoring='f1',   # IMPORTANT CHANGE
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 4))

    return grid.best_estimator_


# --------------------------------------------------
# 4️⃣ Evaluate with Custom Threshold
# --------------------------------------------------
def evaluate(model, X_train, X_test, y_train, y_test, threshold=0.35):

    # Train performance
    train_probs = model.predict_proba(X_train)[:, 1]
    train_preds = (train_probs >= threshold).astype(int)

    # Test performance
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    print("\n==============================")
    print("Train Accuracy:", round(accuracy_score(y_train, train_preds), 4))
    print("Test Accuracy :", round(accuracy_score(y_test, test_preds), 4))
    print("==============================")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds))

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_preds))


# --------------------------------------------------
# 5️⃣ MAIN
# --------------------------------------------------
if __name__ == "__main__":

    X, y, subjects = load_data(binary=True)

    X_train, X_test, y_train, y_test, subjects_train = split_data(X, y, subjects)

    model = train_model(X_train, y_train, subjects_train)

    evaluate(model, X_train, X_test, y_train, y_test, threshold=0.35)
