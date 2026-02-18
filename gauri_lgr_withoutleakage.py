"""
Bonn EEG – Subject-Safe Logistic Regression
-------------------------------------------
- 4097 time samples per segment
- 500 segments total
- 10 true subjects reconstructed
- Group-safe train/test split
- Group-safe cross-validation
- No subject leakage
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1️⃣ Load Data + Reconstruct Subjects
# --------------------------------------------------
def load_data(csv_path='bonn_eeg_combined.csv', binary=False):

    df = pd.read_csv(csv_path)

    X = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values

    # --------------------------------------------------
    # Reconstruct 10 subjects for Bonn dataset
    # Each set has 100 segments
    # Each set has 5 subjects
    # Each subject contributes 20 segments per set
    # --------------------------------------------------

    subjects = []

    for set_idx in range(5):  # A–E
        for i in range(100):
            subject_within_set = i // 20  # 0–4

            if set_idx < 2:
                # Healthy subjects (Sets A,B)
                subject_id = subject_within_set
            else:
                # Epileptic subjects (Sets C,D,E)
                subject_id = subject_within_set + 5

            subjects.append(subject_id)

    subjects = np.array(subjects)

    # --------------------------------------------------
    # Optional: Binary seizure detection
    # --------------------------------------------------

    if binary:
        y = (y_original == 'E').astype(int)
        print("Binary mode: Seizure vs Non-Seizure")
    else:
        y = y_original
        print("5-Class mode")

    print("Dataset shape:", X.shape)
    print("Unique subjects:", np.unique(subjects))
    print("Class distribution:", np.unique(y, return_counts=True))

    return X, y, subjects


# --------------------------------------------------
# 2️⃣ Subject-Safe Train/Test Split
# --------------------------------------------------
def split_data(X, y, subjects):

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], subjects[train_idx]


# --------------------------------------------------
# 3️⃣ Train Model (Group-Safe CV)
# --------------------------------------------------
def train_model(X_train, y_train, subjects_train):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='saga',
            max_iter=5000,
            random_state=42
        ))
    ])

    param_grid = {
        'clf__C': [1, 0.1, 0.01, 0.005, 0.001],
        'clf__penalty': ['l1', 'l2'],
        'clf__class_weight': [None, 'balanced']
    }

    gkf = GroupKFold(n_splits=5)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf.split(X_train, y_train, groups=subjects_train),
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_, 4))

    return grid.best_estimator_


# --------------------------------------------------
# 4️⃣ Evaluate
# --------------------------------------------------
def evaluate(model, X_train, X_test, y_train, y_test):

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print("\n==============================")
    print("Train Accuracy:", round(train_acc, 4))
    print("Test Accuracy :", round(test_acc, 4))
    print("==============================")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, model.predict(X_test)))

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, model.predict(X_test)))


# --------------------------------------------------
# 5️⃣ MAIN
# --------------------------------------------------
if __name__ == "__main__":

    # Set binary=True if you want seizure vs non-seizure
    X, y, subjects = load_data(binary=False)

    X_train, X_test, y_train, y_test, subjects_train = split_data(X, y, subjects)

    model = train_model(X_train, y_train, subjects_train)

    evaluate(model, X_train, X_test, y_train, y_test)
