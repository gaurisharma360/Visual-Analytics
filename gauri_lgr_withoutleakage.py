"""
Bonn EEG – Subject-Safe Logistic Regression (Feature Engineered)
----------------------------------------------------------------
- Raw 4097 samples converted to meaningful signal features
- No PCA
- Group-safe split
- Group-safe CV
"""

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1️⃣ Feature Engineering
# --------------------------------------------------
def extract_features(X_raw, fs=173.61):

    features = []

    for signal in X_raw:

        # ---------- Time Domain ----------
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        peak_to_peak = np.ptp(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)

        # ---------- Frequency Domain ----------
        freqs, psd = welch(signal, fs=fs, nperseg=256)

        def band_power(fmin, fmax):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            return np.sum(psd[idx])

        delta = band_power(0.5, 4)
        theta = band_power(4, 8)
        alpha = band_power(8, 13)
        beta = band_power(13, 30)

        features.append([
            mean_val, std_val, rms, peak_to_peak,
            skewness, kurt,
            delta, theta, alpha, beta
        ])

    return np.array(features)


# --------------------------------------------------
# 2️⃣ Load Data + Reconstruct Subjects
# --------------------------------------------------
def load_data(csv_path='bonn_eeg_combined.csv', binary=False):

    df = pd.read_csv(csv_path)

    X_raw = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values

    print("Extracting signal features...")
    X = extract_features(X_raw)

    # ----- Subject reconstruction -----
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

    # ----- Binary option -----
    if binary:
        y = (y_original == 'E').astype(int)
        print("Binary mode: Seizure vs Non-Seizure")
    else:
        y = y_original
        print("5-Class mode")

    print("Feature matrix shape:", X.shape)
    print("Unique subjects:", np.unique(subjects))

    return X, y, subjects


# --------------------------------------------------
# 3️⃣ Subject-Safe Split
# --------------------------------------------------
def split_data(X, y, subjects):

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], subjects[train_idx]


# --------------------------------------------------
# 4️⃣ Train Model (No PCA)
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
        n_jobs=-1,
        verbose=1
    )

    print("\nTraining Logistic Regression with engineered features...")
    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_, 4))

    return grid.best_estimator_


# --------------------------------------------------
# 5️⃣ Evaluate
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
# 6️⃣ MAIN
# --------------------------------------------------
if __name__ == "__main__":

    print("="*70)
    print("LOGISTIC REGRESSION WITH FEATURE ENGINEERING")
    print("="*70)

    X, y, subjects = load_data(binary=False)

    X_train, X_test, y_train, y_test, subjects_train = split_data(X, y, subjects)

    model = train_model(X_train, y_train, subjects_train)

    evaluate(model, X_train, X_test, y_train, y_test)
