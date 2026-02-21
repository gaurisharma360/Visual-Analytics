"""
FULL TERMINAL SIMULATION (CLEAN VERSION)
----------------------------------------
Feature Engineered EEG
Subject-safe split (no leakage)
Batch Active Learning (10 per round)
Oracle annotation simulation
Learning curve tracking
Confusion matrix updates
Minimal clean output
"""

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================================
# 1️⃣ FEATURE ENGINEERING
# ==========================================================
def extract_features(X_raw, fs=173.61):

    features = []

    for signal in X_raw:

        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        peak_to_peak = np.ptp(signal)
        skewness = skew(signal)
        kurt_val = kurtosis(signal)

        freqs, psd = welch(signal, fs=fs, nperseg=256)

        def band_power(fmin, fmax):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            return np.sum(psd[idx])

        delta = band_power(0.5, 4)
        theta = band_power(4, 8)
        alpha = band_power(8, 13)
        beta  = band_power(13, 30)

        features.append([
            mean_val, std_val, rms, peak_to_peak,
            skewness, kurt_val,
            delta, theta, alpha, beta
        ])

    return np.array(features)


# ==========================================================
# 2️⃣ LOAD + SUBJECT SAFE SPLIT
# ==========================================================
def load_and_split(binary=True):

    df = pd.read_csv("bonn_eeg_combined.csv")

    X_raw = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values

    print("\nExtracting features...")
    X = extract_features(X_raw)

    # Reconstruct subjects
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
        print("Binary classification: Seizure vs Non-Seizure")
    else:
        y = y_original
        print("5-Class mode")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    print(f"Train pool size: {len(train_idx)}")
    print(f"Frozen test size: {len(test_idx)}")

    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        subjects[train_idx]
    )


# ==========================================================
# 3️⃣ TRAIN MODEL (GROUP SAFE)
# ==========================================================
def train_model(X_train, y_train, subjects_train, show_params=False):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='saga',
            max_iter=10000,
            random_state=42
        ))
    ])

    param_grid = {
        'clf__C': [1, 0.1, 0.01],
        'clf__penalty': ['l1', 'l2']
    }

    gkf = GroupKFold(n_splits=5)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf.split(X_train, y_train, groups=subjects_train),
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    if show_params:
        print("Best Parameters:", grid.best_params_)
        print("Best CV Accuracy:", round(grid.best_score_, 4))

    return grid.best_estimator_


# ==========================================================
# 4️⃣ ACTIVE LEARNING SIMULATION
# ==========================================================
def simulate_active_learning(X_train, y_train, subjects_train,
                             X_test, y_test,
                             initial_fraction=0.2,
                             batch_size=10,
                             max_rounds=8):

    print("\nStarting Active Learning Simulation...")

    n_initial = int(initial_fraction * len(X_train))
    indices = np.random.permutation(len(X_train))

    labeled_idx = indices[:n_initial]
    unlabeled_idx = indices[n_initial:]

    learning_curve = []

    for iteration in range(max_rounds):

        print("\n" + "="*70)
        print(f"ROUND {iteration+1}")
        print(f"Labeled samples: {len(labeled_idx)}")

        model = train_model(
            X_train[labeled_idx],
            y_train[labeled_idx],
            subjects_train[labeled_idx],
            show_params=(iteration == 0)  # Show params only once
        )

        # Training performance
        train_pred = model.predict(X_train[labeled_idx])
        train_acc = accuracy_score(y_train[labeled_idx], train_pred)

        # Test performance
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        cm = confusion_matrix(y_test, test_pred)

        learning_curve.append((len(labeled_idx), test_acc))

        print("\nTrain Accuracy:", round(train_acc, 4))
        print("Test Accuracy :", round(test_acc, 4))
        print("Confusion Matrix (Test):")
        print(cm)

        if len(unlabeled_idx) == 0:
            print("All samples labeled.")
            break

        # Compute uncertainty
        probs = model.predict_proba(X_train[unlabeled_idx])
        uncertainty = 1 - np.max(probs, axis=1)

        # Select most uncertain samples
        query_order = np.argsort(uncertainty)[-batch_size:]
        queried = unlabeled_idx[query_order]

        print(f"\nDoctor annotates {len(queried)} uncertain samples...")

        labeled_idx = np.concatenate([labeled_idx, queried])
        unlabeled_idx = np.setdiff1d(unlabeled_idx, queried)

    # Print learning curve summary
    print("\n" + "="*70)
    print("LEARNING CURVE (Labeled Samples → Test Accuracy)")
    for samples, acc in learning_curve:
        print(f"{samples} → {round(acc,4)}")
    print("="*70)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
#True for binary
    X_train, X_test, y_train, y_test, subjects_train = load_and_split(binary=True)
#rounds could be adjusted as per requirement
    simulate_active_learning(
        X_train, y_train, subjects_train,
        X_test, y_test,
        initial_fraction=0.2,
        batch_size=10,
        max_rounds=8
    )