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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# ==========================================================
#  HELPER FUNCTION (For Confusion Matrix Display)
# ==========================================================
def display_confusion_matrix(y_true, y_pred):
    class_labels = ["Non-Seizure", "Seizure"]
    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=[f"True {label}" for label in class_labels],
        columns=[f"Pred {label}" for label in class_labels]
    )

    print("Confusion Matrix (Test):")
    print(cm_df)

    #Sensitivity (Seizure Recall) measures the model's ability to correctly identify seizure cases,
    # while Specificity (Non-Seizure Recall) measures its ability to correctly identify non-seizure cases.
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    print(f"Sensitivity (Seizure Recall): {round(sensitivity,4)}")
    print(f"Specificity (Non-Seizure Recall): {round(specificity,4)}")

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def extract_features(X_raw, fs=173.61):

    features = []

    for signal in X_raw:
        # mean value helps to understand the overall level of the signal, which can differ between seizure and non-seizure states.
        mean_val = np.mean(signal)
        # std deviation captures the variability in the signal, which can be higher during seizures due to erratic brain activity.
        std_val = np.std(signal)
        # RMS gives a measure of the signal's power, which can be elevated during seizures.
        rms = np.sqrt(np.mean(signal**2))
        #peak-to-peak measures the range of the signal, which can be larger during seizures due to sudden spikes in brain activity.
        peak_to_peak = np.ptp(signal)
        #skewness captures the asymmetry of the signal distribution, which can indicate abnormal brain activity during seizures.
        skewness = skew(signal)
        # kutosis measures the "tailedness" of the signal distribution, which can be higher during seizures due to extreme values in the EEG signal.
        kurt_val = kurtosis(signal)

        freqs, psd = welch(signal, fs=fs, nperseg=256)

        def band_power(fmin, fmax):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            return np.sum(psd[idx])
        #each frequency band captures different aspects of brain activity.
        #delta are commonly associated with deep sleep and can be more prominent during certain types of seizures.
        delta = band_power(0.5, 4)
        #theta band is often linked to drowsiness and early stages of sleep, and can also be more active during seizures, especially in the temporal lobe.
        theta = band_power(4, 8)
        #alpha rhythms are typically associated with relaxed wakefulness and can be disrupted during seizures, making them a useful feature for distinguishing between seizure and non-seizure states.
        alpha = band_power(8, 13)
        #beta activity is associated with active thinking and focus, and can be reduced during seizures, making it another important feature for classification.
        beta  = band_power(13, 30)

        features.append([
            mean_val, std_val, rms, peak_to_peak,
            skewness, kurt_val,
            delta, theta, alpha, beta
        ])

    return np.array(features)


# ==========================================================
# LOAD + SUBJECT SAFE SPLIT
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
# TRAIN MODEL WITH CROSS-VALIDATION (GROUP K-FOLD)
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
# ACTIVE LEARNING SIMULATION
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
        

        learning_curve.append((len(labeled_idx), test_acc))

        print("\nTrain Accuracy:", round(train_acc, 4))
        print("Test Accuracy :", round(test_acc, 4))
        
        #Confusion matrix with labeled display
        display_confusion_matrix(y_test, test_pred)
        
        # For a complete picture of model performance, we also print the classification 
        # report which includes precision, recall, and F1-score for each class.
        print("\nClassification Report (Test):")
        print(classification_report(
            y_test,
            test_pred,
            target_names=["Non-Seizure", "Seizure"]
        ))

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