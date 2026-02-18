"""
EEG Windowed Logistic Regression (Random Split)
------------------------------------------------
- 4097 time samples (~23.5 sec)
- 1-second windows (~174 samples)
- Random stratified split
- GridSearchCV tuning
- Segment-level evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter


# ==========================================================
# PARAMETERS
# ==========================================================

CSV_FILE = "bonn_eeg_combined.csv"  # 🔴 change if needed
WINDOW_SIZE = 174
TEST_SIZE = 0.2
MAX_ITER = 5000


# ==========================================================
# WINDOW FUNCTION
# ==========================================================

def create_windows(signal, window_size=WINDOW_SIZE):
    windows = []
    for start in range(0, len(signal) - window_size + 1, window_size):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)

    X_segments = df.iloc[:, 1:-1].values   # skip ID
    y_segments = df.iloc[:, -1].values

    print("Segment shape:", X_segments.shape)

    # --------------------------------------------------
    # Create Windows for ALL segments
    # --------------------------------------------------

    X_windows = []
    y_windows = []
    segment_ids = []

    for idx, (segment, label) in enumerate(zip(X_segments, y_segments)):
        windows = create_windows(segment)
        X_windows.append(windows)
        y_windows.extend([label] * len(windows))
        segment_ids.extend([idx] * len(windows))

    X_windows = np.vstack(X_windows)
    y_windows = np.array(y_windows)
    segment_ids = np.array(segment_ids)

    print("Total windows:", X_windows.shape[0])
    print("Window feature size:", X_windows.shape[1])

    # --------------------------------------------------
    # Train/Test Split (Window Level)
    # --------------------------------------------------

    X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
        X_windows,
        y_windows,
        segment_ids,
        test_size=TEST_SIZE,
        stratify=y_windows,
        random_state=42
    )

    print("Train windows:", X_train.shape[0])
    print("Test windows:", X_test.shape[0])

    # --------------------------------------------------
    # Pipeline + GridSearch
    # --------------------------------------------------

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='saga',
            max_iter=MAX_ITER,
            random_state=42
        ))
    ])

    param_grid = {
        'clf__C': [1, 0.1, 0.01, 0.005, 0.001],
        'clf__penalty': ['l1', 'l2']
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    print("\nTraining with GridSearch...")
    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_, 4))

    best_model = grid.best_estimator_

    # --------------------------------------------------
    # Window-Level Accuracy
    # --------------------------------------------------

    train_window_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_window_acc = accuracy_score(y_test, best_model.predict(X_test))

    print("\n==============================")
    print("Window-Level Accuracy")
    print("Train:", round(train_window_acc, 4))
    print("Test :", round(test_window_acc, 4))
    print("==============================")

    # --------------------------------------------------
    # Segment-Level Aggregation
    # --------------------------------------------------

    test_preds = best_model.predict(X_test)

    segment_pred_dict = {}

    for seg_id, pred in zip(seg_test, test_preds):
        if seg_id not in segment_pred_dict:
            segment_pred_dict[seg_id] = []
        segment_pred_dict[seg_id].append(pred)

    segment_preds = []
    segment_true = []

    for seg_id in segment_pred_dict:
        majority_label = Counter(segment_pred_dict[seg_id]).most_common(1)[0][0]
        segment_preds.append(majority_label)
        segment_true.append(y_segments[seg_id])

    segment_acc = accuracy_score(segment_true, segment_preds)

    print("\n==============================")
    print("Segment-Level Accuracy:", round(segment_acc, 4))
    print("==============================")

    print("\nClassification Report (Segment-Level):")
    print(classification_report(segment_true, segment_preds))

    print("Confusion Matrix (Segment-Level):")
    print(confusion_matrix(segment_true, segment_preds))
