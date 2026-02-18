import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter


WINDOW_SIZE = 174  # ~1 second


# --------------------------------------------------
# 1️⃣ Load Data
# --------------------------------------------------
def load_data(csv_path='bonn_eeg_combined.csv'):
    df = pd.read_csv(csv_path)

    X_segments = df.drop(['ID', 'Y'], axis=1).values
    y_segments = df['Y'].values

    print(f"Original Segment Shape: {X_segments.shape}")
    print(f"Classes: {np.unique(y_segments)}")

    return X_segments, y_segments


# --------------------------------------------------
# 2️⃣ Window Function
# --------------------------------------------------
def create_windows(signal, window_size=WINDOW_SIZE):
    windows = []
    for start in range(0, len(signal) - window_size + 1, window_size):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


# --------------------------------------------------
# 3️⃣ Convert Segments to Windows
# --------------------------------------------------
def segment_to_windows(X_segments, y_segments):

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

    print("Total windows created:", X_windows.shape[0])
    print("Window feature size:", X_windows.shape[1])

    return X_windows, y_windows, segment_ids


# --------------------------------------------------
# 4️⃣ Split Data (Window Level)
# --------------------------------------------------
def split_data(X, y, segment_ids):
    return train_test_split(
        X, y, segment_ids,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


# --------------------------------------------------
# 5️⃣ Train with GridSearch
# --------------------------------------------------
def train_model(X_train, y_train):

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.2%}")

    return grid.best_estimator_


# --------------------------------------------------
# 6️⃣ Evaluate
# --------------------------------------------------
def evaluate(model, X_train, X_test, y_train, y_test,
             seg_train, seg_test, y_segments):

    # Window-level accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print("\n==============================")
    print("Window-Level Accuracy")
    print(f"Train: {train_acc:.2%}")
    print(f"Test : {test_acc:.2%}")
    print("==============================")

    # Segment-level aggregation
    test_preds = model.predict(X_test)

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

    print("\nSegment-Level Accuracy:", f"{segment_acc:.2%}")

    print("\nClassification Report (Segment-Level):")
    print(classification_report(segment_true, segment_preds))

    print("\nConfusion Matrix (SegmentLevel):")
    print(confusion_matrix(segment_true, segment_preds))


# --------------------------------------------------
# 7️⃣ Main
# --------------------------------------------------
if __name__ == "__main__":

    X_segments, y_segments = load_data()

    X_windows, y_windows, segment_ids = segment_to_windows(
        X_segments, y_segments
    )

    X_train, X_test, y_train, y_test, seg_train, seg_test = split_data(
        X_windows, y_windows, segment_ids
    )

    model = train_model(X_train, y_train)

    evaluate(model, X_train, X_test, y_train, y_test,
             seg_train, seg_test, y_segments)
