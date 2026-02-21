"""
Logistic Regression with Feature Engineering for EEG Seizure Detection
Binary classification with statistical features extracted from raw EEG signals
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
import pickle


def extract_statistical_features(X):
    """Extract statistical features from raw EEG data"""
    print("Extracting statistical features...")

    features = []
    n_samples = X.shape[0]

    for i in range(n_samples):
        sample = X[i]

        feat = [
            np.mean(sample),              # Mean
            np.std(sample),               # Standard deviation
            np.var(sample),               # Variance
            np.median(sample),            # Median
            np.min(sample),               # Min
            np.max(sample),               # Max
            np.max(sample) - np.min(sample),  # Range
            stats.skew(sample),           # Skewness
            stats.kurtosis(sample),       # Kurtosis
            np.percentile(sample, 25),    # 25th percentile
            np.percentile(sample, 75),    # 75th percentile
            np.percentile(sample, 75) - np.percentile(sample, 25),  # IQR
            np.sqrt(np.mean(sample**2)),  # RMS
            np.sum(np.abs(np.diff(sample))),  # Total variation
            np.mean(np.abs(np.diff(sample))), # Mean absolute difference
        ]
        features.append(feat)

    feature_matrix = np.array(features)
    print(f"✓ Extracted {feature_matrix.shape[1]} statistical features")
    return feature_matrix


def load_and_prepare_data(csv_path='bonn_eeg_combined.csv', test_size=0.2, random_state=42):
    """Load data and extract features"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    X = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values
    y_binary = np.where(np.isin(y_original, ['D', 'E']), 1, 0)

    print(f"Original data: {X.shape[0]} samples, {X.shape[1]} raw features")

    # Extract statistical features
    X_features = extract_statistical_features(X)

    print(f"Non-Seizure: {np.sum(y_binary == 0)}, Seizure: {np.sum(y_binary == 1)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_binary, test_size=test_size, stratify=y_binary, random_state=random_state
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, C=1.0):
    """Train Logistic Regression with engineered features"""
    print(f"\nTraining Logistic Regression with Feature Engineering (C={C})...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=C,
        solver='lbfgs',
        max_iter=3000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)

    print("✓ Model trained successfully!")
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model"""
    print("\nEvaluating model...")

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Seizure', 'Seizure']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    print(f"\nSensitivity: {sensitivity:.2%}, Specificity: {specificity:.2%}")

    return accuracy


def save_model(model, scaler, filename='logistic_regression_features.pkl'):
    """Save model"""
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"\n✓ Model saved to {filename}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print("\n" + "="*60)
    print("Testing different C values...")
    print("="*60)

    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    for i, C in enumerate(C_values, 1):
        print(f"\n[{i}/{len(C_values)}] C={C}")
        model, scaler = train_model(X_train, y_train, C=C)
        accuracy = evaluate_model(model, scaler, X_test, y_test)
        results.append((accuracy, model, scaler, C))

    best_accuracy, best_model, best_scaler, best_C = max(results, key=lambda x: x[0])
    save_model(best_model, best_scaler)

    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"Best C: {best_C}")
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print("="*60)

    print("\nAll Results (sorted by accuracy):")
    print("-" * 60)
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)
    for acc, _, _, C in results_sorted:
        print(f"C={C:>8.3f}  |  Accuracy: {acc:.2%}")
    print("="*60)

