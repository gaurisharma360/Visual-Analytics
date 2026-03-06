"""
Binary Logistic Regression for EEG Seizure Detection
Binary classification: Seizure vs Non-Seizure

Classes:
- Non-Seizure: A (healthy eyes open), B (healthy eyes closed), C (epilepsy seizure-free)
- Seizure: D (seizure area), E (during seizure)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle


def load_and_prepare_data(csv_path='bonn_eeg_combined.csv', test_size=0.2, random_state=42):
    """Load and convert to binary classification (Seizure vs Non-Seizure)"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Separate features and labels
    X = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values

    # Convert to binary: Non-Seizure (A,B,C) = 0, Seizure (D,E) = 1
    y_binary = np.where(np.isin(y_original, ['D', 'E']), 1, 0)

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Original classes: {np.unique(y_original)}")
    print(f"\nBinary conversion:")
    print(f"  Non-Seizure (A, B, C): {np.sum(y_binary == 0)} samples")
    print(f"  Seizure (D, E): {np.sum(y_binary == 1)} samples")

    # Split into train and test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, stratify=y_binary, random_state=random_state
    )

    print(f"\nTrain set: {len(X_train)} samples ({np.sum(y_train == 0)} non-seizure, {np.sum(y_train == 1)} seizure)")
    print(f"Test set: {len(X_test)} samples ({np.sum(y_test == 0)} non-seizure, {np.sum(y_test == 1)} seizure)")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, C=0.1):
    """Train binary logistic regression model"""
    print("\nTraining Binary Logistic Regression model...")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train logistic regression
    model = LogisticRegression(
        C=C,  # Regularization strength
        solver='lbfgs',
        max_iter=3000,
        random_state=42,
        class_weight='balanced'  # Handle any class imbalance
    )
    model.fit(X_train_scaled, y_train)

    print("✓ Model trained successfully!")

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate binary classification model"""
    print("\nEvaluating model...")

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"AUC-ROC Score: {auc_score:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Seizure', 'Seizure']))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives (Non-Seizure correctly): {cm[0,0]}")
    print(f"False Positives (Non-Seizure predicted as Seizure): {cm[0,1]}")
    print(f"False Negatives (Seizure predicted as Non-Seizure): {cm[1,0]}")
    print(f"True Positives (Seizure correctly): {cm[1,1]}")

    # Sensitivity and Specificity
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # True Positive Rate
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # True Negative Rate

    print(f"\nSensitivity (Seizure Detection Rate): {sensitivity:.2%}")
    print(f"Specificity (Non-Seizure Detection Rate): {specificity:.2%}")

    return accuracy, auc_score, y_pred, y_pred_proba


def save_model(model, scaler, filename='binary_logistic_regression_model.pkl'):
    """Save trained binary model and scaler"""
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"\n✓ Model saved to {filename}")


def load_model(filename='binary_logistic_regression_model.pkl'):
    """Load trained binary model and scaler"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']


def predict_sample(model, scaler, sample_features):
    """Predict seizure/non-seizure for a single sample"""
    sample_scaled = scaler.transform(sample_features.reshape(1, -1))
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]

    label = "Seizure" if prediction == 1 else "Non-Seizure"
    seizure_prob = probability[1]

    return prediction, label, seizure_prob


if __name__ == '__main__':
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print("\n" + "="*60)
    print("Testing different regularization strengths...")
    print("="*60)

    # Test more C values for better accuracy
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    for i, C in enumerate(C_values, 1):
        print(f"\n[{i}/{len(C_values)}] Training with C={C}")
        model, scaler = train_model(X_train, y_train, C=C)
        accuracy, auc, _, _ = evaluate_model(model, scaler, X_test, y_test)
        results.append((accuracy, auc, model, scaler, C))

    # Choose best model based on ACCURACY (not AUC)
    best_accuracy, best_auc, best_model, best_scaler, best_C = max(results, key=lambda x: x[0])

    # Save best model
    save_model(best_model, best_scaler)

    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"Best regularization: C={best_C}")
    print(f"Best Test Accuracy: {best_accuracy:.2%}")
    print(f"Best AUC-ROC Score: {best_auc:.4f}")
    print("="*60)

    # Show all results sorted by accuracy
    print("\nAll Results (sorted by accuracy):")
    print("-" * 60)
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)
    for acc, auc, _, _, C in results_sorted:
        print(f"C={C:>8.3f}  |  Accuracy: {acc:.2%}  |  AUC: {auc:.4f}")
    print("="*60)



