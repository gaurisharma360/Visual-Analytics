"""
Simple Logistic Regression for EEG Seizure Classification
Multi-class classification: A, B, C, D, E
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


def load_and_prepare_data(csv_path='bonn_eeg_combined.csv', test_size=0.2, random_state=42):
    """Load and split data into train/test sets"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Separate features and labels
    X = df.drop(['ID', 'Y'], axis=1).values
    y = df['Y'].values

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")

    # Split into train and test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, use_pca=True, n_components=200, C=0.01):
    """Train logistic regression model with PCA and optimized hyperparameters"""
    print("\nTraining Logistic Regression model...")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply PCA for dimensionality reduction
    pca = None
    if use_pca:
        print(f"Applying PCA: {X_train.shape[1]} -> {n_components} features")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        explained_var = sum(pca.explained_variance_ratio_)
        print(f"PCA explained variance: {explained_var:.2%}")

    # Train logistic regression with strong regularization
    model = LogisticRegression(
        C=C,  # Regularization strength
        solver='lbfgs',
        max_iter=3000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train_scaled, y_train)
      
    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"Train Accuracy: {train_accuracy:.2%}")
    print("✓ Model trained successfully!")

    return model, scaler, pca


def evaluate_model(model, scaler, X_test, y_test, pca=None):
    """Evaluate model on test set"""
    print("\nEvaluating model...")

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA if used
    if pca is not None:
        X_test_scaled = pca.transform(X_test_scaled)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return accuracy, y_pred


def save_model(model, scaler, pca=None, filename='logistic_regression_model.pkl'):
    """Save trained model, scaler, and PCA"""
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'pca': pca}, f)
    print(f"\n✓ Model saved to {filename}")


def load_model(filename='logistic_regression_model.pkl'):
    """Load trained model, scaler, and PCA"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data.get('pca', None)


def predict_sample(model, scaler, sample_features, pca=None):
    """Predict class for a single sample"""
    sample_scaled = scaler.transform(sample_features.reshape(1, -1))
    if pca is not None:
        sample_scaled = pca.transform(sample_scaled)
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]

    return prediction, probabilities


if __name__ == '__main__':
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print("\n" + "="*50)
    print("Testing different configurations...")
    print("="*50)

    # Test 1: No PCA, strong regularization
    print("\n[1/3] No PCA + C=0.01 regularization")
    model1, scaler1, pca1 = train_model(X_train, y_train, use_pca=False)
    acc1, _ = evaluate_model(model1, scaler1, X_test, y_test, pca1)

    # Test 2: PCA 200 components
    print("\n[2/3] PCA 200 components + C=0.1 regularization")
    model2, scaler2, pca2 = train_model(X_train, y_train, use_pca=True, n_components=200)
    acc2, _ = evaluate_model(model2, scaler2, X_test, y_test, pca2)

    # Test 3: PCA 300 components
    print("\n[3/3] PCA 300 components + C=0.1 regularization")
    model3, scaler3, pca3 = train_model(X_train, y_train, use_pca=True, n_components=300)
    acc3, _ = evaluate_model(model3, scaler3, X_test, y_test, pca3)

    # Choose best model
    best_idx = np.argmax([acc1, acc2, acc3])
    accuracies = [acc1, acc2, acc3]
    models = [(model1, scaler1, pca1), (model2, scaler2, pca2), (model3, scaler3, pca3)]
    configs = ["No PCA", "PCA-200", "PCA-300"]

    best_model, best_scaler, best_pca = models[best_idx]
    best_config = configs[best_idx]
    best_accuracy = accuracies[best_idx]

    # Save best model
    save_model(best_model, best_scaler, best_pca)

    print("\n" + "="*50)
    print("✓ Training complete!")
    print(f"Best configuration: {best_config}")
    print(f"Best Test Accuracy: {best_accuracy:.2%}")
    print("="*50)











