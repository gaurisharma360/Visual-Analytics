"""
Active Learning Logistic Regression Model for EEG Seizure Classification
Multi-class classification for seizure detection with 5 categories:
  - A (Z): Healthy subjects, eyes open (Non-Seizure)
  - B (O): Healthy subjects, eyes closed (Non-Seizure)
  - C (N): Epilepsy patients, seizure-free interval (Non-Seizure)
  - D (F): Epilepsy patients, seizure area (Seizure)
  - E (S): Epilepsy patients, during seizure (Seizure)

Uses uncertainty sampling to query a medical expert oracle for annotations.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import pickle
import os


class Oracle:
    """
    Oracle validates user annotations by comparing with ground truth.
    User is the annotator, oracle only checks if annotation is correct.
    """
    def __init__(self, labeled_data):
        """
        Initialize oracle with ground truth for validation

        Args:
            labeled_data: DataFrame with 'ID' and 'Y' columns
        """
        self.labeled_data = labeled_data
        self.label_dict = dict(zip(labeled_data['ID'], labeled_data['Y']))
        self.validation_count = 0
        self.correct_count = 0
        self.query_count = 0  # Keep for backward compatibility

    def validate(self, sample_id, user_label):
        """
        Validate user's annotation against ground truth

        Args:
            sample_id: ID of the sample
            user_label: Label provided by user

        Returns:
            (is_correct, true_label)
        """
        self.validation_count += 1
        true_label = self.label_dict.get(sample_id, None)
        is_correct = (user_label == true_label)
        if is_correct:
            self.correct_count += 1
        return is_correct, true_label

    def get_true_label(self, sample_id):
        """Get ground truth label"""
        return self.label_dict.get(sample_id, None)

    def get_accuracy(self):
        """Get user's annotation accuracy"""
        if self.validation_count == 0:
            return 0.0
        return self.correct_count / self.validation_count

    def query(self, sample_id):
        """
        Query the oracle for the true label of a sample

        Args:
            sample_id: ID of the sample to query

        Returns:
            True label of the sample
        """
        self.query_count += 1
        return self.label_dict.get(sample_id, None)

    def batch_query(self, sample_ids):
        """
        Query multiple samples at once

        Args:
            sample_ids: List of sample IDs

        Returns:
            List of true labels
        """
        labels = [self.query(sid) for sid in sample_ids]
        return labels

    def reset_count(self):
        """Reset counters"""
        self.query_count = 0
        self.validation_count = 0
        self.correct_count = 0


class ActiveLearningClassifier:
    """
    Active Learning wrapper for Logistic Regression with multi-class classification
    """

    def __init__(self, initial_labeled_size=50, batch_size=10,
                 use_pca=False, n_components=100, random_state=42):
        """
        Initialize Active Learning Classifier

        Args:
            initial_labeled_size: Number of initial labeled samples
            batch_size: Number of samples to query in each iteration
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components (if use_pca=True)
            random_state: Random seed for reproducibility
        """
        self.initial_labeled_size = initial_labeled_size
        self.batch_size = batch_size
        self.use_pca = use_pca
        self.n_components = n_components
        self.random_state = random_state

        # Model components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        )

        # Data storage
        self.labeled_pool = None
        self.unlabeled_pool = None
        self.oracle = None

        # Performance tracking
        self.training_history = {
            'iteration': [],
            'labeled_size': [],
            'accuracy': [],
            'queries_made': []
        }

    def load_data(self, csv_path):
        """
        Load and prepare the EEG dataset

        Args:
            csv_path: Path to the CSV file
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Separate features and labels
        self.ids = df['ID'].values
        self.X = df.drop(['ID', 'Y'], axis=1).values
        self.y = df['Y'].values

        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Classes: {np.unique(self.y)}")
        print("Categories: A/B/C=Non-Seizure, D/E=Seizure")

        # Create test set (20% of data, never used in active learning)
        X_train_pool, self.X_test, y_train_pool, self.y_test, ids_train, ids_test = \
            train_test_split(self.X, self.y, self.ids,
                           test_size=0.2,
                           stratify=self.y,
                           random_state=self.random_state)

        # Initialize oracle with all training pool data
        oracle_data = pd.DataFrame({'ID': ids_train, 'Y': y_train_pool})
        self.oracle = Oracle(oracle_data)

        # Normalize features
        self.scaler.fit(X_train_pool)
        X_train_pool_scaled = self.scaler.transform(X_train_pool)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Apply PCA if enabled
        if self.use_pca:
            print(f"Applying PCA: {self.X.shape[1]} -> {self.n_components} features")
            self.pca.fit(X_train_pool_scaled)
            X_train_pool_scaled = self.pca.transform(X_train_pool_scaled)
            self.X_test_scaled = self.pca.transform(self.X_test_scaled)
            explained_var = sum(self.pca.explained_variance_ratio_)
            print(f"PCA explained variance: {explained_var:.2%}")

        # Initialize labeled pool with stratified random samples
        if self.initial_labeled_size > 0:
            initial_indices = []
            for label in np.unique(y_train_pool):
                label_indices = np.where(y_train_pool == label)[0]
                n_samples = max(1, self.initial_labeled_size // len(np.unique(y_train_pool)))
                selected = np.random.choice(label_indices, size=min(n_samples, len(label_indices)),
                                          replace=False)
                initial_indices.extend(selected)

            initial_indices = np.array(initial_indices)
            unlabeled_indices = np.array([i for i in range(len(X_train_pool_scaled))
                                         if i not in initial_indices])

            # Create labeled pool
            self.labeled_pool = {
                'X': X_train_pool_scaled[initial_indices],
                'y': y_train_pool[initial_indices],
                'ids': ids_train[initial_indices],
                'indices': initial_indices
            }

            # Create unlabeled pool
            self.unlabeled_pool = {
                'X': X_train_pool_scaled[unlabeled_indices],
                'y': y_train_pool[unlabeled_indices],  # Ground truth (hidden from model)
                'ids': ids_train[unlabeled_indices],
                'indices': unlabeled_indices
            }
        else:
            # No initial labels - user will annotate from scratch
            self.labeled_pool = {
                'X': np.empty((0, X_train_pool_scaled.shape[1])),
                'y': np.array([]),
                'ids': np.array([]),
                'indices': np.array([])
            }

            # All samples start as unlabeled
            self.unlabeled_pool = {
                'X': X_train_pool_scaled,
                'y': y_train_pool,  # Ground truth (hidden from model)
                'ids': ids_train,
                'indices': np.arange(len(X_train_pool_scaled))
            }

        print(f"\nInitialized pools:")
        print(f"  Labeled: {len(self.labeled_pool['X'])} samples")
        print(f"  Unlabeled: {len(self.unlabeled_pool['X'])} samples")
        print(f"  Test: {len(self.X_test)} samples")

    def random_sampling(self, n_samples):
        """
        Select random samples (used when no trained model exists yet)

        Args:
            n_samples: Number of samples to select

        Returns:
            Indices of randomly selected samples
        """
        if len(self.unlabeled_pool['X']) == 0:
            return np.array([])

        n_samples = min(n_samples, len(self.unlabeled_pool['X']))
        return np.random.choice(len(self.unlabeled_pool['X']), size=n_samples, replace=False)

    def uncertainty_sampling(self, n_samples):
        """
        Select most uncertain samples using entropy-based uncertainty

        Args:
            n_samples: Number of samples to select

        Returns:
            Indices of selected samples in the unlabeled pool
        """
        if len(self.unlabeled_pool['X']) == 0:
            return np.array([])

        # Predict probabilities for unlabeled samples
        probabilities = self.model.predict_proba(self.unlabeled_pool['X'])

        # Calculate entropy (uncertainty measure)
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)

        # Select top-k most uncertain samples
        n_samples = min(n_samples, len(entropy))
        uncertain_indices = np.argsort(entropy)[-n_samples:][::-1]

        return uncertain_indices

    def auto_label_confident_samples(self, confidence_threshold_low=0.4, confidence_threshold_high=0.6):
        """
        Automatically label samples where model is highly confident
        Only uncertain samples (probability between thresholds) need expert annotation

        Args:
            confidence_threshold_low: Lower threshold (e.g., 0.4)
            confidence_threshold_high: Upper threshold (e.g., 0.6)

        Returns:
            Number of samples auto-labeled
        """
        if len(self.unlabeled_pool['X']) == 0:
            return 0

        # Get predictions for all unlabeled samples
        probabilities = self.model.predict_proba(self.unlabeled_pool['X'])
        predicted_labels = self.model.predict(self.unlabeled_pool['X'])

        # Find max probability for each sample (model's confidence)
        max_probs = np.max(probabilities, axis=1)

        # Auto-label samples with high confidence (< 0.4 or > 0.6)
        confident_mask = (max_probs < confidence_threshold_low) | (max_probs > confidence_threshold_high)
        confident_indices = np.where(confident_mask)[0]

        if len(confident_indices) == 0:
            return 0

        # Move confident samples to labeled pool with predicted labels
        auto_X = self.unlabeled_pool['X'][confident_indices]
        auto_y = predicted_labels[confident_indices]
        auto_ids = self.unlabeled_pool['ids'][confident_indices]

        if len(self.labeled_pool['X']) == 0:
            self.labeled_pool['X'] = auto_X
        else:
            self.labeled_pool['X'] = np.vstack([self.labeled_pool['X'], auto_X])
        self.labeled_pool['y'] = np.concatenate([self.labeled_pool['y'], auto_y])
        self.labeled_pool['ids'] = np.concatenate([self.labeled_pool['ids'], auto_ids])

        # Remove from unlabeled pool
        uncertain_mask = ~confident_mask
        self.unlabeled_pool['X'] = self.unlabeled_pool['X'][uncertain_mask]
        self.unlabeled_pool['y'] = self.unlabeled_pool['y'][uncertain_mask]
        self.unlabeled_pool['ids'] = self.unlabeled_pool['ids'][uncertain_mask]

        print(f"✓ Auto-labeled {len(confident_indices)} confident samples")
        return len(confident_indices)

    def get_uncertain_samples_only(self, max_samples=10):
        """
        Get only the truly uncertain samples that need expert annotation
        Samples with probability between 0.4 and 0.6

        Args:
            max_samples: Maximum number of uncertain samples to return

        Returns:
            Indices of uncertain samples in unlabeled pool
        """
        if len(self.unlabeled_pool['X']) == 0:
            return np.array([])

        # Get predictions
        probabilities = self.model.predict_proba(self.unlabeled_pool['X'])
        max_probs = np.max(probabilities, axis=1)

        # Find uncertain samples (probability between 0.4 and 0.6)
        uncertain_mask = (max_probs >= 0.4) & (max_probs <= 0.6)
        uncertain_indices = np.where(uncertain_mask)[0]

        if len(uncertain_indices) == 0:
            return np.array([])

        # If more uncertain samples than max_samples, pick the most uncertain
        if len(uncertain_indices) > max_samples:
            # Calculate entropy for uncertain samples
            probs_uncertain = probabilities[uncertain_indices]
            epsilon = 1e-10
            entropy = -np.sum(probs_uncertain * np.log(probs_uncertain + epsilon), axis=1)

            # Select top-k most uncertain
            top_k_in_uncertain = np.argsort(entropy)[-max_samples:][::-1]
            uncertain_indices = uncertain_indices[top_k_in_uncertain]

        return uncertain_indices

    def add_user_annotation(self, sample_id, user_label):
        """
        Add a user-annotated sample to the labeled pool

        Args:
            sample_id: ID of the sample
            user_label: Label provided by user (A, B, C, D, or E)

        Returns:
            (is_correct, true_label) from oracle validation
        """
        # Find sample in unlabeled pool
        sample_idx = np.where(self.unlabeled_pool['ids'] == sample_id)[0]

        if len(sample_idx) == 0:
            return None, None

        sample_idx = sample_idx[0]

        # Validate with oracle
        is_correct, true_label = self.oracle.validate(sample_id, user_label)

        # Move sample to labeled pool with USER'S label (not ground truth!)
        new_X = self.unlabeled_pool['X'][sample_idx:sample_idx+1]
        new_y = np.array([user_label])
        new_id = np.array([sample_id])

        if len(self.labeled_pool['X']) == 0:
            self.labeled_pool['X'] = new_X
        else:
            self.labeled_pool['X'] = np.vstack([self.labeled_pool['X'], new_X])
        self.labeled_pool['y'] = np.concatenate([self.labeled_pool['y'], new_y])
        self.labeled_pool['ids'] = np.concatenate([self.labeled_pool['ids'], new_id])

        # Remove from unlabeled pool
        mask = np.ones(len(self.unlabeled_pool['X']), dtype=bool)
        mask[sample_idx] = False
        self.unlabeled_pool['X'] = self.unlabeled_pool['X'][mask]
        self.unlabeled_pool['y'] = self.unlabeled_pool['y'][mask]
        self.unlabeled_pool['ids'] = self.unlabeled_pool['ids'][mask]

        return is_correct, true_label

    def get_sample_for_annotation(self, use_uncertainty=True):
        """
        Get next sample for user to annotate

        Args:
            use_uncertainty: Use uncertainty sampling if model trained, else random

        Returns:
            (sample_id, sample_index) or (None, None) if no samples left
        """
        if len(self.unlabeled_pool['X']) == 0:
            return None, None

        # Use uncertainty sampling if model is trained, else random
        if use_uncertainty and len(self.labeled_pool['X']) >= 5:
            try:
                indices = self.uncertainty_sampling(1)
            except:
                indices = self.random_sampling(1)
        else:
            indices = self.random_sampling(1)

        if len(indices) == 0:
            return None, None

        idx = indices[0]
        sample_id = self.unlabeled_pool['ids'][idx]

        return sample_id, idx

    def query_oracle(self, indices):
        """
        Query oracle for true labels and move samples to labeled pool

        Args:
            indices: Indices in the unlabeled pool to query
        """
        if len(indices) == 0:
            return

        # Get sample IDs
        sample_ids = self.unlabeled_pool['ids'][indices]

        # Query oracle for true labels
        true_labels = self.oracle.batch_query(sample_ids)

        # Move samples from unlabeled to labeled pool
        new_X = self.unlabeled_pool['X'][indices]
        new_y = np.array(true_labels)
        new_ids = sample_ids

        self.labeled_pool['X'] = np.vstack([self.labeled_pool['X'], new_X])
        self.labeled_pool['y'] = np.concatenate([self.labeled_pool['y'], new_y])
        self.labeled_pool['ids'] = np.concatenate([self.labeled_pool['ids'], new_ids])

        # Remove queried samples from unlabeled pool
        mask = np.ones(len(self.unlabeled_pool['X']), dtype=bool)
        mask[indices] = False
        self.unlabeled_pool['X'] = self.unlabeled_pool['X'][mask]
        self.unlabeled_pool['y'] = self.unlabeled_pool['y'][mask]
        self.unlabeled_pool['ids'] = self.unlabeled_pool['ids'][mask]

        print(f"Queried {len(indices)} samples. Oracle queries: {self.oracle.query_count}")

    def train(self):
        """Train the logistic regression model on labeled data"""
        if len(self.labeled_pool['X']) < 5:
            return False  # Need at least 5 samples to train
        self.model.fit(self.labeled_pool['X'], self.labeled_pool['y'])
        return True

    def evaluate(self):
        """
        Evaluate model on test set

        Returns:
            Accuracy score
        """
        y_pred = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def active_learning_loop(self, n_iterations=20):
        """
        Run the active learning loop

        Args:
            n_iterations: Number of active learning iterations
        """
        print("\n" + "="*60)
        print("ACTIVE LEARNING LOOP")
        print("="*60)

        for iteration in range(n_iterations):
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

            # Train model
            self.train()

            # Evaluate
            accuracy = self.evaluate()

            # Store metrics
            self.training_history['iteration'].append(iteration + 1)
            self.training_history['labeled_size'].append(len(self.labeled_pool['X']))
            self.training_history['accuracy'].append(accuracy)
            self.training_history['queries_made'].append(self.oracle.query_count)

            print(f"Labeled samples: {len(self.labeled_pool['X'])}")
            print(f"Test accuracy: {accuracy:.4f}")

            # Check if we've used all data
            if len(self.unlabeled_pool['X']) == 0:
                print("\nNo more unlabeled data available!")
                break

            # Select uncertain samples
            uncertain_indices = self.uncertainty_sampling(self.batch_size)

            # Query oracle
            self.query_oracle(uncertain_indices)

        print("\n" + "="*60)
        print("ACTIVE LEARNING COMPLETE")
        print("="*60)

    def get_detailed_evaluation(self):
        """
        Get detailed evaluation metrics

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.model.predict(self.X_test_scaled)

        results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }

        return results

    def save_model(self, path='active_learning_model.pkl'):
        """Save the trained model and components"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'use_pca': self.use_pca,
            'training_history': self.training_history
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to {path}")

    def load_model(self, path='active_learning_model.pkl'):
        """Load a trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.use_pca = model_data['use_pca']
        self.training_history = model_data['training_history']

        print(f"Model loaded from {path}")

    def predict(self, X):
        """
        Predict labels for new data

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Predict class probabilities for new data

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict_proba(X_scaled)


def main():
    """
    Main function to demonstrate active learning
    """
    # Initialize active learning classifier
    print("Initializing Active Learning Classifier...")
    al_classifier = ActiveLearningClassifier(
        initial_labeled_size=50,
        batch_size=10,
        use_pca=True,
        n_components=100,
        random_state=42
    )

    # Load data
    csv_path = "bonn_eeg_combined.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    al_classifier.load_data(csv_path)

    # Run active learning loop
    al_classifier.active_learning_loop(n_iterations=20)

    # Get final detailed evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    results = al_classifier.get_detailed_evaluation()
    print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    # Save model
    al_classifier.save_model('active_learning_model.pkl')

    # Save training history
    history_df = pd.DataFrame(al_classifier.training_history)
    history_df.to_csv('training_history.csv', index=False)
    print("\nTraining history saved to training_history.csv")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Initial labeled samples: {al_classifier.training_history['labeled_size'][0]}")
    print(f"Final labeled samples: {al_classifier.training_history['labeled_size'][-1]}")
    print(f"Total oracle queries: {al_classifier.oracle.query_count}")
    print(f"Initial accuracy: {al_classifier.training_history['accuracy'][0]:.4f}")
    print(f"Final accuracy: {al_classifier.training_history['accuracy'][-1]:.4f}")
    print(f"Improvement: {al_classifier.training_history['accuracy'][-1] - al_classifier.training_history['accuracy'][0]:.4f}")


if __name__ == "__main__":
    main()





