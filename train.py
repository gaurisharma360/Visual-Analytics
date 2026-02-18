"""
Interactive Active Learning - User Annotates EEG Samples
"""

from active_learning_model import ActiveLearningClassifier
import pandas as pd
import os

def display_sample_info(sample_id, iteration):
    """Display info about the sample to annotate"""
    category = sample_id[0]
    descriptions = {
        'A': 'Healthy (eyes open)',
        'B': 'Healthy (eyes closed)', 
        'C': 'Epilepsy (seizure-free)',
        'D': 'Seizure area',
        'E': 'During seizure'
    }
    
    print(f"\n{'='*70}")
    print(f"ANNOTATION #{iteration}")
    print(f"{'='*70}")
    print(f"Sample ID: {sample_id}")
    print(f"Hint: This sample is from category '{category}' ({descriptions[category]})")
    print(f"\nAvailable categories:")
    print("  A - Healthy (eyes open) - Non-Seizure")
    print("  B - Healthy (eyes closed) - Non-Seizure")
    print("  C - Epilepsy patient (seizure-free) - Non-Seizure")
    print("  D - Seizure area - Seizure")
    print("  E - During seizure - Seizure")

def train_interactive():
    """Interactive training with user annotations"""
    
    if not os.path.exists('bonn_eeg_combined.csv'):
        print("Error: bonn_eeg_combined.csv not found!")
        return
    
    print("="*70)
    print("INTERACTIVE ACTIVE LEARNING")
    print("="*70)
    print("\nYou are the annotator!")
    print("- Annotate EEG samples as A, B, C, D, or E")
    print("- Oracle will validate your annotations")
    print("- Model learns from YOUR labels")
    print("- Active learning starts with NO labels")
    
    # Initialize classifier
    classifier = ActiveLearningClassifier(
        initial_labeled_size=0,
        batch_size=1,
        use_pca=False,
        random_state=42
    )
    
    print("\n" + "-"*70)
    print("Loading data...")
    classifier.load_data('bonn_eeg_combined.csv')
    
    print("\n" + "="*70)
    print("START ANNOTATING")
    print("="*70)
    print("\nThe model will:")
    print("1. Show you a sample")
    print("2. You annotate it (A/B/C/D/E)")
    print("3. Oracle validates your annotation")
    print("4. After 5 labels: Model trains")
    print("5. Model selects most uncertain samples for you to annotate")
    print("\nEnter 'q' to quit, 's' to see stats")
    
    iteration = 0
    
    while True:
        # Get sample for annotation
        sample_id, idx = classifier.get_sample_for_annotation(
            use_uncertainty=(len(classifier.labeled_pool['X']) >= 5)
        )
        
        if sample_id is None:
            print("\n✓ All samples annotated!")
            break
        
        iteration += 1
        
        # Show sample info
        display_sample_info(sample_id, iteration)
        print(f"\nProgress: {len(classifier.labeled_pool['X'])}/400 labeled")
        print(f"Unlabeled: {len(classifier.unlabeled_pool['X'])}")
        
        # Get user input
        user_input = input("\nYour annotation (A/B/C/D/E) or 'q' to quit, 's' for stats: ").strip().upper()
        
        if user_input == 'Q':
            print("\nQuitting...")
            break
        
        if user_input == 'S':
            print(f"\n{'='*70}")
            print("STATISTICS")
            print(f"{'='*70}")
            print(f"Labeled samples: {len(classifier.labeled_pool['X'])}")
            print(f"Unlabeled samples: {len(classifier.unlabeled_pool['X'])}")
            print(f"Your annotation accuracy: {classifier.oracle.get_accuracy():.2%}")
            print(f"Correct: {classifier.oracle.correct_count}/{classifier.oracle.validation_count}")
            
            if len(classifier.labeled_pool['X']) >= 5:
                try:
                    classifier.train()
                    acc = classifier.evaluate()
                    print(f"Model test accuracy: {acc:.2%}")
                except:
                    print("Model not trained yet")
            
            iteration -= 1
            continue
        
        if user_input not in ['A', 'B', 'C', 'D', 'E']:
            print("Invalid input!")
            iteration -= 1
            continue
        
        # Add annotation
        is_correct, true_label = classifier.add_user_annotation(sample_id, user_input)
        
        # Show feedback
        if is_correct:
            print(f"\n✓ CORRECT! The true label is {true_label}")
        else:
            print(f"\n✗ INCORRECT! You said {user_input}, but true label is {true_label}")
        
        print(f"Your accuracy so far: {classifier.oracle.get_accuracy():.2%} ({classifier.oracle.correct_count}/{classifier.oracle.validation_count})")
        
        # Train model every 5 annotations
        if len(classifier.labeled_pool['X']) >= 5 and len(classifier.labeled_pool['X']) % 5 == 0:
            print(f"\n{'~'*70}")
            print(f"Training model with {len(classifier.labeled_pool['X'])} labeled samples...")
            trained = classifier.train()
            if trained:
                accuracy = classifier.evaluate()
                print(f"Model test accuracy: {accuracy:.2%}")
                print(f"Model will now suggest uncertain samples for annotation")
            print(f"{'~'*70}")
        
        # Ask if user wants to continue
        if iteration % 10 == 0:
            cont = input(f"\nContinue annotating? (y/n): ").strip().lower()
            if cont == 'n':
                break
    
    # Final summary
    print("\n" + "="*70)
    print("ANNOTATION SESSION COMPLETE")
    print("="*70)
    print(f"Total annotations: {len(classifier.labeled_pool['X'])}")
    print(f"Your accuracy: {classifier.oracle.get_accuracy():.2%}")
    print(f"Correct: {classifier.oracle.correct_count}/{classifier.oracle.validation_count}")
    
    if len(classifier.labeled_pool['X']) >= 5:
        print("\n" + "-"*70)
        print("Training final model...")
        classifier.train()
        accuracy = classifier.evaluate()
        print(f"Final model test accuracy: {accuracy:.2%}")
        
        # Save model
        save = input("\nSave model? (y/n): ").strip().lower()
        if save == 'y':
            classifier.save_model('model.pkl')
            print("Model saved as model.pkl")
            
            history = pd.DataFrame({
                'labeled_samples': [len(classifier.labeled_pool['X'])],
                'test_accuracy': [accuracy],
                'user_accuracy': [classifier.oracle.get_accuracy()],
                'annotations_validated': [classifier.oracle.validation_count]
            })
            history.to_csv('training_history.csv', index=False)
            print("History saved as training_history.csv")
    
    print("\n✓ Done!")

if __name__ == "__main__":
    train_interactive()
