from jbi100_app.main import app

from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from active_learning_model import ActiveLearningClassifier

# Load EEG data (GLOBAL)
eeg_df = pd.read_csv('bonn_eeg_combined.csv')

# Initialize active learning classifier (GLOBAL)
al_classifier = ActiveLearningClassifier(
    initial_labeled_size=50,  # Start with 50 labeled (10% cold start)
    batch_size=10,
    use_pca=False,
    random_state=42
)
al_classifier.load_data('bonn_eeg_combined.csv')

# Train initial model
print("\n🔷 Training initial model with 50 samples...")
trained = al_classifier.train()

if not trained or len(al_classifier.labeled_pool['X']) < 5:
    print("ERROR: Failed to train model properly!")
    raise Exception("Model training failed - check your data")

initial_accuracy = al_classifier.evaluate()
print(f"✓ Initial accuracy: {initial_accuracy:.2%}")

# Get first batch of uncertain samples
print("🔷 Selecting 10 most uncertain samples...")
try:
    uncertain_indices = al_classifier.uncertainty_sampling(10)
    current_batch = [al_classifier.unlabeled_pool['ids'][i] for i in uncertain_indices]
    print(f"✓ First batch: {current_batch}")
except Exception as e:
    print(f"ERROR in uncertainty sampling: {e}")
    print("Using random sampling instead...")
    random_indices = np.random.choice(len(al_classifier.unlabeled_pool['ids']), size=10, replace=False)
    current_batch = [al_classifier.unlabeled_pool['ids'][i] for i in random_indices]
    print(f"✓ First batch (random): {current_batch}")

current_batch_index = 0


if __name__ == '__main__':

    app.layout = html.Div(
        id="app-container",
        children=[
            html.Div(
                id="main-column",
                className="twelve columns",
                children=[
                    html.Div(
                        className="pretty_container",
                        children=[
                            html.H4("🧠 Active Learning: EEG Seizure Classification"),
                            html.P(id='round-info', children=f"Round 1 | Model Accuracy: {initial_accuracy:.1%} | Annotate 10 most uncertain samples"),

                            # Stats
                            html.Div([
                                html.Span(id='labeled-count', children=f"Labeled: 50 | ", style={'marginRight': '10px'}),
                                html.Span(id='unlabeled-count', children=f"Unlabeled: 350 | ", style={'marginRight': '10px'}),
                                html.Span(f"Batch: 0/10", id='batch-progress')
                            ], style={'marginBottom': '20px', 'fontSize': '14px'}),

                            # Current sample
                            html.Div([
                                html.H5("Current Sample:", style={'display': 'inline', 'marginRight': '10px'}),
                                html.H5(current_batch[0], id='current-sample-display', style={'display': 'inline', 'color': '#e74c3c'})
                            ], style={'marginBottom': '20px'}),

                            # Category legend
                            html.Div([
                                html.Strong("Categories: "),
                                html.Span("A: Healthy (eyes open) | ", style={'color': '#27ae60'}),
                                html.Span("B: Healthy (eyes closed) | ", style={'color': '#3498db'}),
                                html.Span("C: Epilepsy (seizure-free) | ", style={'color': '#f39c12'}),
                                html.Span("D: Seizure area | ", style={'color': '#e67e22'}),
                                html.Span("E: During seizure", style={'color': '#e74c3c'})
                            ], style={'fontSize': '12px', 'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),

                            # Annotation buttons
                            html.Div([
                                html.Button('A', id='btn-a', n_clicks=0,
                                          style={'margin': '5px', 'padding': '15px 30px', 'fontSize': '18px',
                                                'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                                                'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                                html.Button('B', id='btn-b', n_clicks=0,
                                          style={'margin': '5px', 'padding': '15px 30px', 'fontSize': '18px',
                                                'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                                'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                                html.Button('C', id='btn-c', n_clicks=0,
                                          style={'margin': '5px', 'padding': '15px 30px', 'fontSize': '18px',
                                                'backgroundColor': '#f39c12', 'color': 'white', 'border': 'none',
                                                'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                                html.Button('D', id='btn-d', n_clicks=0,
                                          style={'margin': '5px', 'padding': '15px 30px', 'fontSize': '18px',
                                                'backgroundColor': '#e67e22', 'color': 'white', 'border': 'none',
                                                'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                                html.Button('E', id='btn-e', n_clicks=0,
                                          style={'margin': '5px', 'padding': '15px 30px', 'fontSize': '18px',
                                                'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                                'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                            ], style={'textAlign': 'center', 'marginBottom': '20px'}),

                            # Train button
                            html.Div([
                                html.Button('🎓 Retrain Model', id='btn-train', n_clicks=0,
                                          style={'margin': '10px', 'padding': '15px 30px', 'fontSize': '18px',
                                                'backgroundColor': '#9b59b6', 'color': 'white', 'border': 'none',
                                                'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                            ], style={'textAlign': 'center', 'marginTop': '20px'}),

                            # Feedback
                            html.Div(id='feedback', style={'marginTop': '20px', 'padding': '15px', 'borderRadius': '5px', 'textAlign': 'center', 'fontSize': '16px'}),

                            # Hidden storage
                            html.Div(id='batch-index', children='0', style={'display': 'none'}),
                            html.Div(id='round-number', children='1', style={'display': 'none'})
                        ],
                        style={'marginBottom': '20px', 'maxWidth': '900px', 'margin': '0 auto'}
                    ),
                ],
            ),
        ],
    )

    # Annotation callback
    @app.callback(
        [Output('current-sample-display', 'children'),
         Output('batch-progress', 'children'),
         Output('batch-index', 'children'),
         Output('feedback', 'children'),
         Output('feedback', 'style'),
         Output('labeled-count', 'children'),
         Output('unlabeled-count', 'children')],
        [Input('btn-a', 'n_clicks'),
         Input('btn-b', 'n_clicks'),
         Input('btn-c', 'n_clicks'),
         Input('btn-d', 'n_clicks'),
         Input('btn-e', 'n_clicks')],
        [State('batch-index', 'children')]
    )
    def annotate(n_a, n_b, n_c, n_d, n_e, batch_idx_str):
        from dash import callback_context
        import numpy as np

        batch_idx = int(batch_idx_str)

        # Initial load
        if not callback_context.triggered:
            return current_batch[0], f"Batch: 0/10", "0", "", {}, f"Labeled: {len(al_classifier.labeled_pool['X'])} | ", f"Unlabeled: {len(al_classifier.unlabeled_pool['X'])} | "

        # Get which button was clicked
        button_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        label_map = {'btn-a': 'A', 'btn-b': 'B', 'btn-c': 'C', 'btn-d': 'D', 'btn-e': 'E'}
        user_label = label_map.get(button_id)

        if not user_label:
            return current_batch[batch_idx], f"Batch: {batch_idx}/10", str(batch_idx), "", {}, f"Labeled: {len(al_classifier.labeled_pool['X'])} | ", f"Unlabeled: {len(al_classifier.unlabeled_pool['X'])} | "

        # Get current sample
        current_sample = current_batch[batch_idx]

        # Validate with oracle
        is_correct, true_label = al_classifier.oracle.validate(current_sample, user_label)

        # Move sample to labeled pool
        sample_idx = np.where(al_classifier.unlabeled_pool['ids'] == current_sample)[0][0]

        new_X = al_classifier.unlabeled_pool['X'][sample_idx:sample_idx+1]
        new_y = np.array([user_label])
        new_id = np.array([current_sample])

        al_classifier.labeled_pool['X'] = np.vstack([al_classifier.labeled_pool['X'], new_X])
        al_classifier.labeled_pool['y'] = np.concatenate([al_classifier.labeled_pool['y'], new_y])
        al_classifier.labeled_pool['ids'] = np.concatenate([al_classifier.labeled_pool['ids'], new_id])

        # Remove from unlabeled
        mask = np.ones(len(al_classifier.unlabeled_pool['X']), dtype=bool)
        mask[sample_idx] = False
        al_classifier.unlabeled_pool['X'] = al_classifier.unlabeled_pool['X'][mask]
        al_classifier.unlabeled_pool['y'] = al_classifier.unlabeled_pool['y'][mask]
        al_classifier.unlabeled_pool['ids'] = al_classifier.unlabeled_pool['ids'][mask]

        # Feedback
        if is_correct:
            feedback = f"✓ Correct! ({user_label})"
            feedback_style = {'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#d4edda',
                            'color': '#155724', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}
        else:
            feedback = f"✗ Wrong! You said {user_label}, correct is {true_label}"
            feedback_style = {'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f8d7da',
                            'color': '#721c24', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}

        # Move to next sample
        batch_idx += 1
        batch_size = len(current_batch)

        if batch_idx < batch_size:
            next_sample = current_batch[batch_idx]
            progress = f"Batch: {batch_idx}/{batch_size}"
        else:
            next_sample = "✅ Batch Complete!"
            progress = f"Batch: {batch_size}/{batch_size} - Click Retrain!"
            feedback = "✅ Batch complete! Click 'Retrain Model' button."
            feedback_style = {'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#d1ecf1',
                            'color': '#0c5460', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}

        # Update counts
        labeled_text = f"Labeled: {len(al_classifier.labeled_pool['X'])} | "
        unlabeled_text = f"Unlabeled: {len(al_classifier.unlabeled_pool['X'])} | "

        return next_sample, progress, str(batch_idx), feedback, feedback_style, labeled_text, unlabeled_text

    # Retrain callback
    @app.callback(
        [Output('current-sample-display', 'children', allow_duplicate=True),
         Output('batch-progress', 'children', allow_duplicate=True),
         Output('batch-index', 'children', allow_duplicate=True),
         Output('feedback', 'children', allow_duplicate=True),
         Output('feedback', 'style', allow_duplicate=True),
         Output('round-number', 'children'),
         Output('labeled-count', 'children', allow_duplicate=True),
         Output('unlabeled-count', 'children', allow_duplicate=True),
         Output('round-info', 'children')],
        Input('btn-train', 'n_clicks'),
        State('round-number', 'children'),
        prevent_initial_call=True
    )
    def retrain(n_clicks, round_str):
        global current_batch
        import numpy as np

        if n_clicks == 0:
            return current_batch[0], "Batch: 0/10", "0", "", {}, "1", f"Labeled: 50 | ", f"Unlabeled: 350 | ", f"Round 1 | Model Accuracy: {initial_accuracy:.1%}"

        # Retrain model
        print(f"\n🔷 Retraining with {len(al_classifier.labeled_pool['X'])} labeled samples...")
        al_classifier.train()
        accuracy = al_classifier.evaluate()
        print(f"✓ New accuracy: {accuracy:.2%}")

        # AUTO-LABEL CONFIDENT SAMPLES (Real Active Learning!)
        print(f"🔷 Auto-labeling confident samples (prob < 0.4 or > 0.6)...")
        auto_labeled_count = al_classifier.auto_label_confident_samples(
            confidence_threshold_low=0.4,
            confidence_threshold_high=0.6
        )

        if auto_labeled_count > 0:
            print(f"✓ {auto_labeled_count} samples auto-labeled!")
            print(f"✓ New labeled pool size: {len(al_classifier.labeled_pool['X'])}")

        # Get only UNCERTAIN samples for expert annotation
        if len(al_classifier.unlabeled_pool['X']) > 0:
            print(f"🔷 Finding uncertain samples (prob 0.4-0.6) for expert annotation...")

            try:
                uncertain_indices = al_classifier.get_uncertain_samples_only(max_samples=10)

                if len(uncertain_indices) == 0:
                    # No uncertain samples - everything is confident!
                    print("✓ No uncertain samples remaining! All are confident.")
                    feedback = f"🎉 No uncertain samples left! Auto-labeled {auto_labeled_count}. Final accuracy: {accuracy:.1%}"
                    feedback_style = {'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#d4edda',
                                    'color': '#155724', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}
                    labeled_text = f"Labeled: {len(al_classifier.labeled_pool['X'])} | "
                    unlabeled_text = f"Unlabeled: {len(al_classifier.unlabeled_pool['X'])} | "
                    round_info = f"Complete! Final Accuracy: {accuracy:.1%}"
                    return "Done!", "Complete", "10", feedback, feedback_style, round_str, labeled_text, unlabeled_text, round_info

                current_batch = [al_classifier.unlabeled_pool['ids'][i] for i in uncertain_indices]
                batch_size = len(current_batch)
                print(f"✓ Found {batch_size} uncertain samples for expert: {current_batch}")

            except Exception as e:
                print(f"ERROR: {e}")
                # Fallback to random
                batch_size = min(10, len(al_classifier.unlabeled_pool['X']))
                random_indices = np.random.choice(len(al_classifier.unlabeled_pool['ids']), size=batch_size, replace=False)
                current_batch = [al_classifier.unlabeled_pool['ids'][i] for i in random_indices]
                print(f"✓ Fallback batch: {current_batch}")

            round_num = int(round_str) + 1

            feedback = f"✓ Round {round_str}: {accuracy:.1%} accuracy | Auto-labeled {auto_labeled_count} confident | {len(current_batch)} uncertain need YOU"
            feedback_style = {'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#d4edda',
                            'color': '#155724', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}

            labeled_text = f"Labeled: {len(al_classifier.labeled_pool['X'])} | "
            unlabeled_text = f"Unlabeled: {len(al_classifier.unlabeled_pool['X'])} | "
            round_info = f"Round {round_num} | Accuracy: {accuracy:.1%} | {len(current_batch)} uncertain samples (auto-labeled {auto_labeled_count})"

            return current_batch[0], f"Batch: 0/{len(current_batch)}", "0", feedback, feedback_style, str(round_num), labeled_text, unlabeled_text, round_info
        else:
            feedback = f"🎉 Complete! Final accuracy: {accuracy:.1%}"
            feedback_style = {'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#d4edda',
                            'color': '#155724', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}
            labeled_text = f"Labeled: {len(al_classifier.labeled_pool['X'])} | "
            unlabeled_text = f"Unlabeled: 0 | "
            round_info = f"Complete! Final Accuracy: {accuracy:.1%}"
            return "Done!", "Complete", "10", feedback, feedback_style, round_str, labeled_text, unlabeled_text, round_info

    app.run(debug=False, dev_tools_ui=False)