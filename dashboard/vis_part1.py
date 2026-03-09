# ==========================================================
# GOURISHA'S VISUAL ANALYTICS DASHBOARD - PART 1
# PEAX-Inspired Active Learning Visualization
# Features:
#   1. PCA Embedding View (2D projection of EEG features)
#   2. Uncertainty Distribution Histogram
# ==========================================================

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
from sklearn.decomposition import PCA

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def binary_entropy(probs):
    """
    Compute binary entropy from probability distributions.
    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    Args:
        probs: array of shape (n_samples, 2) with class probabilities

    Returns:
        array of shape (n_samples,) with entropy values [0, 1]
    """
    p = probs[:, 1]  # Probability of positive class (seizure)
    eps = 1e-10  # Avoid log(0)
    entropy = -p * np.log2(p + eps) - (1 - p) * np.log2(1 - p + eps)
    return np.clip(entropy, 0, 1)  # Ensure [0, 1] range

# ==========================================================
# FEATURE ENGINEERING (from gauri_activelearningcore.py)
# ==========================================================

def extract_features(X_raw, fs=173.61):
    """Extract statistical and spectral features from raw EEG signals"""
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
# DATA LOADING AND SPLITTING
# ==========================================================

def load_and_split(binary=True):
    """Load EEG data with subject-safe split"""
    df = pd.read_csv("../bonn_eeg_combined.csv")

    X_raw = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values

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
    else:
        y = y_original

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    return (
        X_raw[train_idx], X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        subjects[train_idx]
    )

# ==========================================================
# MODEL TRAINING
# ==========================================================

def train_model(X_train, y_train, subjects_train):
    """Train logistic regression with GroupKFold CV"""
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
        'clf__penalty': ['l1','l2']
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
    return grid.best_estimator_

# ==========================================================
# GLOBAL STATE INITIALIZATION
# ==========================================================

print("Loading and splitting data...")
X_raw_train, X_train, X_test, y_train, y_test, subjects_train = load_and_split(binary=True)

# Active Learning parameters
initial_fraction = 0.2
batch_size = 10
confidence_threshold = 0.7
max_rounds = 8

# Initialize pools
n_initial = int(initial_fraction * len(X_train))
indices = np.random.permutation(len(X_train))

labeled_idx = indices[:n_initial].copy()
unlabeled_idx = indices[n_initial:].copy()

# Track initial vs oracle-annotated samples
initial_labeled_idx = labeled_idx.copy()  # Store initial labeled samples
oracle_annotated_idx = np.array([], dtype=int)  # Track oracle annotations

# Training state
current_round = 0
model = None
pca_model = None
umap_model = None

# History tracking
learning_curve = []
confusion_history = []

print(f"Initial labeled: {len(labeled_idx)}, Unlabeled: {len(unlabeled_idx)}")
print("Training initial model...")

# Train initial model
model = train_model(
    X_train[labeled_idx],
    y_train[labeled_idx],
    subjects_train[labeled_idx]
)

# Fit PCA on full training set for consistent embedding
print("Fitting PCA for embedding visualization...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca_model = PCA(n_components=2, random_state=42)
X_train_pca = pca_model.fit_transform(X_train_scaled)
X_test_pca = pca_model.transform(X_test_scaled)

print(f"PCA variance explained: {pca_model.explained_variance_ratio_}")

# Fit UMAP if available
if UMAP_AVAILABLE:
    print("Fitting UMAP for embedding visualization...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_train_umap = umap_model.fit_transform(X_train_scaled)
    X_test_umap = umap_model.transform(X_test_scaled)
    print("UMAP embedding complete!")
else:
    X_train_umap = None
    X_test_umap = None
    print("UMAP not available - only PCA will be shown")

# Initial predictions and uncertainty
train_probs = model.predict_proba(X_train[labeled_idx])
train_uncertainty = 1 - np.max(train_probs, axis=1)

test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
learning_curve.append((len(labeled_idx), test_acc))

print(f"Initial test accuracy: {test_acc:.4f}")

# ==========================================================
# DASH APP INITIALIZATION
# ==========================================================

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("EEG Active Learning Dashboard - Part 1",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Control Panel
    html.Div([
        html.Div([
            html.Label("Active Learning Round:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Span(id='round-display', children=f"Round {current_round}/{max_rounds}",
                     style={'fontSize': '18px', 'color': '#2980b9', 'marginRight': '30px'}),

            html.Label("Labeled Samples:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Span(id='labeled-count', children=f"{len(labeled_idx)}/{len(X_train)}",
                     style={'fontSize': '18px', 'color': '#27ae60', 'marginRight': '30px'}),

            html.Label("Test Accuracy:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Span(id='test-accuracy', children=f"{test_acc:.4f}",
                     style={'fontSize': '18px', 'color': '#e74c3c'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '15px'}),

        html.Div([
            html.Button('Next Round (Query Oracle)', id='next-round-btn', n_clicks=0,
                       style={
                           'padding': '10px 30px',
                           'fontSize': '16px',
                           'backgroundColor': '#3498db',
                           'color': 'white',
                           'border': 'none',
                           'borderRadius': '5px',
                           'cursor': 'pointer',
                           'marginRight': '10px'
                       }),
            html.Button('Reset', id='reset-btn', n_clicks=0,
                       style={
                           'padding': '10px 30px',
                           'fontSize': '16px',
                           'backgroundColor': '#95a5a6',
                           'color': 'white',
                           'border': 'none',
                           'borderRadius': '5px',
                           'cursor': 'pointer'
                       }),
        ], style={'display': 'flex', 'justifyContent': 'center'}),
    ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Message Display
    html.Div(id='status-message', children='Ready to start active learning...',
             style={
                 'textAlign': 'center',
                 'padding': '10px',
                 'backgroundColor': '#d5f4e6',
                 'borderRadius': '5px',
                 'marginBottom': '20px',
                 'fontSize': '14px',
                 'color': '#27ae60'
             }),

    # Two columns layout
    html.Div([
        # Left Column - Embedding View (PCA/UMAP)
        html.Div([
            html.H3("1. Embedding View", id='embedding-title', style={'textAlign': 'center', 'color': '#34495e'}),
            html.P("2D projection of EEG feature space (Training Set)",
                   style={'textAlign': 'center', 'fontSize': '12px', 'color': '#7f8c8d'}),
            dcc.Graph(id='pca-embedding', style={'height': '600px'}),

            html.Div([
                # Embedding Method Toggle
                html.Div([
                    html.Label("Embedding Method:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.RadioItems(
                        id='embedding-method',
                        options=[
                            {'label': ' PCA', 'value': 'pca'},
                            {'label': ' UMAP', 'value': 'umap', 'disabled': not UMAP_AVAILABLE}
                        ],
                        value='pca',
                        inline=True,
                        style={'fontSize': '14px'}
                    )
                ], style={'marginBottom': '10px'}),

                # View Mode
                html.Div([
                    html.Label("View Mode:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.RadioItems(
                        id='pca-view-mode',
                        options=[
                            {'label': ' Label Status', 'value': 'label_status'},
                            {'label': ' True Class', 'value': 'true_class'},
                            {'label': ' Uncertainty', 'value': 'uncertainty'}
                        ],
                        value='label_status',
                        inline=True,
                        style={'fontSize': '14px'}
                    )
                ])
            ], style={'textAlign': 'center', 'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        # Right Column - Uncertainty Histogram
        html.Div([
            html.H3("2. Uncertainty Distribution", style={'textAlign': 'center', 'color': '#34495e'}),
            html.P("Model confidence on unlabeled samples",
                   style={'textAlign': 'center', 'fontSize': '12px', 'color': '#7f8c8d'}),
            dcc.Graph(id='uncertainty-histogram', style={'height': '600px'}),

            html.Div([
                html.Label(f"Confidence Threshold: {confidence_threshold}",
                          id='threshold-label',
                          style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                html.P("Samples below threshold are queried from the oracle",
                      style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '0'})
            ], style={'textAlign': 'center', 'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'})

], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# ==========================================================
# CALLBACK: UPDATE ALL VISUALIZATIONS
# ==========================================================

@app.callback(
    [Output('pca-embedding', 'figure'),
     Output('uncertainty-histogram', 'figure'),
     Output('status-message', 'children'),
     Output('status-message', 'style'),
     Output('round-display', 'children'),
     Output('labeled-count', 'children'),
     Output('test-accuracy', 'children'),
     Output('embedding-title', 'children')],
    [Input('next-round-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks'),
     Input('pca-view-mode', 'value'),
     Input('embedding-method', 'value')]
)
def update_dashboard(next_clicks, reset_clicks, pca_view_mode, embedding_method):
    global model, labeled_idx, unlabeled_idx, current_round, learning_curve, oracle_annotated_idx, initial_labeled_idx

    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_id = 'init'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle Reset
    if trigger_id == 'reset-btn' and reset_clicks > 0:
        # Reset to initial state
        n_initial = int(initial_fraction * len(X_train))
        indices = np.random.permutation(len(X_train))
        labeled_idx = indices[:n_initial].copy()
        unlabeled_idx = indices[n_initial:].copy()
        initial_labeled_idx = labeled_idx.copy()
        oracle_annotated_idx = np.array([], dtype=int)
        current_round = 0
        learning_curve = []

        # Retrain initial model
        model = train_model(
            X_train[labeled_idx],
            y_train[labeled_idx],
            subjects_train[labeled_idx]
        )

        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        learning_curve.append((len(labeled_idx), test_acc))

        status_msg = "Reset to initial state!"
        status_style = {
            'textAlign': 'center', 'padding': '10px',
            'backgroundColor': '#d5f4e6', 'borderRadius': '5px',
            'marginBottom': '20px', 'fontSize': '14px', 'color': '#27ae60'
        }

    # Handle Next Round
    elif trigger_id == 'next-round-btn' and next_clicks > 0 and len(unlabeled_idx) > 0 and current_round < max_rounds:
        # Query uncertain samples
        probs = model.predict_proba(X_train[unlabeled_idx])
        max_probs = np.max(probs, axis=1)

        # Find samples below threshold
        low_conf_mask = max_probs < confidence_threshold
        candidate_indices = unlabeled_idx[low_conf_mask]
        candidate_conf = max_probs[low_conf_mask]

        if len(candidate_indices) == 0:
            status_msg = "⚠️ Model is confident on all remaining samples. Active learning complete!"
            status_style = {
                'textAlign': 'center', 'padding': '10px',
                'backgroundColor': '#fff3cd', 'borderRadius': '5px',
                'marginBottom': '20px', 'fontSize': '14px', 'color': '#856404'
            }
        else:
            # Select batch
            if len(candidate_indices) > batch_size:
                sorted_idx = np.argsort(candidate_conf)
                selected = sorted_idx[:batch_size]
                queried = candidate_indices[selected]
            else:
                queried = candidate_indices

            # Update pools
            labeled_idx = np.concatenate([labeled_idx, queried])
            unlabeled_idx = np.setdiff1d(unlabeled_idx, queried)
            oracle_annotated_idx = np.concatenate([oracle_annotated_idx, queried])  # Track oracle annotations
            current_round += 1

            # Retrain model
            model = train_model(
                X_train[labeled_idx],
                y_train[labeled_idx],
                subjects_train[labeled_idx]
            )

            # Evaluate
            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            learning_curve.append((len(labeled_idx), test_acc))

            status_msg = f"✓ Round {current_round} complete! Queried {len(queried)} samples from oracle. Test Acc: {test_acc:.4f}"
            status_style = {
                'textAlign': 'center', 'padding': '10px',
                'backgroundColor': '#d1ecf1', 'borderRadius': '5px',
                'marginBottom': '20px', 'fontSize': '14px', 'color': '#0c5460'
            }
    else:
        if current_round >= max_rounds:
            status_msg = "🎯 Maximum rounds reached!"
            status_style = {
                'textAlign': 'center', 'padding': '10px',
                'backgroundColor': '#f8d7da', 'borderRadius': '5px',
                'marginBottom': '20px', 'fontSize': '14px', 'color': '#721c24'
            }
        elif len(unlabeled_idx) == 0:
            status_msg = "🎉 All samples labeled!"
            status_style = {
                'textAlign': 'center', 'padding': '10px',
                'backgroundColor': '#d4edda', 'borderRadius': '5px',
                'marginBottom': '20px', 'fontSize': '14px', 'color': '#155724'
            }
        else:
            status_msg = "Ready for next round..."
            status_style = {
                'textAlign': 'center', 'padding': '10px',
                'backgroundColor': '#d5f4e6', 'borderRadius': '5px',
                'marginBottom': '20px', 'fontSize': '14px', 'color': '#27ae60'
            }

    # ==========================================================
    # VISUALIZATION 1: EMBEDDING VIEW (PCA or UMAP)
    # ==========================================================

    # Select embedding coordinates based on user choice
    if embedding_method == 'umap' and UMAP_AVAILABLE and X_train_umap is not None:
        embedding_coords = X_train_umap
        embedding_type = 'UMAP'
        coord_labels = {'x': 'UMAP 1', 'y': 'UMAP 2'}
    else:
        embedding_coords = X_train_pca
        embedding_type = 'PCA'
        coord_labels = {
            'x': f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} var)',
            'y': f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} var)'
        }

    # Create status arrays for all training samples
    sample_status = np.array(['Unlabeled'] * len(X_train))
    sample_status[labeled_idx] = 'Labeled'

    # Compute entropy-based uncertainties for all samples
    all_uncertainties = np.zeros(len(X_train))
    all_predictions = np.zeros(len(X_train))  # Store predicted class
    all_probs_class1 = np.zeros(len(X_train))  # Store P(seizure)

    if len(unlabeled_idx) > 0:
        unlabeled_probs = model.predict_proba(X_train[unlabeled_idx])
        all_uncertainties[unlabeled_idx] = binary_entropy(unlabeled_probs)
        all_predictions[unlabeled_idx] = model.predict(X_train[unlabeled_idx])
        all_probs_class1[unlabeled_idx] = unlabeled_probs[:, 1]

    # Compute uncertainties for labeled samples too (for visualization)
    if len(labeled_idx) > 0:
        labeled_probs = model.predict_proba(X_train[labeled_idx])
        all_uncertainties[labeled_idx] = binary_entropy(labeled_probs)
        all_predictions[labeled_idx] = model.predict(X_train[labeled_idx])
        all_probs_class1[labeled_idx] = labeled_probs[:, 1]

    # Identify samples queried in the most recent round (for visual marking)
    recently_queried = np.zeros(len(X_train), dtype=bool)
    if current_round > 0 and len(labeled_idx) >= batch_size:
        # Mark the last batch of labeled samples as recently queried
        recent_batch = labeled_idx[-batch_size:]
        recently_queried[recent_batch] = True

    # Create DataFrame for embedding plot
    embedding_df = pd.DataFrame({
        'Dim1': embedding_coords[:, 0],
        'Dim2': embedding_coords[:, 1],
        'True_Label': ['Seizure' if y == 1 else 'Non-Seizure' for y in y_train],
        'Predicted_Label': ['Seizure' if p == 1 else 'Non-Seizure' for p in all_predictions],
        'Status': sample_status,
        'Entropy': all_uncertainties,
        'Prob_Seizure': all_probs_class1,
        'Sample_ID': np.arange(len(X_train)),
        'Queried': ['Yes (Oracle Annotated)' if q else 'No' for q in recently_queried]
    })

    # Choose coloring based on view mode
    if pca_view_mode == 'label_status':
        color_col = 'Status'
        color_discrete_map = {'Labeled': '#27ae60', 'Unlabeled': '#95a5a6'}
        title_suffix = "(Labeled vs Unlabeled)"
        embedding_fig = px.scatter(
            embedding_df, x='Dim1', y='Dim2',
            color=color_col,
            color_discrete_map=color_discrete_map,
            hover_data={
                'Dim1': ':.3f',
                'Dim2': ':.3f',
                'True_Label': True,
                'Predicted_Label': True,
                'Entropy': ':.4f',
                'Prob_Seizure': ':.4f',
                'Sample_ID': True,
                'Queried': True
            },
            title=f"{embedding_type} Embedding: {title_suffix}",
            labels={'Dim1': coord_labels['x'],
                   'Dim2': coord_labels['y'],
                   'Entropy': 'Entropy H(p)'}
        )

    elif pca_view_mode == 'true_class':
        color_col = 'True_Label'
        color_discrete_map = {'Seizure': '#e74c3c', 'Non-Seizure': '#3498db'}
        title_suffix = "(True Class)"
        embedding_fig = px.scatter(
            embedding_df, x='Dim1', y='Dim2',
            color=color_col,
            color_discrete_map=color_discrete_map,
            hover_data={
                'Dim1': ':.3f',
                'Dim2': ':.3f',
                'Status': True,
                'Predicted_Label': True,
                'Entropy': ':.4f',
                'Prob_Seizure': ':.4f',
                'Sample_ID': True,
                'Queried': True
            },
            title=f"{embedding_type} Embedding: {title_suffix}",
            labels={'Dim1': coord_labels['x'],
                   'Dim2': coord_labels['y'],
                   'Entropy': 'Entropy H(p)'}
        )

    else:  # uncertainty mode
        title_suffix = "(Entropy-Based Uncertainty)"
        embedding_fig = px.scatter(
            embedding_df, x='Dim1', y='Dim2',
            color='Entropy',
            color_continuous_scale='Reds',
            hover_data={
                'Dim1': ':.3f',
                'Dim2': ':.3f',
                'True_Label': True,
                'Predicted_Label': True,
                'Status': True,
                'Prob_Seizure': ':.4f',
                'Sample_ID': True,
                'Queried': True
            },
            title=f"{embedding_type} Embedding: {title_suffix}",
            labels={'Dim1': coord_labels['x'],
                   'Dim2': coord_labels['y'],
                   'Entropy': 'Entropy H(p)'}
        )

    # Add visual marker for recently queried samples (bold border)
    embedding_fig.update_traces(
        marker=dict(
            opacity=0.7,
            line=dict(
                width=embedding_df['Queried'].apply(lambda x: 3 if x.startswith('Yes') else 0.5),
                color='black'
            )
        )
    )

    embedding_fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # ==========================================================
    # VISUALIZATION 2: UNCERTAINTY HISTOGRAM
    # ==========================================================

    # Compute entropy for ALL training samples (labeled + unlabeled)
    all_train_probs = model.predict_proba(X_train)
    all_train_entropy = binary_entropy(all_train_probs)
    all_train_confidence = np.max(all_train_probs, axis=1)

    hist_fig = go.Figure()

    # 1. Initial labeled samples (randomly selected at start) - shown in light green
    if len(initial_labeled_idx) > 0:
        initial_entropy = all_train_entropy[initial_labeled_idx]
        hist_fig.add_trace(go.Histogram(
            x=initial_entropy,
            name='Initial Labeled (Random)',
            marker_color='#52c41a',  # Light green
            opacity=0.7,
            nbinsx=30
        ))

    # 2. Oracle-annotated samples (queried during active learning) - shown in darker green
    if len(oracle_annotated_idx) > 0:
        oracle_entropy = all_train_entropy[oracle_annotated_idx]
        hist_fig.add_trace(go.Histogram(
            x=oracle_entropy,
            name='Oracle Annotated',
            marker_color='#237804',  # Dark green
            opacity=0.7,
            nbinsx=30
        ))

    # 3. Unlabeled samples - split by confidence
    if len(unlabeled_idx) > 0:
        unlabeled_entropy = all_train_entropy[unlabeled_idx]
        unlabeled_confidence = all_train_confidence[unlabeled_idx]

        # Confident unlabeled (auto-classified)
        above_threshold = unlabeled_confidence >= confidence_threshold
        if np.sum(above_threshold) > 0:
            hist_fig.add_trace(go.Histogram(
                x=unlabeled_entropy[above_threshold],
                name='Unlabeled - Confident (Auto)',
                marker_color='#3498db',
                opacity=0.7,
                nbinsx=30
            ))

        # Uncertain unlabeled (will be queried from oracle)
        below_threshold = unlabeled_confidence < confidence_threshold
        if np.sum(below_threshold) > 0:
            hist_fig.add_trace(go.Histogram(
                x=unlabeled_entropy[below_threshold],
                name='Unlabeled - Uncertain (Needs Oracle)',
                marker_color='#e74c3c',
                opacity=0.7,
                nbinsx=30
            ))

        # Add vertical line for entropy corresponding to confidence threshold
        threshold_p = confidence_threshold
        threshold_entropy = binary_entropy(np.array([[1-threshold_p, threshold_p]]))[0]

        hist_fig.add_vline(
            x=threshold_entropy,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Threshold @ conf={confidence_threshold} (H={threshold_entropy:.3f})",
            annotation_position="top"
        )

        # Add annotation about counts
        n_initial = len(initial_labeled_idx)
        n_oracle = len(oracle_annotated_idx)
        n_uncertain = np.sum(below_threshold)
        n_confident = np.sum(above_threshold)
        avg_unlabeled_entropy = unlabeled_entropy.mean()

        hist_fig.add_annotation(
            text=f"Initial: {n_initial} | Oracle: {n_oracle} | Unlabeled-Confident: {n_confident} | Unlabeled-Uncertain: {n_uncertain} | Avg Unlabeled H: {avg_unlabeled_entropy:.3f}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10, color='#34495e'),
            xanchor='center'
        )

        hist_fig.update_layout(
            title=f"Entropy Distribution (Total: {len(X_train)} samples)",
            xaxis_title="Binary Entropy H(p) = -p·log₂(p) - (1-p)·log₂(1-p)",
            yaxis_title="Count",
            barmode='stack',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        # All samples are labeled
        n_initial = len(initial_labeled_idx)
        n_oracle = len(oracle_annotated_idx)

        hist_fig.update_layout(
            title=f"Entropy Distribution (All {len(X_train)} samples labeled)",
            xaxis_title="Binary Entropy H(p)",
            yaxis_title="Count",
            barmode='stack',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        hist_fig.add_annotation(
            text=f"All samples labeled! Initial: {n_initial} | Oracle Annotated: {n_oracle}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color='#27ae60'),
            xanchor='center'
        )

    # Status displays
    round_display = f"Round {current_round}/{max_rounds}"
    labeled_count = f"{len(labeled_idx)}/{len(X_train)}"

    if len(learning_curve) > 0:
        current_test_acc = learning_curve[-1][1]
    else:
        # Compute current test accuracy if not in learning curve yet
        test_pred = model.predict(X_test)
        current_test_acc = accuracy_score(y_test, test_pred)

    test_acc_display = f"{current_test_acc:.4f}"

    # Update embedding title
    embedding_title = f"1. {embedding_type} Embedding View"

    return (embedding_fig, hist_fig, status_msg, status_style,
            round_display, labeled_count, test_acc_display, embedding_title)

# ==========================================================
# RUN APP
# ==========================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DASHBOARD READY!")
    print("="*70)
    print("Open your browser at: http://127.0.0.1:8050")
    print("Features:")
    print("  1. PCA Embedding View - Visualize sample distribution in 2D")
    print("  2. Uncertainty Histogram - See model confidence distribution")
    print("\nClick 'Next Round' to perform active learning iterations")
    print("="*70 + "\n")

    app.run(debug=True, port=8050)
