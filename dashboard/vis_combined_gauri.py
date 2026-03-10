# ==========================================================
# COMBINED ACTIVE LEARNING DASHBOARD (PART 1 + PART 2)
# Core active learning logic preserved from vis_part2.py
# Extra embedding/uncertainty visualizations from vis_part1.py
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

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px

# Captures runtime mode per embedding so startup logs can confirm DGrid status.
dgrid_runtime_modes = {}


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def extract_features(X_raw, fs=173.61):
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
        beta = band_power(13, 30)

        features.append([
            mean_val, std_val, rms, peak_to_peak,
            skewness, kurt_val,
            delta, theta, alpha, beta
        ])
    return np.array(features)


# ==========================================================
# LOAD + SPLIT
# ==========================================================

def load_and_split():
    df = pd.read_csv("bonn_eeg_combined.csv")

    X_raw = df.drop(["ID", "Y"], axis=1).values
    y_original = df["Y"].values

    X = extract_features(X_raw)
    y = (y_original == "E").astype(int)

    subjects = []
    for set_idx in range(5):
        for i in range(100):
            subject_within_set = i // 20
            subject_id = subject_within_set if set_idx < 2 else subject_within_set + 5
            subjects.append(subject_id)

    subjects = np.array(subjects)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    return (
        X_raw[train_idx],
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        subjects[train_idx]
    )


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(X_train, y_train, subjects_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=10000,
            random_state=42
        ))
    ])

    param_grid = {
        "clf__C": [1, 0.1, 0.01],
        "clf__penalty": ["l1", "l2"]
    }

    gkf = GroupKFold(n_splits=5)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf.split(X_train, y_train, groups=subjects_train),
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


# ==========================================================
# TERMINAL REPORT BUILDER
# ==========================================================

def build_terminal_report(
    round_number,
    labeled_count,
    train_acc,
    test_acc,
    cm,
    sensitivity,
    specificity,
    class_report,
    doctor_count,
    batch_auto_count,
    pool_auto_count,
):
    return f"""
======================================================================
ROUND {round_number}
Labeled samples: {labeled_count}

Train Accuracy: {round(train_acc, 4)}
Test Accuracy : {round(test_acc, 4)}

Confusion Matrix (Test):
{cm}

Sensitivity (Seizure Recall): {round(sensitivity, 4)}
Specificity (Non-Seizure Recall): {round(specificity, 4)}

Classification Report (Test):
{class_report}

There are {doctor_count} uncertain samples...
Annotation Summary:
No of samples to be labeled by the doctor: {doctor_count} / {batch_size}
Automatically classified (pool): {pool_auto_count}
Automatically classified (batch): {batch_auto_count}
======================================================================
"""


# ==========================================================
# DATA + CONSTANTS
# ==========================================================

X_raw_train, X_train, X_test, y_train, y_test, subjects_train = load_and_split()

initial_fraction = 0.2
batch_size = 10
default_confidence_threshold = 0.7


# ==========================================================
# EMBEDDING PRECOMPUTE (from vis_part1)
# ==========================================================

def _normalize_coords(coords):
    """Normalize 2D coordinates to [0, 1] range with stable handling of flat axes."""
    mins = np.min(coords, axis=0)
    ranges = np.ptp(coords, axis=0)
    ranges[ranges == 0] = 1.0
    return (coords - mins) / ranges


def _apply_internal_dgrid(coords):
    """Cluster-preserving spread: separate close points while keeping global structure."""
    n_samples = coords.shape[0]
    if n_samples <= 1:
        return coords.copy()

    mins = np.min(coords, axis=0)
    ranges = np.ptp(coords, axis=0)
    ranges[ranges == 0] = 1.0
    norm_coords = _normalize_coords(coords)
    spread = norm_coords.copy()
    velocity = np.zeros_like(spread)

    target_min_dist = 0.028
    step_size = 0.35
    damping = 0.9

    for _ in range(40):
        delta = spread[:, None, :] - spread[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(dist2)

        overlap = dist < target_min_dist
        if not np.any(overlap):
            break

        safe_dist = np.where(overlap, np.maximum(dist, 1e-6), 1.0)
        direction = delta / safe_dist[:, :, None]
        magnitude = np.where(overlap, target_min_dist - dist, 0.0)
        force = np.sum(direction * magnitude[:, :, None], axis=1)

        velocity = damping * velocity + step_size * force
        spread += velocity
        spread = np.clip(spread, 0.0, 1.0)

    # Keep the original cluster geometry dominant and apply only gentle spreading.
    blend = 0.35
    blended = (1.0 - blend) * norm_coords + blend * spread
    return blended * ranges + mins


def apply_dgrid_transform(coords, embedding_name="embedding"):
    """Apply DGrid-like layout using the internal fallback backend only."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        dgrid_runtime_modes[embedding_name] = "invalid-input"
        return coords

    transformed = _apply_internal_dgrid(coords)
    mode = "internal-soft-spread"

    if transformed.shape != coords.shape or not np.all(np.isfinite(transformed)):
        dgrid_runtime_modes[embedding_name] = "failed-fallback-to-original"
        return coords

    dgrid_runtime_modes[embedding_name] = mode

    return transformed

embedding_scaler = StandardScaler()
X_train_scaled = embedding_scaler.fit_transform(X_train)

pca_model = PCA(n_components=2, random_state=42)
X_train_pca = pca_model.fit_transform(X_train_scaled)
X_train_pca_dgrid = apply_dgrid_transform(X_train_pca, embedding_name="PCA")

if UMAP_AVAILABLE:
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_train_umap = umap_model.fit_transform(X_train_scaled)
    X_train_umap_dgrid = apply_dgrid_transform(X_train_umap, embedding_name="UMAP")
else:
    X_train_umap = None
    X_train_umap_dgrid = None
    dgrid_runtime_modes["UMAP"] = "not-available"

print(
    "[DGRID] "
    "backend=internal-soft-spread (dgrid-like concept) | "
    f"PCA={dgrid_runtime_modes.get('PCA', 'not-run')} | "
    f"UMAP={dgrid_runtime_modes.get('UMAP', 'not-run')}"
)


# ==========================================================
# HYBRID BATCH LOGIC
# ==========================================================

def compute_batch(confidence_threshold):
    probs = model.predict_proba(X_train[unlabeled_idx])
    max_probs = np.max(probs, axis=1)
    uncertainty = 1 - max_probs

    sorted_idx = np.argsort(uncertainty)[::-1]
    top_k = sorted_idx[:batch_size]

    batch_candidates = unlabeled_idx[top_k]
    batch_conf = max_probs[top_k]

    doctor_mask = batch_conf < confidence_threshold
    doctor_batch = batch_candidates[doctor_mask]

    batch_auto_count = len(batch_candidates) - len(doctor_batch)

    confident_mask = max_probs >= confidence_threshold

    batch_mask = np.zeros(len(unlabeled_idx), dtype=bool)
    batch_mask[top_k] = True

    pool_auto_count = np.sum(confident_mask & ~batch_mask)
    return doctor_batch, batch_auto_count, pool_auto_count


train_history = []
test_history = []
round_history = []
sensitivity_history = []
specificity_history = []
annotation_queue = []
selected_sample_id = None


def initialize_active_learning():
    global labeled_idx, unlabeled_idx
    global model, current_batch
    global batch_auto_count, pool_auto_count
    global current_pointer, round_number, phase
    global train_history, test_history, round_history
    global sensitivity_history, specificity_history
    global initial_labeled_idx, oracle_annotated_idx
    global current_confidence_threshold
    global annotation_queue, selected_sample_id

    train_history = []
    test_history = []
    round_history = []
    sensitivity_history = []
    specificity_history = []

    n_initial = int(initial_fraction * len(X_train))
    indices = np.random.permutation(len(X_train))

    labeled_idx = indices[:n_initial]
    unlabeled_idx = indices[n_initial:]

    initial_labeled_idx = labeled_idx.copy()
    oracle_annotated_idx = np.array([], dtype=int)

    round_number = 0
    phase = "annotation"

    current_confidence_threshold = default_confidence_threshold

    model = train_model(
        X_train[labeled_idx],
        y_train[labeled_idx],
        subjects_train[labeled_idx],
    )

    current_batch, batch_auto_count, pool_auto_count = compute_batch(current_confidence_threshold)
    annotation_queue = list(current_batch[:batch_size])
    current_pointer = 0
    selected_sample_id = annotation_queue[0] if len(annotation_queue) > 0 else None


def build_learning_curve():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=round_history,
        y=train_history,
        mode="lines+markers",
        name="Train Accuracy",
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=test_history,
        mode="lines+markers",
        name="Test Accuracy",
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=sensitivity_history,
        mode="lines+markers",
        name="Sensitivity",
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=specificity_history,
        mode="lines+markers",
        name="Specificity",
    ))

    max_round = max(1, max(round_history) if len(round_history) > 0 else 1)
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Performance",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, max_round], dtick=1),
        template="plotly_white",
    )

    return fig


def build_confusion_heatmap(cm):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred: Non-Seizure", "Pred: Seizure"],
        y=["Actual: Non-Seizure", "Actual: Seizure"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
    ))

    fig.update_layout(title=None)
    return fig


def build_data_donut():
    labeled = len(labeled_idx)
    doctor = len(current_batch)
    auto_batch = batch_auto_count
    auto_pool = pool_auto_count

    fig = go.Figure(data=[go.Pie(
        labels=["Labeled", "Auto (Pool)", "Auto (Batch)", "Needs Doctor"],
        values=[labeled, auto_pool, auto_batch, doctor],
        hole=0.5,
        sort=False,
        marker=dict(colors=["#27ae60", "#3498db", "#1d4ed8", "#e74c3c"]),
    )])

    fig.update_layout(title=None)
    return fig


def _build_embedding_boundary_trace(embedding_coords):
    """Build a 0.5-probability boundary in embedding space for visual interpretation."""
    if embedding_coords is None or len(embedding_coords) < 2:
        return None

    if len(np.unique(y_train)) < 2:
        return None

    x_min, x_max = np.min(embedding_coords[:, 0]), np.max(embedding_coords[:, 0])
    y_min, y_max = np.min(embedding_coords[:, 1]), np.max(embedding_coords[:, 1])
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 120),
        np.linspace(y_min - y_pad, y_max + y_pad, 120),
    )

    boundary_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    boundary_model.fit(embedding_coords, y_train)
    zz = boundary_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    return go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=zz,
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=1, coloring="none", showlines=True),
        line=dict(color="black", width=2, dash="dash"),
        name="Decision Boundary",
        hoverinfo="skip",
        showlegend=True,
    )


def build_embedding_figure(pca_view_mode, embedding_method, confidence_threshold):
    if embedding_method == "umap" and UMAP_AVAILABLE and X_train_umap_dgrid is not None:
        embedding_coords = X_train_umap_dgrid
        embedding_type = "UMAP"
        coord_labels = {"x": "UMAP 1", "y": "UMAP 2"}
    else:
        embedding_coords = X_train_pca_dgrid
        embedding_type = "PCA"
        coord_labels = {
            "x": f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%} var)",
            "y": f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%} var)",
        }

    sample_status = np.array(["Unlabeled"] * len(X_train))
    sample_status[labeled_idx] = "Labeled"

    all_uncertainties = np.zeros(len(X_train))
    all_predictions = np.zeros(len(X_train))
    all_probs_class1 = np.zeros(len(X_train))

    if len(unlabeled_idx) > 0:
        unlabeled_probs = model.predict_proba(X_train[unlabeled_idx])
        all_uncertainties[unlabeled_idx] = 1 - np.max(unlabeled_probs, axis=1)
        all_predictions[unlabeled_idx] = model.predict(X_train[unlabeled_idx])
        all_probs_class1[unlabeled_idx] = unlabeled_probs[:, 1]

    if len(labeled_idx) > 0:
        labeled_probs = model.predict_proba(X_train[labeled_idx])
        all_uncertainties[labeled_idx] = 1 - np.max(labeled_probs, axis=1)
        all_predictions[labeled_idx] = model.predict(X_train[labeled_idx])
        all_probs_class1[labeled_idx] = labeled_probs[:, 1]

    all_probs_class0 = 1 - all_probs_class1
    threshold_flag = np.where((1 - all_uncertainties) >= confidence_threshold, "Confident", "Uncertain")

    recently_queried = np.zeros(len(X_train), dtype=bool)
    if len(oracle_annotated_idx) > 0:
        recently_queried[oracle_annotated_idx] = True

    embedding_df = pd.DataFrame({
        "Dim1": embedding_coords[:, 0],
        "Dim2": embedding_coords[:, 1],
        "True_Label": ["Seizure" if y == 1 else "Non-Seizure" for y in y_train],
        "Predicted_Label": ["Seizure" if p == 1 else "Non-Seizure" for p in all_predictions],
        "Status": sample_status,
        "Uncertainty": all_uncertainties,
        "Prob_Seizure": all_probs_class1,
        "Prob_Non_Seizure": all_probs_class0,
        "Sample_ID": np.arange(len(X_train)),
        "Queried": ["Yes (Oracle Annotated)" if q else "No" for q in recently_queried],
        "Threshold_Status": threshold_flag,
    })

    if pca_view_mode == "label_status":
        embedding_fig = px.scatter(
            embedding_df,
            x="Dim1",
            y="Dim2",
            color="Status",
            custom_data=["Sample_ID"],
            color_discrete_map={"Labeled": "#27ae60", "Unlabeled": "#95a5a6"},
            hover_data={
                "Dim1": ":.3f",
                "Dim2": ":.3f",
                "True_Label": True,
                "Predicted_Label": True,
                "Uncertainty": ":.4f",
                "Prob_Non_Seizure": ":.4f",
                "Prob_Seizure": ":.4f",
                "Sample_ID": True,
                "Queried": True,
                "Threshold_Status": True,
            },
            title=None,
            labels={"Dim1": coord_labels["x"], "Dim2": coord_labels["y"]},
        )
    elif pca_view_mode == "true_class":
        embedding_fig = px.scatter(
            embedding_df,
            x="Dim1",
            y="Dim2",
            color="True_Label",
            custom_data=["Sample_ID"],
            color_discrete_map={"Seizure": "#e74c3c", "Non-Seizure": "#3498db"},
            hover_data={
                "Dim1": ":.3f",
                "Dim2": ":.3f",
                "Status": True,
                "Predicted_Label": True,
                "Uncertainty": ":.4f",
                "Prob_Non_Seizure": ":.4f",
                "Prob_Seizure": ":.4f",
                "Sample_ID": True,
                "Queried": True,
                "Threshold_Status": True,
            },
            title=None,
            labels={"Dim1": coord_labels["x"], "Dim2": coord_labels["y"]},
        )
    else:
        embedding_fig = px.scatter(
            embedding_df,
            x="Dim1",
            y="Dim2",
            color="Uncertainty",
            custom_data=["Sample_ID"],
            color_continuous_scale="Reds",
            hover_data={
                "Dim1": ":.3f",
                "Dim2": ":.3f",
                "True_Label": True,
                "Predicted_Label": True,
                "Status": True,
                "Prob_Non_Seizure": ":.4f",
                "Prob_Seizure": ":.4f",
                "Sample_ID": True,
                "Queried": True,
                "Threshold_Status": True,
            },
            title=None,
            labels={"Dim1": coord_labels["x"], "Dim2": coord_labels["y"]},
        )

    border_width = [3 if q.startswith("Yes") else 0.5 for q in embedding_df["Queried"]]
    embedding_fig.update_traces(
        marker=dict(opacity=0.7, line=dict(width=border_width, color="black")),
        selector=dict(type="scatter"),
    )

    if selected_sample_id is not None:
        selected_mask = embedding_df["Sample_ID"] == selected_sample_id
        if selected_mask.any():
            embedding_fig.add_trace(
                go.Scatter(
                    x=embedding_df[selected_mask]["Dim1"],
                    y=embedding_df[selected_mask]["Dim2"],
                    mode="markers",
                    marker=dict(size=16, color="#fde047", line=dict(width=3, color="black")),
                    name="Selected EEG",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    boundary_trace = _build_embedding_boundary_trace(embedding_coords)
    if boundary_trace is not None:
        embedding_fig.add_trace(boundary_trace)
        embedding_fig.data = (embedding_fig.data[-1],) + embedding_fig.data[:-1]

    x_min = float(embedding_df["Dim1"].min())
    x_max = float(embedding_df["Dim1"].max())
    y_min = float(embedding_df["Dim2"].min())
    y_max = float(embedding_df["Dim2"].max())
    x_pad = 0.03 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.03 * (y_max - y_min if y_max > y_min else 1.0)

    embedding_fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], automargin=False)
    embedding_fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], automargin=False)
    embedding_fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=10, b=28, l=36, r=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return embedding_fig, embedding_type


def build_uncertainty_histogram(confidence_threshold):
    all_train_probs = model.predict_proba(X_train)
    all_train_uncertainty = 1 - np.max(all_train_probs, axis=1)
    all_train_confidence = np.max(all_train_probs, axis=1)

    hist_fig = go.Figure()

    if len(initial_labeled_idx) > 0:
        hist_fig.add_trace(go.Histogram(
            x=all_train_uncertainty[initial_labeled_idx],
            name="Initial Labeled (Random)",
            marker_color="#52c41a",
            opacity=0.7,
            nbinsx=30,
        ))

    if len(oracle_annotated_idx) > 0:
        hist_fig.add_trace(go.Histogram(
            x=all_train_uncertainty[oracle_annotated_idx],
            name="Oracle Annotated",
            marker_color="#237804",
            opacity=0.7,
            nbinsx=30,
        ))

    if len(unlabeled_idx) > 0:
        unlabeled_uncertainty = all_train_uncertainty[unlabeled_idx]
        unlabeled_confidence = all_train_confidence[unlabeled_idx]

        above_threshold = unlabeled_confidence >= confidence_threshold
        below_threshold = unlabeled_confidence < confidence_threshold

        if np.sum(above_threshold) > 0:
            hist_fig.add_trace(go.Histogram(
                x=unlabeled_uncertainty[above_threshold],
                name="Unlabeled - Confident (Auto)",
                marker_color="#3498db",
                opacity=0.7,
                nbinsx=30,
            ))

        if np.sum(below_threshold) > 0:
            hist_fig.add_trace(go.Histogram(
                x=unlabeled_uncertainty[below_threshold],
                name="Unlabeled - Uncertain (Needs Oracle)",
                marker_color="#e74c3c",
                opacity=0.7,
                nbinsx=30,
            ))

        threshold_uncertainty = 1 - confidence_threshold
        hist_fig.add_vline(
            x=threshold_uncertainty,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=(
                f"Threshold @ conf={confidence_threshold:.2f} "
                f"(uncertainty={threshold_uncertainty:.2f})"
            ),
            annotation_position="top",
        )

    hist_fig.update_layout(
        title=None,
        title_font=dict(size=14),
        xaxis_title="Uncertainty (1 - max_confidence)",
        yaxis_title="Count",
        barmode="stack",
        template="plotly_white",
        margin=dict(t=56, b=72, l=50, r=24),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    return hist_fig


# Initialize first time
initialize_active_learning()


# ==========================================================
# DASH APP
# ==========================================================

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Location(id="url", refresh=True),

    # Top bar
    html.Div([
        html.Div([
            html.H2(
                "NeuroLens Clinical Assistant",
                style={"margin": "0", "fontSize": "22px", "color": "#0f172a", "letterSpacing": "0.2px"},
            ),
            html.Div(id="status-message", style={"fontWeight": "600", "color": "#334155", "fontSize": "13px", "marginTop": "2px"}),
        ]),
        html.Div([
            html.Button(
                "Annotate (Oracle)",
                id="annotate-btn",
                style={
                    "backgroundColor": "#1d4ed8",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "fontWeight": "600",
                    "cursor": "pointer",
                },
            ),
            html.Button(
                "Train Model",
                id="train-btn",
                disabled=True,
                style={
                    "marginLeft": "8px",
                    "backgroundColor": "#0f766e",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "fontWeight": "600",
                    "cursor": "pointer",
                },
            ),
            html.Button(
                "Reset",
                id="reset-btn",
                style={
                    "marginLeft": "8px",
                    "backgroundColor": "#475569",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "fontWeight": "600",
                    "cursor": "pointer",
                },
            ),
            html.Button(
                "Load Top-K Uncertain",
                id="load-uncertain-btn",
                style={
                    "marginLeft": "8px",
                    "backgroundColor": "#7c3aed",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "fontWeight": "600",
                    "cursor": "pointer",
                },
            ),
            html.Button(
                "Add to Annotation Batch",
                id="add-to-batch-btn",
                style={
                    "marginLeft": "8px",
                    "backgroundColor": "#b45309",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "fontWeight": "600",
                    "cursor": "pointer",
                },
            ),
        ]),
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "8px 12px",
        "backgroundColor": "#f8fafc",
        "border": "1px solid #e2e8f0",
        "borderRadius": "10px",
        "marginBottom": "6px",
    }),

    # KPI + universal confidence control
    html.Div([
        html.Div([
            html.Span("Round", style={"fontWeight": "700", "color": "#334155"}),
            html.Span(id="round-display", style={"marginLeft": "6px", "color": "#0f172a"}),
        ]),
        html.Div([
            html.Span("Labeled", style={"fontWeight": "700", "color": "#334155"}),
            html.Span(id="labeled-count", style={"marginLeft": "6px", "color": "#0f172a"}),
        ]),
        html.Div([
            html.Span("Test Acc", style={"fontWeight": "700", "color": "#334155"}),
            html.Span(id="test-accuracy", style={"marginLeft": "6px", "color": "#0f172a"}),
        ]),
        html.Div([
            html.Span("Confidence", style={"fontWeight": "700", "color": "#334155"}),
            html.Span(id="confidence-value", style={"marginLeft": "6px", "color": "#0f172a"}),
        ]),
        html.Div(id="threshold-label", style={"fontWeight": "600", "color": "#475569"}),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(5, minmax(0, 1fr))",
        "alignItems": "center",
        "gap": "6px",
        "padding": "6px 10px",
        "backgroundColor": "#ffffff",
        "border": "1px solid #e2e8f0",
        "borderRadius": "10px",
        "marginBottom": "6px",
    }),

    html.Div([
        dcc.Slider(
            id="confidence-slider",
            min=0.5,
            max=0.99,
            step=0.01,
            value=default_confidence_threshold,
            marks={0.5: "0.5", 0.6: "0.6", 0.7: "0.7", 0.8: "0.8", 0.9: "0.9", 0.99: "0.99"},
        )
    ], style={
        "padding": "6px 12px 2px 12px",
        "backgroundColor": "#ffffff",
        "border": "1px solid #e2e8f0",
        "borderRadius": "10px",
        "marginBottom": "6px",
    }),

    # Main workspace
    html.Div([
        html.Div([
            html.Div([
                html.H3(id="embedding-title", children="PCA Embedding View", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "16px"}),
                dcc.Graph(id="pca-embedding", style={"height": "calc(100% - 72px)"}),
                html.Div([
                    html.Label("Embedding", style={"fontWeight": "700", "color": "#334155", "fontSize": "12px"}),
                    dcc.RadioItems(
                        id="embedding-method",
                        options=[
                            {"label": " PCA", "value": "pca"},
                            {"label": " UMAP", "value": "umap", "disabled": not UMAP_AVAILABLE},
                        ],
                        value="pca",
                        inline=True,
                        style={"fontSize": "12px"},
                    ),
                    html.Label("View", style={"fontWeight": "700", "color": "#334155", "marginLeft": "10px", "fontSize": "12px"}),
                    dcc.RadioItems(
                        id="pca-view-mode",
                        options=[
                            {"label": " Label Status", "value": "label_status"},
                            {"label": " True Class", "value": "true_class"},
                            {"label": " Uncertainty", "value": "uncertainty"},
                        ],
                        value="label_status",
                        inline=True,
                        style={"fontSize": "12px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px", "flexWrap": "wrap", "paddingTop": "4px"}),
            ], style={
                "width": "42%",
                "height": "100%",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "10px",
                "padding": "6px",
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
            }),
            html.Div([
                html.H3("Uncertainty Distribution", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "16px"}),
                dcc.Graph(id="uncertainty-histogram", style={"height": "calc(100% - 30px)"}),
            ], style={
                "width": "36%",
                "height": "100%",
                "marginLeft": "0.5%",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "10px",
                "padding": "6px",
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
            }),
            html.Div([
                html.H3("Confusion Matrix", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "16px"}),
                dcc.Graph(id="confusion-heatmap", style={"height": "calc(100% - 30px)"}),
            ], style={
                "width": "22%",
                "height": "100%",
                "marginLeft": "0.5%",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "10px",
                "padding": "6px",
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
            }),
        ], style={
            "height": "42%",
            "display": "flex",
            "gap": "6px",
            "minHeight": 0,
        }),
        html.Div([
            html.Div([
                html.H3("EEG Annotation Panel", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "16px"}),
                dcc.Graph(id="eeg-graph", style={"height": "calc(100% - 30px)"}),
            ], style={
                "width": "63.5%",
                "height": "100%",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "10px",
                "padding": "6px",
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
            }),
            html.Div([
                html.H3("Secondary Analytics", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "15px"}),
                dcc.Tabs(
                    value="tab-learning",
                    style={"height": "calc(100% - 30px)", "minHeight": 0},
                    children=[
                        dcc.Tab(
                            label="Learning Rate",
                            value="tab-learning",
                            children=[
                                dcc.Graph(id="learning-curve", style={"height": "100%"}),
                            ],
                            style={"padding": "6px", "fontSize": "12px"},
                            selected_style={"padding": "6px", "fontSize": "12px", "fontWeight": "700"},
                        ),
                        dcc.Tab(
                            label="Data Distribution",
                            value="tab-dd",
                            children=[
                                dcc.Graph(id="data-donut", style={"height": "100%"}),
                            ],
                            style={"padding": "6px", "fontSize": "12px"},
                            selected_style={"padding": "6px", "fontSize": "12px", "fontWeight": "700"},
                        ),
                    ],
                ),
            ], style={
                "width": "36%",
                "height": "100%",
                "marginLeft": "0.5%",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "10px",
                "padding": "6px",
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
            }),
        ], style={
            "height": "58%",
            "marginTop": "18px",
            "display": "flex",
            "gap": "6px",
            "minHeight": 0,
        }),
    ], style={
        "height": "calc(100vh - 166px)",
        "overflow": "hidden",
        "minHeight": 0,
    }),
], style={
    "height": "100vh",
    "padding": "6px",
    "boxSizing": "border-box",
    "fontFamily": "'Segoe UI', 'Helvetica Neue', sans-serif",
    "backgroundColor": "#f1f5f9",
    "overflow": "hidden",
})


# ==========================================================
# MAIN CALLBACK
# ==========================================================

@app.callback(
    Output("eeg-graph", "figure"),
    Output("annotate-btn", "disabled"),
    Output("train-btn", "disabled"),
    Output("confidence-value", "children"),
    Output("status-message", "children"),
    Output("learning-curve", "figure"),
    Output("confusion-heatmap", "figure"),
    Output("data-donut", "figure"),
    Output("pca-embedding", "figure"),
    Output("uncertainty-histogram", "figure"),
    Output("round-display", "children"),
    Output("labeled-count", "children"),
    Output("test-accuracy", "children"),
    Output("embedding-title", "children"),
    Output("threshold-label", "children"),
    Input("url", "pathname"),
    Input("annotate-btn", "n_clicks"),
    Input("train-btn", "n_clicks"),
    Input("confidence-slider", "value"),
    Input("reset-btn", "n_clicks"),
    Input("pca-view-mode", "value"),
    Input("embedding-method", "value"),
    Input("pca-embedding", "clickData"),
    Input("add-to-batch-btn", "n_clicks"),
    Input("load-uncertain-btn", "n_clicks"),
    prevent_initial_call=False,
)
def update_dashboard(
    pathname,
    annotate_clicks,
    train_clicks,
    confidence_value,
    reset_clicks,
    pca_view_mode,
    embedding_method,
    embedding_click_data,
    add_to_batch_clicks,
    load_uncertain_clicks,
):
    global labeled_idx, unlabeled_idx
    global current_batch, current_pointer
    global batch_auto_count, pool_auto_count
    global model, phase, round_number
    global current_confidence_threshold, oracle_annotated_idx
    global annotation_queue, selected_sample_id

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger_id in ["url", "reset-btn"]:
        initialize_active_learning()

    status_message = "Annotation Phase"

    if trigger_id == "pca-embedding" and embedding_click_data:
        selected_sample_id = int(embedding_click_data["points"][0]["customdata"][0])
        status_message = f"Previewing Sample {selected_sample_id}"

    if trigger_id == "add-to-batch-btn":
        if selected_sample_id is None:
            status_message = "No sample selected from embedding"
        elif selected_sample_id in labeled_idx:
            status_message = f"Sample {selected_sample_id} is already labeled"
        elif selected_sample_id in annotation_queue:
            status_message = f"Sample {selected_sample_id} already in queue"
        elif len(annotation_queue) >= batch_size:
            status_message = f"Annotation queue full ({batch_size})"
        else:
            annotation_queue.append(selected_sample_id)
            status_message = f"Added Sample {selected_sample_id} to queue"
            if phase != "annotation":
                phase = "annotation"
                current_pointer = 0

    if trigger_id == "load-uncertain-btn":
        current_batch, batch_auto_count, pool_auto_count = compute_batch(current_confidence_threshold)
        annotation_queue = list(current_batch[:batch_size])
        current_pointer = 0
        phase = "annotation"
        selected_sample_id = annotation_queue[0] if len(annotation_queue) > 0 else selected_sample_id
        status_message = f"Loaded Top-K uncertain queue ({len(annotation_queue)}/{batch_size})"

    if trigger_id == "confidence-slider":
        current_confidence_threshold = confidence_value
        current_batch, batch_auto_count, pool_auto_count = compute_batch(current_confidence_threshold)
        if phase == "annotation":
            status_message = "Confidence updated"
        else:
            status_message = "Confidence updated and reflected in all visualizations"

    if trigger_id == "annotate-btn" and phase == "annotation":
        if len(annotation_queue) > 0 and current_pointer < len(annotation_queue):
            sample_id = int(annotation_queue[current_pointer])
            if sample_id in unlabeled_idx:
                labeled_idx = np.append(labeled_idx, sample_id)
                unlabeled_idx = unlabeled_idx[unlabeled_idx != sample_id]
                oracle_annotated_idx = np.append(oracle_annotated_idx, sample_id)

            current_pointer += 1
            if current_pointer < len(annotation_queue):
                selected_sample_id = int(annotation_queue[current_pointer])

            if current_pointer >= len(annotation_queue):
                phase = "training"
                status_message = "Queue complete. Please Train the Model."

    if trigger_id == "train-btn" and phase == "training":
        round_number += 1

        model = train_model(
            X_train[labeled_idx],
            y_train[labeled_idx],
            subjects_train[labeled_idx],
        )

        current_batch, batch_auto_count, pool_auto_count = compute_batch(confidence_value)
        annotation_queue = []
        current_pointer = 0
        phase = "annotation"
        current_confidence_threshold = confidence_value
        selected_sample_id = None
        status_message = "Model trained. Load Top-K or select samples from embedding."

    train_acc = accuracy_score(y_train[labeled_idx], model.predict(X_train[labeled_idx]))
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    if len(round_history) == 0:
        train_history.append(train_acc)
        test_history.append(test_acc)
        sensitivity_history.append(sensitivity)
        specificity_history.append(specificity)
        round_history.append(0)

    if trigger_id == "train-btn" and phase == "annotation":
        if len(round_history) == 0 or round_history[-1] != round_number:
            train_history.append(train_acc)
            test_history.append(test_acc)
            sensitivity_history.append(sensitivity)
            specificity_history.append(specificity)
            round_history.append(round_number)

    pending_queue = max(0, len(annotation_queue) - current_pointer)

    if selected_sample_id is not None:
        eeg_fig = go.Figure(data=[go.Scatter(y=X_raw_train[selected_sample_id], mode="lines")])
    elif pending_queue > 0:
        preview_idx = int(annotation_queue[current_pointer])
        eeg_fig = go.Figure(data=[go.Scatter(y=X_raw_train[preview_idx], mode="lines")])
    else:
        eeg_fig = go.Figure()

    annotate_disabled = not (phase == "annotation" and pending_queue > 0)
    train_disabled = phase != "training"

    if pending_queue == 0 and phase == "annotation" and len(annotation_queue) == 0:
        status_message = "Queue empty. Load Top-K uncertain or click embedding and add samples."

    if trigger_id in [None, "url", "reset-btn"]:
        pending_queue = max(0, len(annotation_queue) - current_pointer)
        status_message = (
            f"Annotation Phase | Queue: {len(annotation_queue)}/{batch_size} | Pending: {pending_queue}"
        )
        annotate_disabled = not (phase == "annotation" and pending_queue > 0)
        train_disabled = phase != "training"

    heatmap_fig = build_confusion_heatmap(cm)
    donut_fig = build_data_donut()
    learning_curve_fig = build_learning_curve()

    embedding_fig, embedding_type = build_embedding_figure(
        pca_view_mode,
        embedding_method,
        current_confidence_threshold,
    )
    uncertainty_hist = build_uncertainty_histogram(current_confidence_threshold)

    round_display = f"Round {round_number}"
    labeled_count = f"{len(labeled_idx)}/{len(X_train)}"
    test_accuracy_display = f"{test_acc:.4f}"
    queue_status = f"Annotation Queue: {len(annotation_queue)}/{batch_size} | Pending: {pending_queue}"

    return (
        eeg_fig,
        annotate_disabled,
        train_disabled,
        f"{confidence_value:.2f}",
        status_message,
        learning_curve_fig,
        heatmap_fig,
        donut_fig,
        embedding_fig,
        uncertainty_hist,
        round_display,
        labeled_count,
        test_accuracy_display,
        f"{embedding_type} Embedding View",
        queue_status,
    )


if __name__ == "__main__":
    app.run(debug=True)
