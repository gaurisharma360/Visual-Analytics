# ==========================================================
# COMBINED ACTIVE LEARNING DASHBOARD - FINAL VERSION
# Base: vis_combined_gourisha.py (with feature explanation)
# Added: Interactive embedding-to-EEG from vis_combined_gauri.py
# ==========================================================
#
#

import warnings
import importlib
import os
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, maximum_filter1d, minimum_filter1d

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
    UMAP_IMPORT_ERROR = None
except ImportError:
    UMAP_AVAILABLE = False
    UMAP_IMPORT_ERROR = "ImportError: install 'umap-learn' in the active environment"
    UMAP = None
except Exception as exc:
    UMAP_AVAILABLE = False
    UMAP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    UMAP = None

try:
    shap = importlib.import_module("shap")
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except ImportError:
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = "ImportError: install 'shap' in the active environment"
    shap = None
except Exception as exc:
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    shap = None

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

# ==========================================================
# GLOBAL RANDOM SEED
# ==========================================================

GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)

# Classify as seizure when P(seizure) >= 0.30 to prioritize recall and reduce false negatives.
SEIZURE_PROB_THRESHOLD = 0.30

# Captures runtime mode per embedding so startup logs can confirm DGrid status.
dgrid_runtime_modes = {}

# Canonical runtime keys used across embedding compute + logging.
DGRID_KEY_UMAP_DATA = "UMAP-DATA"
DGRID_KEY_UMAP_MODEL = "UMAP-MODEL"


def predict_binary_with_threshold(X):
    """Predict binary labels using the custom seizure probability threshold."""
    seizure_probs = model.predict_proba(X)[:, 1]
    return (seizure_probs >= SEIZURE_PROB_THRESHOLD).astype(int)


def _print_dgrid_status(tag):
    """Print concise runtime status for data/model embedding transforms."""
    print(
        "[DGRID] "
        f"{tag} | "
        "backend=internal-soft-spread (dgrid-like concept) | "
        f"UMAP-DATA={dgrid_runtime_modes.get(DGRID_KEY_UMAP_DATA, 'pending')} | "
        f"UMAP-MODEL={dgrid_runtime_modes.get(DGRID_KEY_UMAP_MODEL, 'pending')}"
    )


# ==========================================================
# DGRID TRANSFORMATION - Prevents overlapping points in embedding views
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

    # MODERATE: Just enough to reduce overlaps, preserve structure
    target_min_dist = 0.032  # Small increase from 0.028
    step_size = 0.40  # Small increase from 0.35
    damping = 0.88  # Slightly reduced

    for _ in range(50):  # 50 iterations (between 40 and 80)
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

    # MODERATE: 45% spread (between original 35% and aggressive 70%)
    blend = 0.45
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


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def extract_features(X_raw, fs=173.61):
    features = []
    for signal in X_raw:
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms = np.sqrt(np.mean(signal ** 2))
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
# SIGNAL PROCESSING HELPERS FOR FEATURE→EEG MAPPING
# ==========================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def highlight_feature_in_eeg(signal, feature_name, fs=173.61):
    """
    Highlight the EEG regions corresponding to a specific feature.
    Returns indices/ranges to highlight based on the feature type.
    """
    signal = np.asarray(signal, dtype=float)
    window_size = max(8, int(fs * 0.5))

    def _window_std(x):
        mean_x = uniform_filter1d(x, size=window_size, mode="nearest")
        mean_sq_x = uniform_filter1d(x ** 2, size=window_size, mode="nearest")
        return np.sqrt(np.maximum(mean_sq_x - mean_x ** 2, 0.0))

    def _window_rms(x):
        return np.sqrt(uniform_filter1d(x ** 2, size=window_size, mode="nearest"))

    def _window_p2p(x):
        return maximum_filter1d(x, size=window_size, mode="nearest") - minimum_filter1d(x, size=window_size, mode="nearest")

    def _top_mask(score, pct=75):
        threshold = np.percentile(score, pct)
        return score >= threshold

    if feature_name == "Delta Power":
        # Highlight slow oscillations (0.5-4 Hz).
        filtered = bandpass_filter(signal, 0.5, 4, fs)
        return filtered, "Delta band (0.5-4 Hz)", None

    elif feature_name == "Theta Power":
        # Highlight 4-8 Hz oscillations.
        filtered = bandpass_filter(signal, 4, 8, fs)
        return filtered, "Theta band (4-8 Hz)", None

    elif feature_name == "Alpha Power":
        # Highlight 8-13 Hz oscillations.
        filtered = bandpass_filter(signal, 8, 13, fs)
        return filtered, "Alpha band (8-13 Hz)", None

    elif feature_name == "Beta Power":
        # Highlight 13-30 Hz oscillations.
        filtered = bandpass_filter(signal, 13, 30, fs)
        return filtered, "Beta band (13-30 Hz)", None

    elif feature_name == "Std Dev" or feature_name == "RMS":
        # Highlight high-variance/high-energy regions using rolling statistics.
        score = _window_std(signal) if feature_name == "Std Dev" else _window_rms(signal)
        return score, "High variability regions", _top_mask(score)

    elif feature_name == "Peak-to-Peak":
        score = _window_p2p(signal)
        return score, "Large amplitude spikes", _top_mask(score)

    elif feature_name == "Kurtosis":
        # Sharp transients are emphasized by residual energy.
        smooth = gaussian_filter1d(signal, sigma=4)
        residual = np.abs(signal - smooth)
        return residual, "Sharp peaks/spikes", _top_mask(residual, pct=85)

    elif feature_name == "Skewness":
        # Approximate local asymmetry with rolling standardized third moment.
        smooth = gaussian_filter1d(signal, sigma=2)
        local_mean = uniform_filter1d(smooth, size=window_size, mode="nearest")
        centered = smooth - local_mean
        local_std = np.sqrt(np.maximum(uniform_filter1d(centered ** 2, size=window_size, mode="nearest"), 1e-8))
        local_skew = uniform_filter1d((centered / local_std) ** 3, size=window_size, mode="nearest")
        skew_mask = np.abs(local_skew) >= np.percentile(np.abs(local_skew), 75)
        return local_skew, "Asymmetric waveform segments", skew_mask

    elif feature_name == "Mean":
        # Highlight strongest baseline-shift regions.
        baseline = uniform_filter1d(signal, size=int(fs))
        baseline_shift = np.abs(signal - baseline)
        return baseline_shift, "Baseline-shift regions", _top_mask(baseline_shift, pct=80)

    else:
        return signal, "Raw signal", None


FEATURE_NAMES = [
    "Mean", "Std Dev", "RMS", "Peak-to-Peak",
    "Skewness", "Kurtosis",
    "Delta Power", "Theta Power", "Alpha Power", "Beta Power",
]

FEATURE_HIGHLIGHT_PALETTE = [
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#8b5cf6",  # violet
    "#ef4444",  # red
    "#0ea5e9",  # sky
    "#f97316",  # orange
    "#14b8a6",  # teal
    "#e11d48",  # rose
    "#22c55e",  # green
    "#6366f1",  # indigo
]
FEATURE_HIGHLIGHT_COLORS = {
    name: FEATURE_HIGHLIGHT_PALETTE[i % len(FEATURE_HIGHLIGHT_PALETTE)]
    for i, name in enumerate(FEATURE_NAMES)
}

shap_explainer = None


def initialize_shap_explainer():
    """Initialize SHAP explainer for the current trained pipeline."""
    global shap_explainer

    shap_explainer = None

    if not SHAP_AVAILABLE:
        print(f"[SHAP] UNAVAILABLE | {SHAP_IMPORT_ERROR}")
        return

    if model is None:
        print("[SHAP] SKIPPED | model not ready yet")
        return

    try:
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        background_scaled = scaler.transform(X_train)
        shap_explainer = shap.LinearExplainer(clf, background_scaled)
        print(f"[SHAP] OK | background_shape={background_scaled.shape}")
    except Exception as exc:
        shap_explainer = None
        print(f"[SHAP] FAILED | {type(exc).__name__}: {exc}")


def compute_feature_contributions(sample_idx):
    """Compute exact linear contributions: logistic_weight * scaled_feature_value."""
    if not hasattr(model.named_steps['clf'], 'coef_'):
        return None, None, None

    weights = model.named_steps['clf'].coef_[0]
    sample_features = X_train[sample_idx:sample_idx + 1]
    sample_scaled = model.named_steps['scaler'].transform(sample_features)[0]
    contributions = weights * sample_scaled
    probs = model.predict_proba(sample_features)[0]
    return np.asarray(FEATURE_NAMES), contributions, probs


def compute_feature_attributions(sample_idx):
    """Compute per-feature attributions and additive baseline for one sample."""
    sample_features = X_train[sample_idx:sample_idx + 1]
    probs = model.predict_proba(sample_features)[0]
    logit = float(model.decision_function(sample_features)[0])

    if SHAP_AVAILABLE and shap_explainer is not None:
        try:
            sample_scaled = model.named_steps["scaler"].transform(sample_features)
            explanation = shap_explainer(sample_scaled)
            attributions = np.asarray(explanation.values).reshape(-1)
            base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
            return {
                "feature_names": np.asarray(FEATURE_NAMES),
                "attributions": attributions,
                "base_value": base_value,
                "probs": probs,
                "logit": logit,
                "method": "shap-logodds",
            }
        except Exception as exc:
            print(f"[SHAP] sample explain failed | {type(exc).__name__}: {exc}")

    feature_names, contributions, probs = compute_feature_contributions(sample_idx)
    if feature_names is None:
        return None

    intercept = float(model.named_steps["clf"].intercept_[0])
    return {
        "feature_names": feature_names,
        "attributions": contributions,
        "base_value": intercept,
        "probs": probs,
        "logit": logit,
        "method": "linear-fallback",
    }


def compute_decision_summary(sample_idx):
    """Compute logit, probability and grouped contributions for the decision panel."""
    attribution_data = compute_feature_attributions(sample_idx)
    if attribution_data is None:
        return None

    feature_names = attribution_data["feature_names"]
    attributions = attribution_data["attributions"]
    base_value = float(attribution_data["base_value"])
    probs = attribution_data["probs"]
    logit = float(attribution_data["logit"])
    pred_prob = float(probs[1])
    pred_class = int(np.argmax(probs))

    pos_pairs = sorted(
        [(feature_names[i], float(attributions[i])) for i in range(len(feature_names)) if attributions[i] > 0],
        key=lambda x: -x[1],
    )
    neg_pairs = sorted(
        [(feature_names[i], float(attributions[i])) for i in range(len(feature_names)) if attributions[i] < 0],
        key=lambda x: x[1],
    )
    return {
        "logit": logit,
        "base_value": base_value,
        "pred_prob": pred_prob,
        "pred_class": pred_class,
        "for_seizure": pos_pairs,
        "against_seizure": neg_pairs,
        "attributions": attributions,
        "feature_names": feature_names,
        "method": attribution_data["method"],
    }


def generate_feature_panel_content(sample_idx, selected_feature=None):
    """Return (decision_header, balance_bar, reflection_text) as Dash HTML children."""
    summary = compute_decision_summary(sample_idx)

    if summary is None:
        placeholder = html.Span(
            "Train the model to see decision insights.",
            style={"fontSize": "10px", "color": "#94a3b8"},
        )
        return placeholder, "", ""

    pred_label = "Seizure" if summary["pred_class"] == 1 else "Non-Seizure"
    pred_color = "#dc2626" if summary["pred_class"] == 1 else "#2563eb"
    bg_color = "#fef2f2" if summary["pred_class"] == 1 else "#eff6ff"
    border_color = "#fca5a5" if summary["pred_class"] == 1 else "#93c5fd"
    logit = summary["logit"]
    base_value = summary["base_value"]
    pred_prob = summary["pred_prob"]
    method_text = "SHAP" if summary.get("method") == "shap-logodds" else "Linear"

    # 1. Decision header
    decision_header = html.Div([
        html.Span("Model Decision  ", style={"fontWeight": "700", "fontSize": "10px", "color": "#374151"}),
        html.Span("Prediction: ", style={"fontSize": "10px", "color": "#6b7280"}),
        html.Span(pred_label, style={"fontWeight": "700", "fontSize": "11px", "color": pred_color}),
        html.Span(f"  P(seizure): {pred_prob:.2f}", style={"fontSize": "10px", "color": "#6b7280", "marginLeft": "6px"}),
        html.Span(f"  Base: {base_value:+.2f}", style={"fontSize": "10px", "color": "#6b7280", "marginLeft": "6px"}),
        html.Span(f"  Score: {logit:+.2f}", style={"fontSize": "10px", "color": "#6b7280", "marginLeft": "6px"}),
        html.Span(f"  Method: {method_text}", style={"fontSize": "10px", "color": "#6b7280", "marginLeft": "6px"}),
    ], style={
        "backgroundColor": bg_color,
        "border": f"1px solid {border_color}",
        "borderRadius": "6px",
        "padding": "4px 8px",
        "marginBottom": "3px",
    })

    # 2. Balance bar (0% = full Non-Seizure, 100% = full Seizure, midpoint=50%)
    clamped = max(-5.0, min(5.0, logit))
    pct = (clamped + 5.0) / 10.0 * 100

    balance_bar = html.Div([
        html.Div(style={
            "height": "10px",
            "background": "linear-gradient(to right, #bfdbfe 0%, #e5e7eb 50%, #fca5a5 100%)",
            "borderRadius": "5px",
            "position": "relative",
            "marginBottom": "1px",
        }, children=[
            html.Div("▲", style={
                "position": "absolute",
                "left": f"calc({pct:.0f}% - 5px)",
                "top": "-2px",
                "fontSize": "12px",
                "color": pred_color,
                "lineHeight": "1",
            }),
        ]),
        html.Div([
            html.Span("← Non-Seizure", style={"fontSize": "8px", "color": "#2563eb"}),
            html.Span(f"score {logit:+.2f}", style={"fontSize": "8px", "color": "#374151"}),
            html.Span("Seizure →", style={"fontSize": "8px", "color": "#dc2626"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "3px"}),
    ])

    # 3. Reflection text
    if selected_feature:
        explanation = get_clinical_feature_explanation(selected_feature)
        fnames = list(summary["feature_names"])
        fidx = fnames.index(selected_feature) if selected_feature in fnames else None
        contrib_text = ""
        if fidx is not None:
            v = float(summary["attributions"][fidx])
            direction = "toward seizure" if v >= 0 else "toward non-seizure"
            contrib_text = f"  Attribution: {v:+.4f} ({direction})."
        reflection = html.Div([
            html.Span("Inspecting: ", style={"fontWeight": "700", "fontSize": "10px", "color": "#374151"}),
            html.Span(selected_feature, style={"fontWeight": "600", "fontSize": "10px", "color": "#7c3aed"}),
            html.Span(contrib_text, style={"fontSize": "9px", "color": "#6b7280"}),
            html.Br(),
            html.Span(explanation, style={"fontSize": "9px", "color": "#374151", "lineHeight": "1.4"}),
        ], style={
            "backgroundColor": "#faf5ff",
            "border": "1px solid #d8b4fe",
            "borderRadius": "6px",
            "padding": "4px 8px",
            "marginTop": "2px",
        })
    else:
        top_for = summary["for_seizure"][:3]
        top_against = summary["against_seizure"][:3]
        lines = []
        if top_for:
            lines.append("↑ " + ", ".join(n for n, _ in top_for) + " push toward seizure.")
        if top_against:
            lines.append("↓ " + ", ".join(n for n, _ in top_against) + " push toward non-seizure.")
        lines.append(f"Overall: evidence {'favors Seizure' if logit > 0 else 'favors Non-Seizure'} (score {logit:+.2f}).")
        reflection = html.Div([
            html.Span("Model reasoning:  ", style={"fontWeight": "700", "fontSize": "10px", "color": "#374151"}),
            html.Span("  ".join(lines), style={"fontSize": "9px", "color": "#374151", "lineHeight": "1.5"}),
        ], style={
            "backgroundColor": "#f8fafc",
            "border": "1px solid #e2e8f0",
            "borderRadius": "6px",
            "padding": "4px 8px",
            "marginTop": "2px",
        })

    return decision_header, balance_bar, reflection


def get_clinical_feature_explanation(feature_name):
    """Return a concise clinical interpretation for the selected engineered feature."""
    explanation_map = {
        "Delta Power": "High delta power can reflect abnormal slow-wave activity often observed in ictal or peri-ictal states.",
        "Theta Power": "Elevated theta power may indicate abnormal rhythmic slowing associated with seizure propagation.",
        "Alpha Power": "Alpha suppression or abnormal alpha shifts can mark disrupted cortical rhythms during pathological events.",
        "Beta Power": "Beta changes can indicate altered fast activity; focal increases may align with epileptiform discharges.",
        "Peak-to-Peak": "Large peak-to-peak amplitude can correspond to epileptic spike-and-wave or sharp transient discharges.",
        "Kurtosis": "High kurtosis suggests sharp, heavy-tailed transients, a common signature of epileptiform spikes.",
        "Std Dev": "High standard deviation indicates unstable signal amplitude and increased neural variability.",
        "RMS": "High RMS reflects sustained signal energy, which can increase during abnormal synchronized activity.",
        "Skewness": "High skewness indicates waveform asymmetry, which may arise from non-sinusoidal epileptiform patterns.",
        "Mean": "Baseline drift in the local mean may reveal slow shifts in underlying cortical potential dynamics.",
    }
    return explanation_map.get(feature_name, "This feature captures a model-relevant EEG pattern for the current segment.")


# ==========================================================
# LOAD + SPLIT
# ==========================================================

def load_and_split():
    data_path = os.path.join(os.path.dirname(__file__), "..", "bonn_eeg_combined.csv")
    df = pd.read_csv(data_path)

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

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=GLOBAL_RANDOM_SEED)
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
            random_state=GLOBAL_RANDOM_SEED
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
# DATA + CONSTANTS
# ==========================================================

X_raw_train, X_train, X_test, y_train, y_test, subjects_train = load_and_split()

# Keep EEG annotation panel on a consistent amplitude scale across samples.
# Use robust percentile bounds to avoid extreme outliers dominating the axis.
GLOBAL_Y_MIN = float(np.percentile(X_raw_train, 1))
GLOBAL_Y_MAX = float(np.percentile(X_raw_train, 99))

initial_fraction = 0.1
batch_size = 10
default_confidence_threshold = 0.7

# ==========================================================
# EMBEDDING PRECOMPUTE
# ==========================================================

embedding_scaler = StandardScaler()
X_train_scaled = embedding_scaler.fit_transform(X_train)
model_umap = None

if UMAP_AVAILABLE:
    try:
        umap_model = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            init="spectral",
            metric="euclidean",
            random_state=GLOBAL_RANDOM_SEED,
        )
        model_umap = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            init="spectral",
            metric="euclidean",
            random_state=GLOBAL_RANDOM_SEED,
        )
        X_train_umap = umap_model.fit_transform(X_train_scaled)
        X_train_umap_data_dgrid = apply_dgrid_transform(X_train_umap, embedding_name=DGRID_KEY_UMAP_DATA)
        X_train_umap_model_dgrid = None
        dgrid_runtime_modes[DGRID_KEY_UMAP_MODEL] = "pending-train"
        print(f"[UMAP][DATA] OK | shape={X_train_umap.shape} | dgrid={dgrid_runtime_modes.get(DGRID_KEY_UMAP_DATA)}")
    except Exception as exc:
        X_train_umap = None
        X_train_umap_data_dgrid = None
        X_train_umap_model_dgrid = None
        model_umap = None
        dgrid_runtime_modes[DGRID_KEY_UMAP_DATA] = "failed"
        dgrid_runtime_modes[DGRID_KEY_UMAP_MODEL] = "pending-train"
        print(f"[UMAP][DATA] FAILED | {type(exc).__name__}: {exc}")
else:
    X_train_umap = None
    X_train_umap_data_dgrid = None
    X_train_umap_model_dgrid = None
    model_umap = None
    dgrid_runtime_modes[DGRID_KEY_UMAP_DATA] = "not-available"
    dgrid_runtime_modes[DGRID_KEY_UMAP_MODEL] = "not-available"
    print(f"[UMAP][DATA] UNAVAILABLE | {UMAP_IMPORT_ERROR}")

_print_dgrid_status("pre-init")


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

# Track selected feature for EEG highlighting
selected_feature_for_highlight = None

# NEW: Annotation queue system from gauri's version
annotation_queue = []
selected_sample_id = None

# Active learning state variables (must be initialized at module level)
current_pointer = 0
phase = "annotation"
round_number = 0
current_batch = np.array([])
batch_auto_count = 0
pool_auto_count = 0
current_confidence_threshold = default_confidence_threshold
annotations_this_round = 0
previous_predictions = None
prediction_history = []
test_prediction_history = []
sankey_fig_cache = None
feature_panel_cache = None
stable_rounds = 0
stop_active_learning = False

# Stop when prediction flips stay below this ratio and no doctor samples are needed.
stability_flip_ratio_threshold = 0.01
stability_required_rounds = 2


def compute_model_umap_embedding():
    """Compute a model-space UMAP from signed contributions and confidence terms."""
    global model_umap

    if not UMAP_AVAILABLE or model is None:
        if not UMAP_AVAILABLE:
            dgrid_runtime_modes[DGRID_KEY_UMAP_MODEL] = "not-available"
            print(f"[UMAP][MODEL] UNAVAILABLE | {UMAP_IMPORT_ERROR}")
        else:
            dgrid_runtime_modes[DGRID_KEY_UMAP_MODEL] = "pending-train"
            print("[UMAP][MODEL] SKIPPED | model not ready yet")
        return None

    try:
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        sample_scaled = scaler.transform(X_train)

        if hasattr(clf, "coef_"):
            weights = clf.coef_[0]
            intercept = float(clf.intercept_[0]) if hasattr(clf, "intercept_") else 0.0
            contributions = sample_scaled * weights
            logits = sample_scaled @ weights + intercept
            probs = model.predict_proba(X_train)[:, 1]
            model_repr = np.column_stack([contributions, logits, probs])
        else:
            model_repr = model.predict_proba(X_train)

        if model_umap is None:
            model_umap = UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                init="spectral",
                metric="euclidean",
                random_state=GLOBAL_RANDOM_SEED,
            )

        model_coords = model_umap.fit_transform(model_repr)
        transformed = apply_dgrid_transform(model_coords, embedding_name=DGRID_KEY_UMAP_MODEL)
        print(f"[UMAP][MODEL] OK | shape={model_coords.shape} | dgrid={dgrid_runtime_modes.get(DGRID_KEY_UMAP_MODEL)}")
        return transformed
    except Exception as exc:
        dgrid_runtime_modes[DGRID_KEY_UMAP_MODEL] = "failed"
        print(f"[UMAP][MODEL] FAILED | {type(exc).__name__}: {exc}")
        return None


def initialize_active_learning():
    global labeled_idx, unlabeled_idx
    global model, current_batch
    global batch_auto_count, pool_auto_count
    global current_pointer, round_number, phase
    global train_history, test_history, round_history
    global sensitivity_history, specificity_history
    global initial_labeled_idx, oracle_annotated_idx
    global current_confidence_threshold, X_train_umap_model_dgrid
    global annotation_queue, selected_sample_id
    global annotations_this_round
    global previous_predictions, prediction_history, test_prediction_history, sankey_fig_cache, feature_panel_cache, stable_rounds, stop_active_learning

    train_history = []
    test_history = []
    round_history = []
    sensitivity_history = []
    specificity_history = []

    n_initial = int(initial_fraction * len(X_train))
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED)
    indices = rng.permutation(len(X_train))

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

    initialize_shap_explainer()

    X_train_umap_model_dgrid = compute_model_umap_embedding()

    current_batch, batch_auto_count, pool_auto_count = compute_batch(current_confidence_threshold)
    annotation_queue = []
    current_pointer = 0
    selected_sample_id = None
    annotations_this_round = 0
    previous_predictions = predict_binary_with_threshold(X_train)
    prediction_history = [previous_predictions.copy()]
    test_predictions = predict_binary_with_threshold(X_test)
    test_prediction_history = [test_predictions.copy()]
    sankey_fig_cache = build_multiround_sankey(test_prediction_history, y_test)
    feature_panel_cache = None
    stable_rounds = 0
    stop_active_learning = False


def build_learning_curve():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=round_history,
        y=train_history,
        mode="lines+markers",
        name="Train Acc",
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=test_history,
        mode="lines+markers",
        name="Test Acc",
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=sensitivity_history,
        mode="lines+markers",
        name="Sensitivity",
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=specificity_history,
        mode="lines+markers",
        name="Specificity",
        line=dict(width=2),
    ))

    max_round = max(1, max(round_history) if len(round_history) > 0 else 1)
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Performance",
        yaxis=dict(range=[0, 1], tickfont=dict(size=10), automargin=True, title_standoff=8),
        xaxis=dict(range=[0, max_round], dtick=1, tickfont=dict(size=10), automargin=True, title_standoff=8),
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=90, t=24, b=52),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )

    return fig


def build_confusion_heatmap(cm):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred:<br>Non-Seizure", "Pred:<br>Seizure"],
        y=["Actual:<br>Non-Seizure", "Actual:<br>Seizure"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        textfont=dict(size=14),
    ))

    fig.update_layout(
        title=None,
        autosize=True,
        margin=dict(l=86, r=24, t=12, b=78),
        xaxis=dict(
            side="bottom",
            tickfont=dict(size=9),
            automargin=True,
            title_standoff=10,
        ),
        yaxis=dict(
            tickfont=dict(size=9),
            automargin=True,
            title_standoff=10,
        ),
        width=None,
        height=260,
    )
    return fig


def build_sankey_confusion(y_true, y_pred):
    """Build a Sankey diagram showing true-to-predicted class flow with raw counts."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    labels = [
        f"True: Non-Seizure ({tn + fp})",
        f"True: Seizure ({fn + tp})",
        f"Pred: Non-Seizure ({tn + fn})",
        f"Pred: Seizure ({fp + tp})",
    ]

    source = [0, 0, 1, 1]
    target = [2, 3, 2, 3]
    values = [tn, fp, fn, tp]
    colors = ["#2563eb", "#ef4444", "#f97316", "#22c55e"]

    # Hover text with exact confusion matrix counts.
    hover_text = [
        f"TN (True Negative)<br>Count: {tn}",
        f"FP (False Positive)<br>Count: {fp}",
        f"FN (False Negative)<br>Count: {fn}",
        f"TP (True Positive)<br>Count: {tp}",
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#1e40af", "#991b1b", "#1e40af", "#991b1b"],
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color=colors,
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            hoverlabel=dict(
                font=dict(color="#ffffff"),
                bgcolor="#111827",
            ),
        )
    )

    fig.update_layout(
        title=None,
        font=dict(size=11, color="#000000"),
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        autosize=True,
    )

    return fig


def build_multiround_sankey(pred_history, y_true):
    """Build Sankey showing TN/FP/FN/TP transitions across rounds using raw counts."""
    if pred_history is None or len(pred_history) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=None,
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            annotations=[
                dict(
                    text="Train more rounds to see prediction flow.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#ffffff"),
                )
            ],
        )
        return fig

    if len(pred_history) == 1:
        # Round 0 only: show the initially computed flow.
        return build_sankey_confusion(y_true, np.asarray(pred_history[0]))

    y_true = np.asarray(y_true)

    category_code = {
        (0, 0): "TN",
        (0, 1): "FP",
        (1, 0): "FN",
        (1, 1): "TP",
    }
    category_color = {
        (0, 0): "#2563eb",
        (0, 1): "#ef4444",
        (1, 0): "#f97316",
        (1, 1): "#22c55e",
    }

    round_category_counts = []
    for preds in pred_history:
        preds_arr = np.asarray(preds).astype(int)
        counts = {}
        for key in category_code:
            t_label, p_label = key
            counts[key] = int(np.sum((y_true == t_label) & (preds_arr == p_label)))
        round_category_counts.append(counts)

    labels = []
    node_keys = []
    node_map = {}
    source = []
    target = []
    values = []
    link_colors = []
    hover_text = []
    flow_index = {}
    flow_counts = []

    def get_node(round_idx, true_label, pred_label):
        key = (round_idx, true_label, pred_label)
        if key not in node_map:
            node_map[key] = len(labels)
            count = round_category_counts[round_idx][(true_label, pred_label)]
            labels.append(f"R{round_idx} {category_code[(true_label, pred_label)]} ({count})")
            node_keys.append(key)
        return node_map[key]

    for r in range(len(pred_history) - 1):
        prev = np.asarray(pred_history[r]).astype(int)
        curr = np.asarray(pred_history[r + 1]).astype(int)

        for i in range(len(y_true)):
            true_label = int(y_true[i])
            prev_pred = int(prev[i])
            curr_pred = int(curr[i])
            src = get_node(r, true_label, prev_pred)
            dst = get_node(r + 1, true_label, curr_pred)
            key = (src, dst)

            if key not in flow_index:
                flow_index[key] = len(source)
                source.append(src)
                target.append(dst)
                values.append(1.0)
                flow_counts.append(1)
            else:
                idx = flow_index[key]
                values[idx] += 1.0
                flow_counts[idx] += 1

    for idx, (src, dst) in enumerate(zip(source, target)):
        dst_key = node_keys[dst]
        dst_category = (dst_key[1], dst_key[2])
        count = flow_counts[idx]
        link_colors.append(category_color[dst_category])
        hover_text.append(
            f"<br>Count: {count}"
        )

    node_colors = [category_color[(k[1], k[2])] for k in node_keys]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                color=node_colors,
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color=link_colors,
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            hoverlabel=dict(
                font=dict(color="#ffffff"),
                bgcolor="#111827",
            ),
        )
    )

    fig.update_layout(
        title=None,
        font=dict(size=11, color="#000000"),
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        autosize=True,
    )

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
        textfont=dict(size=10),
    )])

    fig.update_layout(
        title=None,
        autosize=True,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=8),
        ),
    )

    return fig


def _build_embedding_boundary_trace(embedding_coords):
    """Build decision boundary contour for embedding."""
    if len(labeled_idx) < 10:
        return None

    x_min, x_max = float(embedding_coords[:, 0].min()), float(embedding_coords[:, 0].max())
    y_min, y_max = float(embedding_coords[:, 1].min()), float(embedding_coords[:, 1].max())
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 120),
        np.linspace(y_min - y_pad, y_max + y_pad, 120),
    )

    boundary_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=GLOBAL_RANDOM_SEED,
    )
    boundary_model.fit(embedding_coords, y_train)
    zz = boundary_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    return go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=zz,
        showscale=False,
        contours=dict(start=SEIZURE_PROB_THRESHOLD, end=SEIZURE_PROB_THRESHOLD, size=1, coloring="none", showlines=True),
        line=dict(color="black", width=2, dash="dash"),
        name="Decision Boundary",
        hoverinfo="skip",
        showlegend=True,
    )


def build_feature_importance(sample_idx, importance_mode="contribution", selected_features=None):
    """
    Build feature importance visualization for a specific sample.

    Args:
        sample_idx: Index of the sample to explain
        importance_mode: "contribution" or "uncertainty"
    """
    attribution_data = compute_feature_attributions(sample_idx)
    if attribution_data is None:
        # Model not trained yet
        return go.Figure().update_layout(
            title="Model not trained yet",
            template="plotly_white",
        )

    feature_names = attribution_data["feature_names"]
    attributions = attribution_data["attributions"]
    probs = attribution_data["probs"]
    base_value = float(attribution_data["base_value"])
    logit = float(attribution_data["logit"])
    method = attribution_data["method"]

    pred_prob = probs[1]  # Probability of seizure

    selected_set = set(selected_features or [])

    if importance_mode == "contribution":
        # Sort by contribution value descending: most FOR seizure at top, AGAINST at bottom
        sorted_indices = sorted(range(len(feature_names)), key=lambda i: attributions[i], reverse=True)
        bar_names = [feature_names[i] for i in sorted_indices]
        bar_vals = [float(attributions[i]) for i in sorted_indices]
        colors = []
        border_widths = []
        for i in sorted_indices:
            fname = feature_names[i]
            val = float(attributions[i])
            colors.append('#dc2626' if val > 0 else '#2563eb')
            border_widths.append(3 if fname in selected_set else 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=bar_names,
            x=bar_vals,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=border_widths, color="#111827"),
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))

        fig.update_layout(
            title=None,
            xaxis_title="← Non-Seizure  |  Seizure →",
            yaxis_title="",
            template="plotly_white",
            height=260,
            margin=dict(l=120, r=14, t=22, b=44),
            showlegend=False,
        )

        fig.update_xaxes(automargin=True, title_standoff=6, title_font=dict(size=9, color="#555"))
        fig.update_yaxes(automargin=True)

        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="#374151", line_width=1)

        # Group direction labels
        fig.add_annotation(
            text="AGAINST Seizure ◀",
            xref="paper", yref="paper", x=0.0, y=1.07,
            xanchor="left", showarrow=False,
            font=dict(size=8, color="#2563eb"),
        )
        fig.add_annotation(
            text="▶ FOR Seizure",
            xref="paper", yref="paper", x=1.0, y=1.07,
            xanchor="right", showarrow=False,
            font=dict(size=8, color="#dc2626"),
        )

        method_label = "SHAP log-odds" if method == "shap-logodds" else "Linear fallback"
        fig.add_annotation(
            text=f"{method_label} | Base={base_value:+.3f} | Logit={logit:+.3f} | P(seizure)={pred_prob:.3f}",
            xref="paper", yref="paper", x=0.5, y=-0.16,
            showarrow=False,
            font=dict(size=8, color="#475569"),
        )


    else:  # uncertainty mode
        # Uncertainty Explanation View
        positive_mask = attributions > 0
        negative_mask = attributions < 0

        pos_contributions = attributions[positive_mask]
        neg_contributions = attributions[negative_mask]
        pos_features = [feature_names[i] for i in range(len(feature_names)) if positive_mask[i]]
        neg_features = [feature_names[i] for i in range(len(feature_names)) if negative_mask[i]]

        # Calculate uncertainty
        uncertainty = 1 - np.max(probs)

        fig = go.Figure()

        # Positive evidence (pushes toward seizure)
        if len(pos_contributions) > 0:
            sorted_pos_idx = np.argsort(pos_contributions)[::-1]
            fig.add_trace(go.Bar(
                y=[pos_features[i] for i in sorted_pos_idx],
                x=[pos_contributions[i] for i in sorted_pos_idx],
                orientation='h',
                name="Evidence FOR Seizure",
                marker=dict(color='#e74c3c'),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))

        # Negative evidence (pushes toward non-seizure)
        if len(neg_contributions) > 0:
            sorted_neg_idx = np.argsort(np.abs(neg_contributions))[::-1]
            fig.add_trace(go.Bar(
                y=[neg_features[i] for i in sorted_neg_idx],
                x=[neg_contributions[i] for i in sorted_neg_idx],
                orientation='h',
                name="Evidence AGAINST Seizure",
                marker=dict(color='#3498db'),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))

        fig.update_layout(
            title=None,
            xaxis_title="Contribution",
            yaxis_title="",
            template="plotly_white",
            height=260,
            margin=dict(l=120, r=14, t=14, b=44),
            barmode='relative',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                font=dict(size=8),
            ),
        )

        fig.update_xaxes(automargin=True, title_standoff=8)
        fig.update_yaxes(automargin=True)

        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)

        fig.add_annotation(
            text=f"Unc={uncertainty:.3f} | P(seizure)={pred_prob:.3f}",
            xref="paper", yref="paper",
            x=0.5, y=-0.14,
            showarrow=False,
            font=dict(size=8, color="#666"),
        )

    return fig


def build_embedding_figure(pca_view_mode, embedding_space_mode, confidence_threshold, sample_filter):
    """Build embedding figure with DGrid transformation and sample filter"""
    if not UMAP_AVAILABLE or X_train_umap_data_dgrid is None:
        if UMAP_IMPORT_ERROR:
            umap_msg = f"UMAP unavailable: {UMAP_IMPORT_ERROR}"
        else:
            umap_msg = "UMAP not available. Install umap-learn to enable embedding."
        embedding_fig = go.Figure()
        embedding_fig.update_layout(
            template="plotly_white",
            margin=dict(t=20, b=30, l=30, r=20),
            annotations=[
                dict(
                    text=umap_msg,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=13, color="#64748b"),
                )
            ],
        )
        return embedding_fig, "UMAP"

    if embedding_space_mode == "model" and X_train_umap_model_dgrid is not None:
        embedding_coords = X_train_umap_model_dgrid
        embedding_type = "UMAP Model"
        coord_labels = {"x": "Model UMAP 1", "y": "Model UMAP 2"}
    else:
        embedding_coords = X_train_umap_data_dgrid
        embedding_type = "UMAP Data"
        coord_labels = {"x": "Data UMAP 1", "y": "Data UMAP 2"}

    sample_status = np.array(["Unlabeled"] * len(X_train))
    sample_status[labeled_idx] = "Labeled"

    all_uncertainties = np.zeros(len(X_train))
    all_predictions = np.zeros(len(X_train))
    all_probs_class1 = np.zeros(len(X_train))

    if len(unlabeled_idx) > 0:
        unlabeled_probs = model.predict_proba(X_train[unlabeled_idx])
        all_uncertainties[unlabeled_idx] = 1 - np.max(unlabeled_probs, axis=1)
        all_predictions[unlabeled_idx] = predict_binary_with_threshold(X_train[unlabeled_idx])
        all_probs_class1[unlabeled_idx] = unlabeled_probs[:, 1]

    if len(labeled_idx) > 0:
        labeled_probs = model.predict_proba(X_train[labeled_idx])
        all_uncertainties[labeled_idx] = 1 - np.max(labeled_probs, axis=1)
        all_predictions[labeled_idx] = predict_binary_with_threshold(X_train[labeled_idx])
        all_probs_class1[labeled_idx] = labeled_probs[:, 1]

    all_probs_class0 = 1 - all_probs_class1
    threshold_flag = np.where((1 - all_uncertainties) >= confidence_threshold, "Confident", "Uncertain")

    recently_queried = np.zeros(len(X_train), dtype=bool)
    if len(oracle_annotated_idx) > 0:
        recently_queried[oracle_annotated_idx] = True

    # NEW: Track samples in annotation queue
    in_queue = np.zeros(len(X_train), dtype=bool)
    if len(annotation_queue) > 0:
        in_queue[np.array(annotation_queue, dtype=int)] = True

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
        "In_Queue": ["Yes" if q else "No" for q in in_queue],
        "Threshold_Status": threshold_flag,
    })

    # Confidence is the probability of the predicted class; Uncertainty is 1 - Confidence.
    embedding_df["Confidence"] = np.maximum(
        embedding_df["Prob_Seizure"],
        embedding_df["Prob_Non_Seizure"],
    )
    embedding_df["Uncertainty"] = 1.0 - embedding_df["Confidence"]

    embedding_df["Displayed_Prob"] = np.where(
        embedding_df["Predicted_Label"] == "Seizure",
        embedding_df["Prob_Seizure"],
        embedding_df["Prob_Non_Seizure"],
    )
    embedding_df["Displayed_Label"] = np.where(
        embedding_df["Predicted_Label"] == "Seizure",
        "P(Seizure)",
        "P(Non-Seizure)",
    )

    embedding_df["Threshold_Display"] = embedding_df["Threshold_Status"]

    embedding_df["Correct"] = np.where(
        embedding_df["True_Label"] == embedding_df["Predicted_Label"],
        "Yes",
        "No",
    )

    # Apply sample filter from gauri's version
    plot_df = embedding_df
    if sample_filter == "labeled":
        plot_df = embedding_df[embedding_df["Status"] == "Labeled"].copy()
    elif sample_filter == "unlabeled":
        plot_df = embedding_df[embedding_df["Status"] == "Unlabeled"].copy()

    if plot_df.empty:
        embedding_fig = go.Figure()
    elif pca_view_mode == "label_status":
        embedding_fig = px.scatter(
            plot_df,
            x="Dim1",
            y="Dim2",
            color="Status",
            custom_data=[
                "Sample_ID",
                "Predicted_Label",
                "Status",
                "Threshold_Display",
                "Displayed_Label",
                "Displayed_Prob",
            ],
            color_discrete_map={"Labeled": "#27ae60", "Unlabeled": "#95a5a6"},
            hover_data={
                "Predicted_Label": True,
                "Status": True,
                "Threshold_Display": True,
                "Displayed_Prob": ":.2f",
                # "True_Label": True,
                # "Uncertainty": ":.2f",
                # "Prob_Seizure": ":.2f",
                # "Prob_Non_Seizure": ":.2f",
                # "Sample_ID": True,
                # "Queried": True,
                # "In_Queue": True,
                # "Threshold_Status": True,
            },
            title=None,
            labels={
                "Dim1": coord_labels["x"],
                "Dim2": coord_labels["y"],
                "Threshold_Display": "Threshold Status",
                "Displayed_Prob": "Probability",
            },
        )
        embedding_fig.update_traces(
            hovertemplate=(
                "Predicted Label: %{customdata[1]}<br>"
                "Status: %{customdata[2]}<br>"
                "Threshold Status: %{customdata[3]}<br>"
                "%{customdata[4]}: %{customdata[5]:.2f}<extra></extra>"
            )
        )
    elif pca_view_mode == "true_class":
        embedding_fig = px.scatter(
            plot_df,
            x="Dim1",
            y="Dim2",
            color="True_Label",
            custom_data=[
                "Sample_ID",
                "Predicted_Label",
                "True_Label",
                "Status",
                "Threshold_Display",
                "Displayed_Label",
                "Displayed_Prob",
            ],
            color_discrete_map={"Seizure": "#e74c3c", "Non-Seizure": "#3498db"},
            hover_data={
                "Predicted_Label": True,
                "True_Label": True,
                "Status": True,
                "Threshold_Display": True,
                "Displayed_Prob": ":.2f",
                # "Correct": True,
                # "Uncertainty": ":.2f",
                # "Prob_Seizure": ":.2f",
                # "Prob_Non_Seizure": ":.2f",
                # "Sample_ID": True,
                # "Queried": True,
                # "In_Queue": True,
                # "Threshold_Status": True,
            },
            title=None,
            labels={
                "Dim1": coord_labels["x"],
                "Dim2": coord_labels["y"],
                "Threshold_Display": "Threshold Status",
                "Displayed_Prob": "Probability",
            },
        )
        embedding_fig.update_traces(
            hovertemplate=(
                "Predicted Label: %{customdata[1]}<br>"
                "True Label: %{customdata[2]}<br>"
                "Status: %{customdata[3]}<br>"
                "Threshold Status: %{customdata[4]}<br>"
                "%{customdata[5]}: %{customdata[6]:.2f}<extra></extra>"
            )
        )
    else:
        embedding_fig = px.scatter(
            plot_df,
            x="Dim1",
            y="Dim2",
            color="Uncertainty",
            custom_data=[
                "Sample_ID",
                "Predicted_Label",
                "Status",
                "Threshold_Display",
                "Displayed_Label",
                "Displayed_Prob",
                # "Uncertainty",
            ],
            color_continuous_scale="Reds",
            hover_data={
                "Predicted_Label": True,
                "Status": True,
                "Threshold_Display": True,
                "Displayed_Prob": ":.2f",
                # "True_Label": True,
                # "Uncertainty": ":.2f",
                # "Prob_Seizure": ":.2f",
                # "Prob_Non_Seizure": ":.2f",
                # "Sample_ID": True,
                # "Queried": True,
                # "In_Queue": True,
                # "Threshold_Status": True,
            },
            title=None,
            labels={
                "Dim1": coord_labels["x"],
                "Dim2": coord_labels["y"],
                "Threshold_Display": "Threshold Status",
                "Displayed_Prob": "Probability",
            },
        )
        embedding_fig.update_traces(
            hovertemplate=(
                "Predicted Label: %{customdata[1]}<br>"
                "Status: %{customdata[2]}<br>"
                "Threshold Status: %{customdata[3]}<br>"
                "%{customdata[4]}: %{customdata[5]:.2f}<extra></extra>"
            )
        )

    if not plot_df.empty:
        embedding_fig.update_traces(
            marker=dict(opacity=0.7, line=dict(width=0.5, color="black")),
            selector=dict(type="scatter"),
        )

        # Add bold border for oracle annotated samples
        queried_df = plot_df[plot_df["Queried"].str.startswith("Yes")]
        if not queried_df.empty:
            embedding_fig.add_trace(
                go.Scatter(
                    x=queried_df["Dim1"],
                    y=queried_df["Dim2"],
                    mode="markers",
                    marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(width=2.5, color="black")),
                    name="Oracle Annotated",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    # Add highlight for currently selected sample (yellow with black border from gauri)
    if selected_sample_id is not None:
        selected_mask = plot_df["Sample_ID"] == selected_sample_id
        if selected_mask.any():
            embedding_fig.add_trace(
                go.Scatter(
                    x=plot_df[selected_mask]["Dim1"],
                    y=plot_df[selected_mask]["Dim2"],
                    mode="markers",
                    marker=dict(size=16, color="#fde047", line=dict(width=3, color="black")),
                    name="Selected EEG",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    # Add decision boundary
    boundary_trace = _build_embedding_boundary_trace(embedding_coords)
    if boundary_trace is not None:
        embedding_fig.add_trace(boundary_trace)
        # Move boundary to back
        embedding_fig.data = (embedding_fig.data[-1],) + embedding_fig.data[:-1]

    # Set axis ranges with padding (from gauri's version)
    x_min = float(plot_df["Dim1"].min()) if not plot_df.empty else float(embedding_df["Dim1"].min())
    x_max = float(plot_df["Dim1"].max()) if not plot_df.empty else float(embedding_df["Dim1"].max())
    y_min = float(plot_df["Dim2"].min()) if not plot_df.empty else float(embedding_df["Dim2"].min())
    y_max = float(plot_df["Dim2"].max()) if not plot_df.empty else float(embedding_df["Dim2"].max())
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

    if plot_df.empty:
        embedding_fig.update_layout(
            annotations=[
                dict(
                    text="No samples for selected filter",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="#64748b"),
                )
            ]
        )

    return embedding_fig, embedding_type


def build_uncertainty_histogram(confidence_threshold):
    all_train_probs = model.predict_proba(X_train)
    all_train_uncertainty = 1 - np.max(all_train_probs, axis=1)
    all_train_confidence = np.max(all_train_probs, axis=1)

    # Calculate counts for percentage calculation
    total_samples = len(X_train)
    initial_count = len(initial_labeled_idx)
    oracle_count = len(oracle_annotated_idx)

    # IMPORTANT: Use current_batch (not all uncertain samples) to match donut chart
    doctor_count = len(current_batch)
    auto_batch_count = batch_auto_count
    auto_pool_count = pool_auto_count

    hist_fig = go.Figure()

    if initial_count > 0:
        initial_pct = (initial_count / total_samples) * 100
        hist_fig.add_trace(go.Histogram(
            x=all_train_uncertainty[initial_labeled_idx],
            name=f"Init ({initial_count})",
            marker=dict(color="#52c41a"),
            opacity=0.7,
            nbinsx=30,
        ))

    if oracle_count > 0:
        oracle_pct = (oracle_count / total_samples) * 100
        hist_fig.add_trace(go.Histogram(
            x=all_train_uncertainty[oracle_annotated_idx],
            name=f"Oracle ({oracle_count})",
            marker=dict(color="#237804"),
            opacity=0.7,
            nbinsx=30,
        ))

    # Show Auto (Pool) - confident samples outside top-10 batch
    if auto_pool_count > 0:
        # Get all unlabeled samples
        unlabeled_confidence = all_train_confidence[unlabeled_idx]
        unlabeled_uncertainty = all_train_uncertainty[unlabeled_idx]

        # Find confident samples (outside the current batch)
        confident_mask = unlabeled_confidence >= confidence_threshold
        batch_mask = np.isin(unlabeled_idx, current_batch)
        pool_mask = confident_mask & ~batch_mask

        pool_pct = (auto_pool_count / total_samples) * 100
        hist_fig.add_trace(go.Histogram(
            x=unlabeled_uncertainty[pool_mask],
            name=f"Pool ({auto_pool_count})",
            marker=dict(color="#3498db"),
            opacity=0.7,
            nbinsx=30,
        ))

    # Show Auto (Batch) - confident samples IN top-10 batch
    if auto_batch_count > 0:
        unlabeled_confidence = all_train_confidence[unlabeled_idx]
        unlabeled_uncertainty = all_train_uncertainty[unlabeled_idx]

        # Get top-10 uncertain indices
        uncertainty_full = 1 - unlabeled_confidence
        sorted_idx = np.argsort(uncertainty_full)[::-1]
        top_k = sorted_idx[:batch_size]
        top_k_samples = unlabeled_idx[top_k]

        # Find which ones are confident (not in current_batch)
        confident_in_batch = np.setdiff1d(top_k_samples, current_batch)
        confident_in_batch_mask = np.isin(unlabeled_idx, confident_in_batch)

        batch_pct = (auto_batch_count / total_samples) * 100
        if np.sum(confident_in_batch_mask) > 0:
            hist_fig.add_trace(go.Histogram(
                x=unlabeled_uncertainty[confident_in_batch_mask],
                name=f"Batch ({auto_batch_count})",
                marker=dict(color="#1d4ed8"),
                opacity=0.7,
                nbinsx=30,
            ))

    # Show Needs Doctor - samples in current_batch (matching donut chart)
    if doctor_count > 0:
        doctor_pct = (doctor_count / total_samples) * 100
        hist_fig.add_trace(go.Histogram(
            x=all_train_uncertainty[current_batch],
            name=f"Doctor ({doctor_count})",
            marker=dict(color="#e74c3c"),
            opacity=0.7,
            nbinsx=30,
        ))

    threshold_uncertainty = 1 - confidence_threshold
    hist_fig.add_vline(
        x=threshold_uncertainty,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text=f"thr={threshold_uncertainty:.2f}",
        annotation_position="top left",
    )

    hist_fig.update_layout(
        title=None,
        title_font=dict(size=14),
        xaxis_title="Uncertainty (1 - max_confidence)",
        yaxis_title="Count (log)",
        barmode="stack",
        bargap=0.1,
        template="plotly_white",
        margin=dict(t=24, b=102, l=50, r=16),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=8),
            entrywidth=62,
            entrywidthmode="pixels",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        xaxis=dict(tickfont=dict(size=9), automargin=True, title_standoff=6),
        yaxis=dict(
            type="log",
            tickfont=dict(size=9),
            dtick=1,
            automargin=True,
            title_standoff=6,
        ),
    )

    return hist_fig


# Initialize first time
initialize_active_learning()
_print_dgrid_status("post-init")

# ==========================================================
# DASH APP
# ==========================================================

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, viewport-fit=cover",
        }
    ],
)

# Add responsive CSS via external stylesheet
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
            }

            #react-entry-point {
                height: 100%;
            }

            /* Prevent text overflow globally */
            * {
                box-sizing: border-box;
            }

            /* Ensure text wraps and doesn't overflow */
            h1, h2, h3, h4, h5, h6, p, span, div, label {
                word-wrap: break-word;
                overflow-wrap: break-word;
                hyphens: auto;
            }

            /* Base responsive styles */
            @media (max-width: 1400px) {
                .workspace-grid { grid-template-columns: 1fr !important; }
                .left-column, .right-column { width: 100% !important; min-width: 0 !important; max-width: none !important; margin-left: 0 !important; }
                .right-row1 { grid-template-columns: 1fr 1fr !important; }
                .viz-panel { width: 100% !important; min-width: 0 !important; margin-left: 0 !important; margin-bottom: 10px; }
                .kpi-grid { grid-template-columns: repeat(3, 1fr) !important; gap: 4px !important; }
                .button-group { flex-wrap: wrap; gap: 4px; }
                .button-group button { margin-left: 0 !important; margin-top: 4px; font-size: 11px !important; padding: 6px 10px !important; }
            }

            @media (max-width: 1024px) {
                .right-row1 { grid-template-columns: 1fr !important; }
                .top-bar { flex-direction: column !important; align-items: flex-start !important; gap: 8px; }
                .kpi-grid { grid-template-columns: repeat(2, 1fr) !important; gap: 3px !important; font-size: 11px !important; }
                .kpi-grid span { font-size: 11px !important; }
                .control-row { flex-direction: column !important; gap: 4px; }
                .viz-panel h3 { font-size: 13px !important; }
                .radio-controls label { font-size: 10px !important; }
                .workspace-grid { overflow-y: auto !important; }
            }

            @media (max-width: 768px) {
                .kpi-grid { grid-template-columns: 1fr !important; gap: 2px !important; padding: 4px !important; }
                .top-bar h2 { font-size: 16px !important; }
                .button-group { width: 100%; }
                .button-group button { width: 100%; margin-left: 0 !important; }
                .radio-controls { flex-direction: column !important; align-items: flex-start !important; gap: 2px !important; }
                .radio-controls label { margin-left: 0 !important; margin-top: 4px; }
            }

            /* Make graphs responsive */
            .js-plotly-plot { min-height: 250px; }

            /* Keep Plotly controls from colliding with long titles */
            .js-plotly-plot .modebar {
                top: 4px !important;
                right: 4px !important;
                transform: scale(0.88);
                transform-origin: top right;
            }

            @media (max-width: 768px) {
                .js-plotly-plot .modebar {
                    transform: scale(0.78);
                }
            }

            /* Fix legend text overflow */
            .legend text { font-size: 10px !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Location(id="url", refresh=True),
    dcc.Store(id="perturbation-mode-store", data=False),  # Track perturbation mode
    dcc.Store(id="current-sample-store", data=0),  # Track current sample index
    dcc.Store(id="selected-features-store", data=[]),  # Multi-select feature state

    # Top bar with NEW buttons from gauri's version
    html.Div([
        html.Div([
            html.H2(
                "NeuroLens Clinical Assistant",
                style={"margin": "0", "fontSize": "clamp(14px, 2.5vw, 20px)", "color": "#0f172a",
                       "letterSpacing": "0.2px", "overflow": "hidden",
                       "textOverflow": "ellipsis"},
            ),
            html.Div(id="status-message",
                     style={"fontWeight": "600", "color": "#334155", "fontSize": "clamp(10px, 1.8vw, 12px)",
                            "marginTop": "2px", "overflow": "hidden",
                            "textOverflow": "ellipsis"}),
        ], style={"flex": "1", "minWidth": "150px", "maxWidth": "100%", "overflow": "hidden"}),
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
                    "fontSize": "clamp(10px, 1.2vw, 13px)",
                    "whiteSpace": "nowrap",
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
                    "fontSize": "clamp(10px, 1.2vw, 13px)",
                    "whiteSpace": "nowrap",
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
                    "fontSize": "clamp(10px, 1.2vw, 13px)",
                    "whiteSpace": "nowrap",
                },
            ),
            # NEW: Additional buttons from gauri's version
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
                    "fontSize": "clamp(10px, 1.2vw, 13px)",
                    "whiteSpace": "nowrap",
                },
            ),
            html.Button(
                "Clear Features",
                id="clear-features-btn",
                style={
                    "marginLeft": "8px",
                    "backgroundColor": "#f59e0b",
                    "color": "#111827",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "fontWeight": "600",
                    "cursor": "pointer",
                    "fontSize": "clamp(10px, 1.2vw, 13px)",
                    "whiteSpace": "nowrap",
                },
            ),
        ], className="button-group", style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}),
    ], className="top-bar", style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "8px 12px",
        "backgroundColor": "#f8fafc",
        "border": "1px solid #e2e8f0",
        "borderRadius": "10px",
        "marginBottom": "6px",
        "flexWrap": "wrap",
        "gap": "8px",
    }),

    # KPI + universal confidence control
    html.Div([
        html.Div([
            html.Span("R", style={"fontWeight": "700", "color": "#334155", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
            html.Span(id="round-display",
                      style={"marginLeft": "4px", "color": "#0f172a", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
        ], style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
        html.Div([
            html.Span("Labeled",
                      style={"fontWeight": "700", "color": "#334155", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
            html.Span(id="labeled-count",
                      style={"marginLeft": "4px", "color": "#0f172a", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
        ], style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
        html.Div([
            html.Span("Acc", style={"fontWeight": "700", "color": "#334155", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
            html.Span(id="test-accuracy",
                      style={"marginLeft": "4px", "color": "#0f172a", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
        ], style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
        html.Div([
            html.Span("Conf", style={"fontWeight": "700", "color": "#334155", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
            html.Span(id="confidence-value",
                      style={"marginLeft": "4px", "color": "#0f172a", "fontSize": "clamp(10px, 1.3vw, 12px)"}),
        ], style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
        # NEW: Queue status from gauri's version
        html.Div(id="queue-status",
                 style={"fontWeight": "600", "color": "#7c3aed", "fontSize": "clamp(10px, 1.3vw, 12px)",
                        "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
    ], className="kpi-grid", style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(100px, 1fr))",
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

    # Main workspace - left: embedding + annotation; right: (uncertainty + confusion), learning, feature explanation
    html.Div([
        html.Div([
            html.Div([
                html.H3(id="embedding-title", children="UMAP Embedding View",
                        style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "clamp(11px, 1.6vw, 13px)"}),
                dcc.Graph(id="pca-embedding", style={"flex": "1 1 auto", "minHeight": "200px"},
                      config={'responsive': True, 'displayModeBar': 'hover'}),
                html.Div([
                    dcc.RadioItems(
                        id="embedding-space-mode",
                        options=[
                            {"label": " Model UMAP", "value": "model"},
                        ],
                        value="model",
                        inline=True,
                        style={"fontSize": "9px", "marginRight": "8px"},
                    ),
                    dcc.RadioItems(
                        id="pca-view-mode",
                        options=[
                            {"label": " Status", "value": "label_status"},
                            {"label": " Class", "value": "true_class"},
                            {"label": " Uncertain", "value": "uncertainty"},
                        ],
                        value="label_status",
                        inline=True,
                        style={"fontSize": "9px", "marginRight": "8px"},
                    ),
                    dcc.RadioItems(
                        id="sample-filter",
                        options=[
                            {"label": " All", "value": "all"},
                            {"label": " Lab", "value": "labeled"},
                            {"label": " Unlab", "value": "unlabeled"},
                        ],
                        value="all",
                        inline=True,
                        style={"fontSize": "9px"},
                    ),
                ], className="radio-controls control-row", style={"display": "flex", "gap": "4px", "flexWrap": "wrap", "paddingTop": "2px"}),
            ], style={
                "flex": "1 1 0",
                "minWidth": "0",
                "minHeight": "0",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "8px",
                "padding": "5px",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
            }, className="viz-panel"),

            html.Div([
                html.H3("Annotation Panel", style={"margin": "0 0 3px 0", "color": "#0f172a", "fontSize": "clamp(11px, 1.5vw, 12px)"}),
                dcc.Graph(id="eeg-graph", style={"flex": "1 1 auto", "minHeight": "200px"}, config={'responsive': True, 'displayModeBar': 'hover'}),
                html.Div(
                    id="clinical-explanation",
                    style={
                        "marginTop": "4px",
                        "fontSize": "10px",
                        "color": "#334155",
                        "backgroundColor": "#f8fafc",
                        "border": "1px solid #e2e8f0",
                        "borderRadius": "6px",
                        "padding": "6px",
                        "lineHeight": "1.35",
                        "overflow": "auto",
                        "maxHeight": "60px",
                    },
                ),
            ], style={
                "flex": "1 1 0",
                "minWidth": "0",
                "minHeight": "0",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "8px",
                "padding": "4px",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
            }, className="viz-panel"),
        ], className="left-column", style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
            "minWidth": "0",
            "minHeight": "0",
            "height": "100%",
        }),

        html.Div([
            html.Div([
                html.Div([
                    html.H3("Uncertainty", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "clamp(11px, 1.6vw, 13px)"}),
                    dcc.Graph(id="uncertainty-histogram", style={"flex": "1 1 auto", "minHeight": "150px"}, config={'responsive': True}),
                ], style={
                    "minWidth": "0",
                    "minHeight": "0",
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #e2e8f0",
                    "borderRadius": "8px",
                    "padding": "5px",
                    "display": "flex",
                    "flexDirection": "column",
                    "overflow": "hidden",
                }, className="viz-panel"),

                html.Div([
                    html.H3("Learning", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "clamp(11px, 1.6vw, 13px)"}),
                    dcc.Graph(id="learning-curve", style={"flex": "1 1 auto", "minHeight": "150px"}, config={'responsive': True}),
                ], style={
                    "minWidth": "0",
                    "minHeight": "0",
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #e2e8f0",
                    "borderRadius": "8px",
                    "padding": "5px",
                    "display": "flex",
                    "flexDirection": "column",
                    "overflow": "hidden",
                }, className="viz-panel"),
            ], className="right-row1", style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1.2fr",
                "gap": "6px",
                "alignItems": "stretch",
                "minHeight": "0",
                "flex": "0 0 auto",
                "height": "45%",
            }),

            html.Div([
                html.H3("Prediction Flow Across Rounds", style={"margin": "0 0 3px 0", "color": "#0f172a", "fontSize": "clamp(11px, 1.5vw, 12px)"}),
                dcc.Graph(id="confusion-heatmap", style={"flex": "1 1 auto", "minHeight": "150px"}, config={'responsive': True}),
            ], style={
                "flex": "1 1 auto",
                "minWidth": "0",
                "minHeight": "0",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "8px",
                "padding": "4px",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
            }, className="viz-panel"),

            html.Div([
                html.H3("Feature Importance", style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "clamp(11px, 1.6vw, 13px)"}),
                html.Div(id="feature-decision-header"),
                html.Div(id="feature-balance-bar"),
                dcc.Graph(
                    id="feature-importance",
                    style={"flex": "1 1 auto", "minHeight": "150px"},
                    config={
                        'responsive': True,
                        'displayModeBar': 'hover',
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    },
                ),
                html.Div(id="feature-reflection-text"),
            ], style={
                "flex": "1 1 auto",
                "minWidth": "0",
                "minHeight": "0",
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "8px",
                "padding": "5px",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
            }, className="viz-panel"),
        ], className="right-column", style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            "minWidth": "0",
            "minHeight": "0",
            "height": "100%",
            "overflow": "hidden",
        }),
    ], className="main-workspace workspace-grid", style={
        "display": "grid",
        "gridTemplateColumns": "1.2fr 1fr",
        "gap": "4px",
        "flex": "1 1 auto",
        "minHeight": "0",
        "height": "100%",
        "overflow": "hidden",
        "alignItems": "stretch",
    }),

    # Hidden perturbation mount to keep callback IDs available without showing the panel.
    html.Div([
        html.Button("Enable Perturbation", id="perturbation-toggle"),
        html.Div(id="perturbation-controls", style={"display": "none"}, children=[
            dcc.Slider(id="slider-delta", min=-3, max=3, step=0.1, value=0),
            dcc.Slider(id="slider-theta", min=-3, max=3, step=0.1, value=0),
            dcc.Slider(id="slider-beta", min=-3, max=3, step=0.1, value=0),
            dcc.Slider(id="slider-kurtosis", min=-3, max=3, step=0.1, value=0),
            html.Div(id="perturbed-prediction"),
            html.Button("Reset", id="reset-perturbation"),
        ]),
        html.Div(id="perturbation-placeholder"),
    ], style={"display": "none"}),
], style={
    "display": "flex",
    "flexDirection": "column",
    "height": "100dvh",
    "minHeight": "100vh",
    "maxHeight": "none",
    "padding": "6px",
    "boxSizing": "border-box",
    "fontFamily": "'Segoe UI', 'Helvetica Neue', sans-serif",
    "backgroundColor": "#f1f5f9",
    "overflow": "auto",
}, className="app-shell")


# ==========================================================
# MAIN CALLBACK - Enhanced with embedding click interaction
# ==========================================================

@app.callback(
    Output("selected-features-store", "data"),
    Input("feature-importance", "clickData"),
    Input("clear-features-btn", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    State("selected-features-store", "data"),
    prevent_initial_call=True,
)
def toggle_feature_selection(click_data, clear_clicks, reset_clicks, selected_features):
    """Toggle feature selection on bar click and clear via button/reset."""
    selected_features = list(selected_features or [])

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger_id in {"clear-features-btn", "reset-btn"}:
        return []

    if trigger_id == "feature-importance" and click_data and "points" in click_data:
        feature = click_data["points"][0].get("y")
        if feature in selected_features:
            selected_features.remove(feature)
        elif feature in FEATURE_NAMES:
            selected_features.append(feature)

    return selected_features

@app.callback(
    Output("eeg-graph", "figure"),
    Output("annotate-btn", "disabled"),
    Output("train-btn", "disabled"),
    Output("confidence-slider", "disabled"),
    Output("confidence-value", "children"),
    Output("status-message", "children"),
    Output("learning-curve", "figure"),
    Output("confusion-heatmap", "figure"),
    Output("pca-embedding", "figure"),
    Output("uncertainty-histogram", "figure"),
    Output("feature-importance", "figure"),
    Output("round-display", "children"),
    Output("labeled-count", "children"),
    Output("test-accuracy", "children"),
    Output("embedding-title", "children"),
    Output("queue-status", "children"),
    Output("current-sample-store", "data"),
    Output("clinical-explanation", "children"),
    Output("feature-decision-header", "children"),
    Output("feature-balance-bar", "children"),
    Output("feature-reflection-text", "children"),
    Input("url", "pathname"),
    Input("annotate-btn", "n_clicks"),
    Input("train-btn", "n_clicks"),
    Input("confidence-slider", "value"),
    Input("reset-btn", "n_clicks"),
    Input("embedding-space-mode", "value"),
    Input("pca-view-mode", "value"),
    Input("pca-embedding", "clickData"),  # Capture embedding clicks
    Input("load-uncertain-btn", "n_clicks"),  # NEW: Load top-K button
    Input("sample-filter", "value"),  # NEW: Sample filter
    Input("selected-features-store", "data"),
    prevent_initial_call=False,
)
def update_dashboard(
        pathname,
        annotate_clicks,
        train_clicks,
        confidence_value,
        reset_clicks,
        embedding_space_mode,
        pca_view_mode,
        embedding_click_data,  # NEW
        load_uncertain_clicks,  # NEW
        sample_filter,  # NEW
        selected_features_store,
):
    global labeled_idx, unlabeled_idx
    global current_batch, current_pointer
    global batch_auto_count, pool_auto_count
    global model, phase, round_number
    global X_train_umap_model_dgrid
    global current_confidence_threshold, oracle_annotated_idx
    global annotation_queue, selected_sample_id  # NEW
    global annotations_this_round
    global previous_predictions, prediction_history, test_prediction_history, sankey_fig_cache, feature_panel_cache, stable_rounds, stop_active_learning

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    selected_features = list(selected_features_store or [])

    if trigger_id in ["url", "reset-btn"]:
        initialize_active_learning()

    status_message = "Annotation Phase"

    if stop_active_learning and trigger_id not in ["url", "reset-btn"]:
        status_message = (
            "Stopping criterion met: predictions stable and no uncertain samples. "
            "Reset to start a new run."
        )

    # Handle embedding click
    if trigger_id == "pca-embedding" and embedding_click_data:
        selected_sample_id = int(embedding_click_data["points"][0]["customdata"][0])
        status_message = f"Sample {selected_sample_id} selected. Click 'Annotate (Oracle)' to label it."

    # NEW: Handle load top-K uncertain button
    if trigger_id == "load-uncertain-btn":
        if stop_active_learning:
            status_message = "Active learning has converged. Click Reset to run again."
        else:
            current_batch, batch_auto_count, pool_auto_count = compute_batch(current_confidence_threshold)

            if len(current_batch) == 0:
                annotation_queue = []
                current_pointer = 0
                selected_sample_id = None
                status_message = "No samples require doctor annotation."
            else:
                annotation_queue = [int(idx) for idx in current_batch]
                current_pointer = 0
                phase = "annotation"
                selected_sample_id = annotation_queue[0]
                status_message = f"Loaded {len(annotation_queue)} samples needing doctor annotation."

    if trigger_id == "confidence-slider":
        if phase == "annotation":
            current_confidence_threshold = confidence_value
            current_batch, batch_auto_count, pool_auto_count = compute_batch(current_confidence_threshold)

            # Keep queue consistent with the latest threshold policy.
            annotation_queue = [int(idx) for idx in current_batch]
            current_pointer = 0
            if len(annotation_queue) > 0:
                selected_sample_id = int(annotation_queue[0])
                status_message = "Confidence updated (annotation phase). Queue refreshed."
            else:
                selected_sample_id = None
                status_message = "Confidence updated (annotation phase). No doctor samples at this threshold."
        else:
            status_message = "Cannot change confidence after training. Continue to next annotation phase or reset."

    # NEW: Modified annotate logic to work with queue
    if trigger_id == "annotate-btn" and phase == "annotation":
        if stop_active_learning:
            status_message = "Active learning has converged. Click Reset to run again."

        if annotations_this_round >= batch_size:
            phase = "training"
            status_message = f"Reached {batch_size} annotations. Please click Train Model."

        # If queue is empty, annotate the selected UMAP sample directly.
        elif len(annotation_queue) == 0:
            if selected_sample_id is None:
                status_message = "No sample selected. Click a UMAP point or load Top-K uncertain first."
            elif selected_sample_id in labeled_idx:
                status_message = f"Sample {selected_sample_id} is already labeled"
            else:
                annotation_queue.append(int(selected_sample_id))

        if (
            not stop_active_learning
            and phase == "annotation"
            and len(annotation_queue) > 0
            and current_pointer < len(annotation_queue)
        ):
            sample_id = int(annotation_queue.pop(current_pointer))
            if sample_id in unlabeled_idx:
                labeled_idx = np.append(labeled_idx, sample_id)
                unlabeled_idx = unlabeled_idx[unlabeled_idx != sample_id]
                oracle_annotated_idx = np.append(oracle_annotated_idx, sample_id)
                annotations_this_round += 1

            current_pointer = 0

            if annotations_this_round >= batch_size:
                annotation_queue = []
                selected_sample_id = None
                phase = "training"
                status_message = f"Reached {batch_size}/{batch_size} annotations. Please Train the Model."
            elif len(annotation_queue) > 0:
                selected_sample_id = int(annotation_queue[current_pointer])
                status_message = f"Annotated! Next sample in queue: {selected_sample_id}"
            else:
                selected_sample_id = None
                status_message = f"Annotated ({annotations_this_round}/{batch_size}). Select next sample or load Top-K uncertain."

    if trigger_id == "train-btn" and (phase == "training" or annotations_this_round > 0):
        if stop_active_learning:
            status_message = "Active learning already converged. Click Reset to run again."
        else:
            round_number += 1

            model = train_model(
                X_train[labeled_idx],
                y_train[labeled_idx],
                subjects_train[labeled_idx],
            )

            initialize_shap_explainer()

            X_train_umap_model_dgrid = compute_model_umap_embedding()

            # Compute metrics BEFORE setting phase back to annotation
            train_acc_new = accuracy_score(y_train[labeled_idx], predict_binary_with_threshold(X_train[labeled_idx]))
            test_pred_new = predict_binary_with_threshold(X_test)
            test_acc_new = accuracy_score(y_test, test_pred_new)
            cm_new = confusion_matrix(y_test, test_pred_new)
            tn, fp, fn, tp = cm_new.ravel()
            sensitivity_new = tp / (tp + fn) if (tp + fn) else 0
            specificity_new = tn / (tn + fp) if (tn + fp) else 0

            print(
                f"[ROUND {round_number}] TN={tn} FP={fp} FN={fn} TP={tp} "
                f"| Sensitivity={sensitivity_new:.4f} Specificity={specificity_new:.4f}"
            )

            # Record history for this round
            train_history.append(train_acc_new)
            test_history.append(test_acc_new)
            sensitivity_history.append(sensitivity_new)
            specificity_history.append(specificity_new)
            round_history.append(round_number)

            # Now reset for next round
            current_batch, batch_auto_count, pool_auto_count = compute_batch(confidence_value)
            annotation_queue = []
            current_pointer = 0
            current_confidence_threshold = confidence_value
            selected_sample_id = None
            annotations_this_round = 0

            # Hybrid stopping: no uncertain doctor samples + stable predictions.
            current_predictions = predict_binary_with_threshold(X_train)
            previous_round_predictions = previous_predictions.copy() if previous_predictions is not None else None
            flips = 0
            flip_ratio = 0.0
            if previous_round_predictions is not None:
                flips = int(np.sum(current_predictions != previous_round_predictions))
                flip_ratio = flips / float(len(current_predictions))

            if len(current_batch) == 0 and flip_ratio < stability_flip_ratio_threshold:
                stable_rounds += 1
            else:
                stable_rounds = 0

            previous_predictions = current_predictions.copy()
            prediction_history.append(current_predictions.copy())
            test_prediction_history.append(test_pred_new.copy())

            if stable_rounds >= stability_required_rounds and len(current_batch) == 0:
                stop_active_learning = True
                phase = "stopped"
                status_message = (
                    "Stopping criterion met: no uncertain samples and stable predictions "
                    f"for {stable_rounds} rounds (flip ratio {flip_ratio:.3f})."
                )
            else:
                phase = "annotation"
                status_message = (
                    "Model trained! Load Top-K uncertain or click UMAP to select samples. "
                    f"Stability: {stable_rounds}/{stability_required_rounds} rounds "
                    f"(flip ratio {flip_ratio:.3f})."
                )

    # Always compute current metrics for display
    train_acc = accuracy_score(y_train[labeled_idx], predict_binary_with_threshold(X_train[labeled_idx]))
    test_pred = predict_binary_with_threshold(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    # Initialize history on first load
    if len(round_history) == 0:
        train_history.append(train_acc)
        test_history.append(test_acc)
        sensitivity_history.append(sensitivity)
        specificity_history.append(specificity)
        round_history.append(0)
        if len(test_prediction_history) == 0:
            test_prediction_history.append(test_pred.copy())
        print(
            f"[ROUND 0] TN={tn} FP={fp} FN={fn} TP={tp} "
            f"| Sensitivity={sensitivity:.4f} Specificity={specificity:.4f}"
        )

    pending_queue = max(0, len(annotation_queue) - current_pointer)

    # NEW: EEG graph shows selected sample or queue preview
    eeg_sample_available = False
    if selected_sample_id is not None:
        eeg_sample_available = True
        current_sample_idx = selected_sample_id
        raw_signal = X_raw_train[selected_sample_id]

        eeg_fig = go.Figure()

        # Add raw EEG signal
        eeg_fig.add_trace(go.Scatter(
            y=raw_signal,
            mode="lines",
            line=dict(color='#1d4ed8', width=1),
            name="Raw EEG",
            showlegend=True,
        ))

        # Multi-select feature overlays in EEG panel.
        feature_descriptions = []
        if selected_features:
            for feature in selected_features:
                try:
                    highlighted_signal, description, highlight_mask = highlight_feature_in_eeg(raw_signal, feature)
                    feature_color = FEATURE_HIGHLIGHT_COLORS.get(feature, "#f59e0b")

                    if feature in ["Delta Power", "Theta Power", "Alpha Power", "Beta Power"]:
                        eeg_fig.add_trace(go.Scatter(
                            y=highlighted_signal,
                            mode="lines",
                            line=dict(color=feature_color, width=2),
                            name=f"{feature}",
                            showlegend=True,
                            opacity=0.75,
                        ))

                    if highlight_mask is not None and np.any(highlight_mask):
                        eeg_fig.add_trace(go.Scatter(
                            y=np.where(highlight_mask, raw_signal, np.nan),
                            mode="lines",
                            line=dict(color=feature_color, width=3),
                            name=f"{feature} regions",
                            showlegend=True,
                            opacity=0.9,
                        ))

                        # For spike-oriented features, add marker points on peaks for clarity.
                        if feature == "Kurtosis":
                            eeg_fig.add_trace(go.Scatter(
                                y=np.where(highlight_mask, raw_signal, np.nan),
                                mode="markers",
                                marker=dict(color=feature_color, size=6),
                                name=f"{feature} spikes",
                                showlegend=True,
                                opacity=0.95,
                            ))

                    feature_descriptions.append(f"{feature}: {description}")

                except Exception as e:
                    print(f"Error highlighting {feature}: {e}")

            if feature_descriptions:
                eeg_fig.add_annotation(
                    text="[FEATURES] " + " | ".join(feature_descriptions),
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=10, color="#b45309"),
                    bgcolor="#fffbeb",
                    bordercolor="#f59e0b",
                    borderwidth=1,
                )

        eeg_fig.update_layout(
            title=f"Sample {selected_sample_id}",
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude (μV)",
            yaxis=dict(range=[GLOBAL_Y_MIN, GLOBAL_Y_MAX]),
            template="plotly_white",
            margin=dict(l=50, r=20, t=60, b=50),
            autosize=True,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
        )

    elif pending_queue > 0:
        eeg_sample_available = True
        preview_idx = int(annotation_queue[current_pointer])
        eeg_fig = go.Figure(data=[go.Scatter(y=X_raw_train[preview_idx], mode="lines")])
        eeg_fig.update_layout(
            title=f"Queue Preview: Sample {preview_idx}",
            yaxis=dict(range=[GLOBAL_Y_MIN, GLOBAL_Y_MAX]),
            template="plotly_white",
            margin=dict(l=40, r=20, t=40, b=40),
            autosize=True,
        )
        current_sample_idx = preview_idx
    else:
        eeg_fig = go.Figure()
        eeg_fig.update_layout(
            title="No sample selected or in queue",
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=40),
            autosize=True,
        )
        current_sample_idx = 0

    # Button states
    annotate_disabled = stop_active_learning or not (phase == "annotation" and pending_queue > 0)
    if (
        phase == "annotation"
        and annotations_this_round < batch_size
        and pending_queue == 0
        and selected_sample_id is not None
        and selected_sample_id not in labeled_idx
    ):
        annotate_disabled = False
    train_disabled = stop_active_learning or annotations_this_round == 0
    confidence_slider_disabled = phase != "annotation"

    if pending_queue == 0 and phase == "annotation" and len(annotation_queue) == 0:
        status_message = "Queue empty. Load Top-K uncertain or click UMAP, then Annotate."

    if stop_active_learning:
        status_message = (
            "Stopping criterion met: predictions stable and no uncertain samples. "
            "Reset to start a new run."
        )

    if trigger_id in [None, "url", "reset-btn"]:
        pending_queue = max(0, len(annotation_queue) - current_pointer)
        status_message = (
            f"Annotation Phase | Annotated: {annotations_this_round}/{batch_size} | "
            f"Queue: {len(annotation_queue)}/{batch_size} | Pending: {pending_queue} | "
            f"Stability: {stable_rounds}/{stability_required_rounds}"
        )
        annotate_disabled = stop_active_learning or not (phase == "annotation" and pending_queue > 0)
        if (
            phase == "annotation"
            and annotations_this_round < batch_size
            and pending_queue == 0
            and selected_sample_id is not None
            and selected_sample_id not in labeled_idx
        ):
            annotate_disabled = False
        train_disabled = stop_active_learning or annotations_this_round == 0
        confidence_slider_disabled = phase != "annotation"

    # Show multi-round test-set flow (round-to-round TN/FP/FN/TP transitions).
    sankey_fig_cache = build_multiround_sankey(test_prediction_history, y_test)
    sankey_output = sankey_fig_cache
    learning_curve_fig = build_learning_curve()

    embedding_fig, embedding_type = build_embedding_figure(
        pca_view_mode,
        embedding_space_mode,
        current_confidence_threshold,
        sample_filter,
    )
    uncertainty_hist = build_uncertainty_histogram(current_confidence_threshold)

    # Keep feature contributions hidden unless the EEG panel is actively showing a sample.
    importance_mode = "contribution"
    selected_feature = selected_features[-1] if selected_features else None

    use_cached_feature_panel = trigger_id == "annotate-btn" and feature_panel_cache is not None

    if use_cached_feature_panel:
        feature_fig = feature_panel_cache["feature_fig"]
        clinical_explanation = feature_panel_cache["clinical_explanation"]
        feature_decision_header = feature_panel_cache["feature_decision_header"]
        feature_balance_bar = feature_panel_cache["feature_balance_bar"]
        feature_reflection_text = feature_panel_cache["feature_reflection_text"]
    elif eeg_sample_available:
        if selected_sample_id is not None:
            feature_sample_idx = int(selected_sample_id)
        else:
            feature_sample_idx = int(annotation_queue[current_pointer])

        feature_fig = build_feature_importance(feature_sample_idx, importance_mode, selected_features)

        clinical_explanation = "Click one or more feature bars to inspect clinically relevant EEG evidence."
        if selected_features:
            attribution_data = compute_feature_attributions(feature_sample_idx)
            contribution_value = None
            if attribution_data is not None:
                feature_names = attribution_data["feature_names"]
                attributions = attribution_data["attributions"]
                feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
                if selected_feature in feature_to_idx:
                    contribution_value = float(attributions[feature_to_idx[selected_feature]])

            if contribution_value is not None:
                direction_text = "toward seizure" if contribution_value >= 0 else "toward non-seizure"
                clinical_explanation = (
                    f"Selected ({len(selected_features)}): {', '.join(selected_features)}. "
                    f"Focus: {selected_feature} attribution={contribution_value:+.4f} ({direction_text}). "
                    f"{get_clinical_feature_explanation(selected_feature)}"
                )
            else:
                clinical_explanation = (
                    f"Selected ({len(selected_features)}): {', '.join(selected_features)}. "
                    f"Focus: {selected_feature}. {get_clinical_feature_explanation(selected_feature)}"
                )

        # Generate decision panel content for Feature Importance panel
        feature_decision_header, feature_balance_bar, feature_reflection_text = generate_feature_panel_content(
            feature_sample_idx, selected_feature
        )

        feature_panel_cache = {
            "feature_fig": feature_fig,
            "clinical_explanation": clinical_explanation,
            "feature_decision_header": feature_decision_header,
            "feature_balance_bar": feature_balance_bar,
            "feature_reflection_text": feature_reflection_text,
        }
    else:
        feature_fig = go.Figure()
        feature_fig.update_layout(
            title="No sample selected or in queue",
            template="plotly_white",
            height=260,
            margin=dict(l=120, r=14, t=22, b=44),
        )
        clinical_explanation = "No EEG sample is currently displayed. Select a sample to view feature contributions."
        feature_decision_header = html.Span(
            "Select or queue a sample to see decision insights.",
            style={"fontSize": "10px", "color": "#94a3b8"},
        )
        feature_balance_bar = ""
        feature_reflection_text = ""

        feature_panel_cache = {
            "feature_fig": feature_fig,
            "clinical_explanation": clinical_explanation,
            "feature_decision_header": feature_decision_header,
            "feature_balance_bar": feature_balance_bar,
            "feature_reflection_text": feature_reflection_text,
        }

    round_display = f"Round {round_number}"
    labeled_count = f"{len(labeled_idx)}/{len(X_train)}"
    test_accuracy_display = f"{test_acc:.4f}"
    queue_status = (
        f"Annotated: {annotations_this_round}/{batch_size} | "
        f"Queue: {len(annotation_queue)}/{batch_size} | Pending: {pending_queue} | "
        f"Stable: {stable_rounds}/{stability_required_rounds}"
    )

    return (
        eeg_fig,
        annotate_disabled,
        train_disabled,
        confidence_slider_disabled,
        f"{current_confidence_threshold:.2f}",
        status_message,
        learning_curve_fig,
        sankey_output,
        embedding_fig,
        uncertainty_hist,
        feature_fig,
        round_display,
        labeled_count,
        test_accuracy_display,
        f"{embedding_type} Embedding View",
        queue_status,
        current_sample_idx,
        clinical_explanation,
        feature_decision_header,
        feature_balance_bar,
        feature_reflection_text,
    )


# ==========================================================
# PERTURBATION MODE CALLBACKS
# ==========================================================

@app.callback(
    Output("perturbation-mode-store", "data"),
    Output("perturbation-toggle", "children"),
    Output("perturbation-toggle", "style"),
    Output("perturbation-controls", "style"),
    Output("perturbation-placeholder", "style"),
    Input("perturbation-toggle", "n_clicks"),
    State("perturbation-mode-store", "data"),
    prevent_initial_call=True,
)
def toggle_perturbation_mode(n_clicks, current_mode):
    """Toggle perturbation mode on/off."""
    new_mode = not current_mode if current_mode is not None else True

    if new_mode:
        button_text = "Disable Perturbation Mode"
        button_style = {
            "marginLeft": "20px",
            "backgroundColor": "#ef4444",
            "color": "white",
            "border": "none",
            "borderRadius": "6px",
            "padding": "4px 12px",
            "fontSize": "11px",
            "cursor": "pointer",
        }
        controls_style = {"display": "block"}
        placeholder_style = {"display": "none"}
    else:
        button_text = "Enable Perturbation Mode"
        button_style = {
            "marginLeft": "20px",
            "backgroundColor": "#f3f4f6",
            "color": "#374151",
            "border": "1px solid #d1d5db",
            "borderRadius": "6px",
            "padding": "4px 12px",
            "fontSize": "11px",
            "cursor": "pointer",
        }
        controls_style = {"display": "none"}
        placeholder_style = {"display": "block"}

    return new_mode, button_text, button_style, controls_style, placeholder_style


@app.callback(
    Output("perturbed-prediction", "children"),
    Input("slider-delta", "value"),
    Input("slider-theta", "value"),
    Input("slider-beta", "value"),
    Input("slider-kurtosis", "value"),
    State("current-sample-store", "data"),
    prevent_initial_call=True,
)
def update_perturbed_prediction(delta_shift, theta_shift, beta_shift, kurtosis_shift, sample_idx):
    """Compute perturbed prediction when sliders change."""
    if sample_idx is None or model is None:
        return "No sample selected"

    try:
        # Get original features
        original_features = X_train[sample_idx:sample_idx + 1].copy()
        perturbed_features = original_features.copy()

        # Apply perturbations (shifts in scaled space)
        # Delta Power (index 6)
        perturbed_features[0, 6] += delta_shift * np.std(X_train[:, 6])
        # Theta Power (index 7)
        perturbed_features[0, 7] += theta_shift * np.std(X_train[:, 7])
        # Beta Power (index 9)
        perturbed_features[0, 9] += beta_shift * np.std(X_train[:, 9])
        # Kurtosis (index 5)
        perturbed_features[0, 5] += kurtosis_shift * np.std(X_train[:, 5])

        # Get original and perturbed predictions
        original_probs = model.predict_proba(original_features)[0]
        perturbed_probs = model.predict_proba(perturbed_features)[0]

        original_pred = "Seizure" if original_probs[1] >= SEIZURE_PROB_THRESHOLD else "Non-Seizure"
        perturbed_pred = "Seizure" if perturbed_probs[1] >= SEIZURE_PROB_THRESHOLD else "Non-Seizure"

        # Calculate change
        prob_change = perturbed_probs[1] - original_probs[1]

        result = html.Div([
            html.Div([
                html.Strong("Original: "),
                html.Span(f"{original_pred} ({original_probs[1]:.3f})",
                          style={"color": "#ef4444" if original_pred != perturbed_pred else "#0f172a"}),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Strong("Perturbed: "),
                html.Span(
                    f"{perturbed_pred} ({perturbed_probs[1]:.3f})",
                    style={"color": "#ef4444" if perturbed_pred != original_pred else "#0f172a"},
                ),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Strong("Change: "),
                html.Span(
                    f"{prob_change:+.3f}",
                    style={"color": "#10b981" if abs(prob_change) < 0.1 else "#ef4444"},
                ),
            ]),
        ])

        return result

    except Exception as e:
        return f"Error: {str(e)}"


@app.callback(
    Output("slider-delta", "value"),
    Output("slider-theta", "value"),
    Output("slider-beta", "value"),
    Output("slider-kurtosis", "value"),
    Input("reset-perturbation", "n_clicks"),
    prevent_initial_call=True,
)
def reset_perturbation_sliders(n_clicks):
    """Reset all perturbation sliders to 0."""
    return 0, 0, 0, 0


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMBINED DASHBOARD - FINAL VERSION")
    print("=" * 70)
    print("[OK] Feature Explanation System (from vis_combined_gourisha.py)")
    print("[OK] Interactive UMAP-to-EEG (from vis_combined_gauri.py)")
    print("[OK] Annotation Queue System")
    print("[OK] Load Top-K Uncertain Button")
    print("[OK] Annotate Button supports direct labeling")
    print("[OK] DGrid Transformation (no overlapping points)")
    print("[OK] Decision Boundary Visualization")
    print("=" * 70)
    print("Opening dashboard at http://127.0.0.1:8050")
    print("=" * 70)
    print("\nHow to use:")
    print("1. Click any point in the UMAP embedding")
    print("2. Click 'Annotate (Oracle)' to label selected sample")
    print("3. Or click 'Load Top-K Uncertain' to auto-load uncertain samples")
    print("4. Click 'Annotate (Oracle)' to label queued samples")
    print("5. Click feature bars to see EEG evidence")
    print("=" * 70 + "\n")

    app.run(debug=True)

