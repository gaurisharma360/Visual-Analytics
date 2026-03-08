# ==========================================================
# FULL HYBRID ACTIVE LEARNING DASHBOARD
# STRICT PHASE CONTROL + TERMINAL STYLE CORE DISPLAY
# REFRESH SAFE + FINAL SUMMARY SAFE
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

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
        beta  = band_power(13, 30)

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

    X_raw = df.drop(['ID', 'Y'], axis=1).values
    y_original = df['Y'].values

    X = extract_features(X_raw)
    y = (y_original == 'E').astype(int)

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
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_

# ==========================================================
# TERMINAL REPORT BUILDER
# ==========================================================

def build_terminal_report(round_number,
                          labeled_count,
                          train_acc,
                          test_acc,
                          cm,
                          sensitivity,
                          specificity,
                          class_report,
                          doctor_count,
                          batch_auto_count,
                          pool_auto_count):

    return f"""
======================================================================
ROUND {round_number}
Labeled samples: {labeled_count}

Train Accuracy: {round(train_acc,4)}
Test Accuracy : {round(test_acc,4)}

Confusion Matrix (Test):
{cm}

Sensitivity (Seizure Recall): {round(sensitivity,4)}
Specificity (Non-Seizure Recall): {round(specificity,4)}

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
# DATA LOAD
# ==========================================================

X_raw_train, X_train, X_test, y_train, y_test, subjects_train = load_and_split()

initial_fraction = 0.2
batch_size = 10
confidence_threshold = 0.7

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

global train_history, test_history, round_history

train_history = []
test_history = []
round_history = []
# ==========================================================
# INITIALIZATION FUNCTION (RESET SAFE)
# ==========================================================

def initialize_active_learning():

    global labeled_idx, unlabeled_idx
    global model, current_batch
    global batch_auto_count, pool_auto_count
    global current_pointer, round_number, phase
    global train_history, test_history, round_history
    global sensitivity_history, specificity_history


    # RESET LEARNING CURVE HISTORY
    train_history = []
    test_history = []
    round_history = []

    sensitivity_history = []
    specificity_history = []

    n_initial = int(initial_fraction * len(X_train))
    indices = np.random.permutation(len(X_train))

    labeled_idx = indices[:n_initial]
    unlabeled_idx = indices[n_initial:]

    round_number = 0
    phase = "annotation"

    model = train_model(
        X_train[labeled_idx],
        y_train[labeled_idx],
        subjects_train[labeled_idx]
    )

    current_batch, batch_auto_count, pool_auto_count = compute_batch(confidence_threshold)
    current_pointer = 0
def build_learning_curve():

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=round_history,
        y=train_history,
        mode='lines+markers',
        name='Train Accuracy'
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=test_history,
        mode='lines+markers',
        name='Test Accuracy'
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=sensitivity_history,
        mode='lines+markers',
        name='Sensitivity'
    ))

    fig.add_trace(go.Scatter(
        x=round_history,
        y=specificity_history,
        mode='lines+markers',
        name='Specificity'
    ))

    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Performance",
        yaxis=dict(range=[0,1]),
        xaxis=dict(
        range=[0, max(1, max(round_history))],
        dtick=1
    ),
        template="plotly_white"
    )

    return fig

def build_confusion_heatmap(cm):

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred: Non-Seizure", "Pred: Seizure"],
        y=["Actual: Non-Seizure", "Actual: Seizure"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues"
    ))

    fig.update_layout(title="Confusion Matrix")

    return fig

def build_data_donut():

    labeled = len(labeled_idx)
    doctor = len(current_batch)           # uncertain samples in batch
    auto_batch = batch_auto_count         # confident inside batch
    auto_pool = pool_auto_count           # confident outside batch

    fig = go.Figure(data=[go.Pie(
        labels=[
            "Labeled",
            "Auto (Pool)",
            "Auto (Batch)",
            "Needs Doctor"
        ],
        values=[
            labeled,
            auto_pool,
            auto_batch,
            doctor
        ],
        hole=0.5
    )])

    fig.update_layout(title="Data State Distribution")
    
    return fig

# Initialize first time
initialize_active_learning()

# ==========================================================
# DASH APP
# ==========================================================


app = dash.Dash(__name__)

app.layout = html.Div([

    dcc.Location(id="url", refresh=True),

    html.H2("NeuroLens Active Learning Dashboard"),

    html.Div(id="status-message", style={"fontWeight":"bold"}),

    html.Button("Annotate (Oracle)", id="annotate-btn"),
    html.Button("Train Model", id="train-btn", disabled=True),

    html.Hr(),

    html.H4("Confidence Threshold"),

    dcc.Slider(
        id="confidence-slider",
        min=0.5,
        max=0.99,
        step=0.01,
        value=0.7,
        marks={
            0.5: "0.5",
            0.6: "0.6",
            0.7: "0.7",
            0.8: "0.8",
            0.9: "0.9",
            0.99: "0.99"
        },
    ),

    html.Div(id="confidence-value", style={"marginBottom":"20px"}),

    html.Hr(),

    # Top row
    html.Div([
        dcc.Graph(id="confusion-heatmap", style={"width":"50%", "display":"inline-block"}),
        dcc.Graph(id="data-donut", style={"width":"50%", "display":"inline-block"})
    ]),

    html.Hr(),

    html.H3("EEG Annotation Pane"),
    dcc.Graph(id="eeg-graph"),

    html.Hr(),

    html.H3("Active Learning Core Output"),
    html.Div(id="performance-metrics"),
    
    html.Hr(),

    html.H3("Learning Curves (Accuracy vs Rounds)"),
    dcc.Graph(id="learning-curve"),


])

# ==========================================================
# MAIN CALLBACK (MERGED RESET + LOGIC)
# ==========================================================

@app.callback(
    Output("eeg-graph","figure"),
    Output("performance-metrics","children"),
    Output("annotate-btn","disabled"),
    Output("train-btn","disabled"),
    Output("confidence-value","children"),
    Output("status-message","children"),
    Output("learning-curve","figure"),
    Output("confusion-heatmap","figure"),
    Output("data-donut","figure"),
    Input("url","pathname"),
    Input("annotate-btn","n_clicks"),
    Input("train-btn","n_clicks"),
    Input("confidence-slider","value"),
    prevent_initial_call=False
)
def update_dashboard(pathname, annotate_clicks, train_clicks, confidence_value):

    global labeled_idx, unlabeled_idx
    global current_batch, current_pointer
    global batch_auto_count, pool_auto_count
    global model, phase, round_number

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # =====================================================
    # REFRESH RESET
    # =====================================================
    if trigger_id == "url":
        initialize_active_learning()

    # =====================================================
    # INITIAL DISPLAY
    # =====================================================
    if trigger_id in [None, "url"]:

        train_acc = accuracy_score(
            y_train[labeled_idx],
            model.predict(X_train[labeled_idx])
        )

        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        heatmap_fig = build_confusion_heatmap(cm)
        donut_fig = build_data_donut()
        

        sensitivity = tp/(tp+fn) if (tp+fn) else 0
        specificity = tn/(tn+fp) if (tn+fp) else 0
        # Store baseline (Round 0) only once
        if len(round_history) == 0:
            train_history.append(train_acc)
            test_history.append(test_acc)
            sensitivity_history.append(sensitivity)
            specificity_history.append(specificity)
            round_history.append(0)

        class_report = classification_report(
            y_test, test_pred,
            target_names=["Non-Seizure","Seizure"]
        )

        terminal_block = build_terminal_report(
            round_number,
            len(labeled_idx),
            train_acc,
            test_acc,
            cm,
            sensitivity,
            specificity,
            class_report,
            len(current_batch),
            batch_auto_count,
            pool_auto_count
        )

        fig = go.Figure(data=[go.Scatter(
            y=X_raw_train[current_batch[0]],
            mode='lines'
        )])

        return fig, html.Pre(terminal_block), False, True, "Annotation Phase",f"Confidence Threshold: {confidence_value}",build_learning_curve(), heatmap_fig, donut_fig
    
    # =====================================================
    # SLIDER UPDATE (ONLY BETWEEN ROUNDS)
    # =====================================================
    if trigger_id == "confidence-slider":

        # allow change only when we are not annotating a batch
        if phase != "annotation":
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                f"Confidence Threshold: {confidence_value}",
                "Slider only active between rounds",
                dash.no_update,
                dash.no_update,
                dash.no_update
            )

        # recompute batch using new threshold
        current_batch, batch_auto_count, pool_auto_count = compute_batch(confidence_value)
        current_pointer = 0

        # rebuild figures
        test_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, test_pred)

        heatmap_fig = build_confusion_heatmap(cm)
        donut_fig = build_data_donut()

        if len(current_batch) > 0:
            fig = go.Figure(data=[go.Scatter(
                y=X_raw_train[current_batch[0]],
                mode='lines'
            )])
        else:
            fig = go.Figure()

        return (
            fig,
            dash.no_update,
            False,
            True,
            f"Confidence Threshold: {confidence_value}",
            "Annotation Phase",
            build_learning_curve(),
            heatmap_fig,
            donut_fig
        )
    # =====================================================
    # ANNOTATION PHASE
    # =====================================================
    if trigger_id == "annotate-btn" and phase == "annotation":

        sample_id = current_batch[current_pointer]
        labeled_idx = np.append(labeled_idx, sample_id)
        unlabeled_idx = unlabeled_idx[unlabeled_idx != sample_id]
        current_pointer += 1
    # Recompute figures (needed for 9-output schema)
        test_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, test_pred)

        heatmap_fig = build_confusion_heatmap(cm)
        donut_fig = build_data_donut()
        if current_pointer >= len(current_batch):
            phase = "training"
            return go.Figure(), html.Pre("Round complete. Please Train the Model."), True, False, "Training Phase",f"Confidence Threshold: {confidence_value}",build_learning_curve(), heatmap_fig, donut_fig


        fig = go.Figure(data=[go.Scatter(
            y=X_raw_train[current_batch[current_pointer]],
            mode='lines'
        )])

        return fig, dash.no_update, False, True, "Annotation Phase",f"Confidence Threshold: {confidence_value}",build_learning_curve(), heatmap_fig, donut_fig


    # =====================================================
    # TRAIN PHASE
    # =====================================================
    if trigger_id == "train-btn" and phase == "training":

        round_number += 1

        model = train_model(
            X_train[labeled_idx],
            y_train[labeled_idx],
            subjects_train[labeled_idx]
        )

        train_acc = accuracy_score(
            y_train[labeled_idx],
            model.predict(X_train[labeled_idx])
        )

        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        heatmap_fig = build_confusion_heatmap(cm)
        

        sensitivity = tp/(tp+fn) if (tp+fn) else 0
        specificity = tn/(tn+fp) if (tn+fp) else 0


        train_history.append(train_acc)
        test_history.append(test_acc)
        sensitivity_history.append(sensitivity)
        specificity_history.append(specificity)
        round_history.append(round_number)

        class_report = classification_report(
            y_test, test_pred,
            target_names=["Non-Seizure","Seizure"]
        )

        current_batch, batch_auto_count, pool_auto_count = compute_batch(confidence_value)
        donut_fig = build_data_donut()
        current_pointer = 0
        phase = "annotation"

        if len(current_batch) == 0:

            terminal_block = build_terminal_report(
                round_number,
                len(labeled_idx),
                train_acc,
                test_acc,
                cm,
                sensitivity,
                specificity,
                class_report,
                0,
                0,
                pool_auto_count
            )

            return go.Figure(), html.Pre(terminal_block + "\nACTIVE LEARNING COMPLETE."), True, True, "STOPPED",f"Confidence Threshold: {confidence_value}",build_learning_curve(), heatmap_fig, donut_fig


        terminal_block = build_terminal_report(
            round_number,
            len(labeled_idx),
            train_acc,
            test_acc,
            cm,
            sensitivity,
            specificity,
            class_report,
            len(current_batch),
            batch_auto_count,
            pool_auto_count
        )

        fig = go.Figure(data=[go.Scatter(
            y=X_raw_train[current_batch[0]],
            mode='lines'
        )])

        return fig, html.Pre(terminal_block), False, True, "Annotation Phase",f"Confidence Threshold: {confidence_value}",build_learning_curve(), heatmap_fig, donut_fig

# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)