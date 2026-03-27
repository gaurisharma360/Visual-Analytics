# AI Usage and Work Attribution

This document explains how AI was used in this project and clearly separates:
1. what we implemented ourselves, and
2. what is provided by external libraries.

## AI Usage 

AI tools were used as coding support (idea generation, refactoring suggestions), while we made the final engineering decisions, selected approaches, integrated components, and validated the behavior in the app.

## What we implemented 

The following logic and product decisions are authored and integrated by me in `dashboard/vis_combined_forsubmission.py`:

- End-to-end application design for an active-learning EEG dashboard workflow.
- Feature engineering pipeline orchestration for EEG segments.
- Custom overlap-reduction transform for embedding display.
- Grouped subject split strategy and dataset preparation.
- Active-learning loop and state machine (annotation/training phases, queue handling, stopping criteria).
- UI composition and dashboard interactions (layout, callback wiring, queue controls, uncertainty views, EEG highlighting).
- Domain-specific interpretation glue code that maps model outputs to user-facing explanations.
- Visualization assembly logic for this specific workflow (learning curve, uncertainty histogram, embedding overlays, confusion and Sankey plots).

In short: the overall architecture, app behavior, workflow rules, and integration across all components.

## What external libraries provide

External packages execute core algorithms and rendering primitives. We configure and combine them, but these low-level implementations are library code.

### Data and numeric processing
- `numpy`: arrays, vectorized math, random seed control.
- `pandas`: CSV loading and dataframe operations.

### Signal processing and statistics
- `scipy.signal` (`welch`, `butter`, `filtfilt`): PSD estimation and digital filtering.
- `scipy.stats` (`skew`, `kurtosis`): statistical moments.
- `scipy.ndimage` filters: rolling/smoothing operations used in feature-region highlighting.

### Machine learning
- `scikit-learn`:
  - `GroupShuffleSplit` and `GroupKFold` for grouped splitting/CV.
  - `Pipeline`, `StandardScaler`, `LogisticRegression` for model training.
  - `GridSearchCV` for hyperparameter search.
  - `accuracy_score`, `confusion_matrix` for evaluation metrics.

### Embedding and explainability
- `umap-learn` (`UMAP`): 2D embedding algorithm.
- `shap` (`LinearExplainer`): SHAP-based feature attribution when available.

### Dashboard and plotting
- `dash`: app server, layout components, callback runtime.
- `plotly` (`graph_objs`, `express`): chart primitives and rendering.

## Evidence sources in this repository

- Main implementation: `dashboard/vis_combined_forsubmission.py`
- Dependency list: `requirements.txt`
- Project-level usage instructions: `README.md`

