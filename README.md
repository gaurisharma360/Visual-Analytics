# Visual Analytics APP

## About this app

This is an **Active Learning EEG Dashboard** for seizure detection and classification. The app combines feature engineering, machine learning explainability (SHAP), and interactive visualizations using Dash and Plotly.

**Key Features:**
- EEG signal processing with statistical and spectral feature extraction
- Active learning with uncertainty-driven sample selection
- Model explainability with SHAP and feature attribution
- Interactive 2D embeddings (UMAP) of EEG data
- Real-time EEG highlighting based on engineered features


## Requirements

* Python 3.13.5 or higher (add it to your path (system variables) to make sure you can access it from the command prompt)
* Git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## How to run this app

We suggest you create a virtual environment for running this app with Python 3. Clone or download this repository and open your terminal/command prompt in the root folder.

### Setup Instructions

**Option 1: Using Python Virtual Environment (Recommended for Windows)**

Open a command prompt and run the following commands in the project root directory:

```
> python -m venv venv
```

If `python` is not recognized, use `python3` instead.

**Activate the virtual environment:**

In Windows: 
```
> venv\Scripts\activate
```

In Unix/Linux/Mac:
```
> source venv/bin/activate
```

**Option 2: Using Anaconda/Miniconda**

Alternatively, you can use Anaconda or Miniconda:

* Anaconda (https://www.anaconda.com/) - includes a user-friendly UI but requires more space
* Miniconda (https://docs.conda.io/en/latest/miniconda.html) - Command Prompt based, no UI, but requires less space

Create and activate a conda environment:
```
> conda create -n eeg-dashboard python=3.11
> conda activate eeg-dashboard
```

### Install Dependencies

Install all required packages by running:
```
> pip install -r requirements.txt
```

This installs the following key dependencies:
- **dash** & **plotly** - Interactive web dashboards and visualizations
- **numpy**, **pandas**, **scipy** - Numerical and signal processing
- **scikit-learn** - Machine learning models and preprocessing
- **umap-learn** - Dimensionality reduction for embedding visualization
- **shap** - Model explainability and feature attribution

### Run the App

Run the active learning dashboard locally with:
```
> python dashboard/vis_combined_forsubmission.py
```

Open your browser and navigate to the HTTP link provided in the terminal (typically `http://127.0.0.1:8050/`). 

You can edit the code in any editor (e.g., Visual Studio Code) and refresh the browser to see the results in real-time.

## Data

The app expects a CSV file named `bonn_eeg_combined.csv` in the project root directory with the following structure:
- **ID**: Sample identifier
- **Y**: Label (e.g., "E" for seizure, others for non-seizure)
- Remaining columns: Raw EEG signal values (1D time-series data)

## Resources

* [Dash Documentation](https://dash.plot.ly/)
* [Plotly Documentation](https://plotly.com/python/)
* [SHAP Documentation](https://shap.readthedocs.io/)
* [UMAP Documentation](https://umap-learn.readthedocs.io/)
