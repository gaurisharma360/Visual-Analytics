# Visual Analytics APP

## About this app

This is an **Active Learning EEG Dashboard** for seizure detection and classification. The app combines feature engineering, machine learning explainability, and interactive visualizations using Dash and Plotly.

**Key Features:**
- EEG signal processing with statistical and spectral feature extraction
- Active learning with uncertainty-driven sample selection
- Model explainability with  feature attribution
- Interactive 2D embeddings (UMAP) of EEG data
- Uncertainty histogram that gives insights on data distribution.
- Real-time EEG highlighting based on engineered features
- Learning Curve (showcasing Sensitivity, Specificity, Test and Train Accuracy)
- Sankey Diagram that shows prediction flow across rounds
- Confidence Slider 



## Requirements

* Python 3.13.5 or higher (add it to your path (system variables) to make sure you can access it from the command prompt)
* Git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## How to run this app

We suggest you create a virtual environment for running this app with Python 3. Clone or download this repository and open your terminal/command prompt in the root folder.

### Setup Instructions

**Option 1: Using Python Virtual Environment (Recommended for Windows)**

Open a command prompt and run the following commands in the project root directory:

```
> python -m venv .venv
```

If `python` is not recognized, use `python3` instead.

**Activate the virtual environment:**

In Windows: 
```
> .venv\Scripts\activate
```

In Unix/Linux/Mac:
```
> source .venv/bin/activate
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


### Run the App

Run the active learning dashboard locally with:
```
> python dashboard/vis_combined_forsubmission.py
```

Open your browser and navigate to the HTTP link provided in the terminal (typically `http://127.0.0.1:8050/`). 

You can edit the code in any editor (e.g., Visual Studio Code) and refresh the browser to see the results in real-time.

## Data

The dataset we decided to use is the **University of Bonn EEG Dataset**.

- **Composition**: It contains five folders, each with 100 channels of EEG segments.
- **Recording Duration**: Each individual was recorded for 23.6 seconds.
- **Size**: 500 EEG segments (100 segments per subset across five subsets, ensuring balanced class distribution).
- **Data Type**: Single-channel EEG signals stored in text files (ASCII format).
- **Sampling Rate**: 173.61 Hz, providing high temporal resolution.
- **Segment Length**: 23.6 seconds per segment, equivalent to 4,096 data points.
- **Labels**: Clearly defined for each segment:
	- A: healthy, eyes open
	- B: healthy, eyes closed
	- C: interictal, epileptogenic zone
	- D: interictal, opposite hemisphere
	- E: ictal

The attributes are the 4,096 continuous EEG amplitude measurements recorded sequentially over 23.6 seconds, along with a categorical class label.

## Resources

* [Dash Documentation](https://dash.plot.ly/)
* [Plotly Documentation](https://plotly.com/python/)
* [UMAP Documentation](https://umap-learn.readthedocs.io/)
