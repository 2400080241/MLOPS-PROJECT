# Smart Agriculture MLOps Project

## Overview

Smart Agriculture is a machine-learning application that provides agricultural predictions using soil and weather information.

The project includes four machine-learning models:

1. Crop Recommendation
2. Crop Yield Prediction
3. Soil Health Classification
4. Irrigation Requirement Prediction

The project also includes MLflow experiment tracking, DVC data versioning, automated tests, and drift analysis.

## Features

- Predict suitable crops using agricultural input features.
- Estimate crop yield without changing the dataset's original units.
- Classify soil health.
- Predict whether irrigation is required.
- Track training runs and metrics using MLflow.
- Version datasets and pipelines using DVC.
- Run automated tests using pytest and GitHub Actions.
- Inspect the generated drift report.

## Technology Stack

- Python
- FastAPI
- scikit-learn
- pandas and NumPy
- Joblib
- MLflow
- DVC
- pytest
- GitHub Actions
- Jinja2

## Project Structure

```text
smart-agriculture-mlops/
├── app.py
├── train.py
├── validation.py
├── drift.py
├── inspect_models.py
├── evaluate_models.py
├── models/
│   ├── crop_recommendation_model.pkl
│   ├── crop_yield_model.pkl
│   ├── irrigation_model.pkl
│   └── soil_health_model.pkl
├── data/
│   ├── agriculture.csv
│   ├── real_agriculture.csv
│   └── processed/
│       └── agriculture_clean.csv
├── templates/
│   └── index.html
├── tests/
│   ├── test_api.py
│   └── test_validation.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── drift_report.html
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd smart-agriculture-mlops
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn pandas numpy scikit-learn joblib jinja2 mlflow dvc pytest httpx
```

## Run the Web Application

Start the FastAPI application:

```bash
python -m uvicorn app:app --reload
```

Open the following address in your browser:

http://127.0.0.1:8000

Use the web interface to submit agricultural inputs and view the available predictions.

## Model Evaluation

Run the evaluation script:

```bash
python evaluate_models.py
```

The script reports classification metrics for crop recommendation, soil health, and irrigation, along with MAE, RMSE, and R-squared for crop yield.

Note: Evaluation against data used during training can produce overly optimistic results. Use a held-out test set for a reliable estimate of generalization performance.

## MLflow Experiment Tracking

Start the MLflow UI in a separate terminal:

```bash
python -m mlflow ui --host 127.0.0.1 --port 5000
```

Open:

http://127.0.0.1:5000

The existing tracking store contains experiment runs for crop yield, soil health, and irrigation models. The dashboard displays recorded metrics and run information.

The tracking store must match the one used during training to display the expected experiments.

## DVC Data Versioning

Check the status of data and pipelines:

```bash
dvc status
```

List configured DVC remotes:

```bash
dvc remote list
```

The current local DVC status has reported that data and pipelines are up to date. No DVC remote is currently configured.

A remote can be configured later to store versioned data outside the local machine.

## Automated Tests

Run the test suite:

```bash
python -m pytest
```

The project also uses GitHub Actions to run automated tests when changes are pushed to GitHub.

## Drift Analysis

The project contains `drift.py` and a generated `drift_report.html`.

Open the HTML report in a browser to inspect the available drift analysis. The report should be regenerated using the project's configured drift-analysis workflow whenever new data needs to be evaluated.

## Model Files

The trained models are stored in the `models/` directory and loaded by the application.

Keep model files, training code, feature names, and input preprocessing consistent so predictions use the expected inputs.

## Future Improvements

- Evaluate all models on held-out test data.
- Improve dataset size and diversity.
- Configure a remote DVC storage location.
- Automate model evaluation and drift checks.
- Expand CI checks for data and model quality.

## License

Add the appropriate license for your project before distributing it publicly.
