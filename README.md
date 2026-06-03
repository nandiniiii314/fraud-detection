# Credit Card Fraud Detection

This project is a fraud detection system built for identifying suspicious credit card transactions. It combines data analysis, machine learning, and a Flask web application so that transaction records can be explored, modeled, and tested through a simple user interface.

The main goal of the project is to classify transactions as legitimate or fraudulent using historical transaction data. Since fraud datasets are usually highly imbalanced, the project focuses on preprocessing, feature scaling, model training, and evaluation before using the trained models inside the web app.

## Project Overview

Credit card fraud detection is a binary classification problem:

- `0` means the transaction is normal.
- `1` means the transaction is fraudulent.

The dataset contains anonymized transaction features, transaction amount, and fraud labels. The project uses these features to train machine learning models that can detect patterns linked to fraudulent activity.

## What This Project Includes

- Exploratory data analysis using R.
- Data cleaning and preprocessing.
- Model training for fraud classification.
- Saved machine learning model files.
- A Flask web application for prediction.
- HTML templates and static assets for the user interface.
- Generated analysis output from R scripts.

## Repository Structure

```text
fraud_detection_complete/
├── app.py
├── README.md
├── .gitignore
├── fraud_detection/
│   ├── app.py
│   ├── requirements.txt
│   ├── creditcard.csv
│   ├── creditcard clean.csv
│   ├── models/
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   ├── scaler.pkl
│   │   ├── pca.pkl
│   │   └── metadata.json
│   ├── r_scripts/
│   │   ├── eda.R
│   │   ├── modeling.R
│   │   ├── modeling_fast.R
│   │   └── report.R
│   ├── static/
│   └── templates/
└── r_output/
```

## Dataset

The project uses credit card transaction data with anonymized numerical features. These types of datasets commonly include transformed variables such as PCA-based features, transaction amount, transaction time, and a target class label.

Because transaction datasets can be large or sensitive, raw data files are ignored from GitHub by default. Anyone running the project should place the required dataset files inside the `fraud_detection/` folder.

Expected data files:

- `creditcard.csv`
- `creditcard clean.csv`

## Machine Learning Workflow

The general workflow is:

1. Load the transaction dataset.
2. Explore fraud and non-fraud transaction patterns.
3. Clean and preprocess the data.
4. Scale numerical features.
5. Train classification models.
6. Evaluate model performance.
7. Save trained models.
8. Use the saved models in the Flask application.

## Models

The project contains saved model artifacts in `fraud_detection/models/`.

Current model files include:

- Logistic Regression model
- Random Forest model
- Feature scaler
- PCA transformer
- Model metadata

These files are used by the application to process input data and generate fraud predictions.

## R Scripts

The `fraud_detection/r_scripts/` folder contains R scripts for analysis and modeling:

- `eda.R` performs exploratory data analysis.
- `modeling.R` trains and evaluates models.
- `modeling_fast.R` provides a faster modeling workflow.
- `report.R` generates report-style output.

Generated charts, tables, and reports are saved to output folders and are not meant to be committed unless specifically needed.

## Flask Web Application

The Flask app provides an interface for using the fraud detection model. It loads the trained model artifacts, accepts transaction input, applies preprocessing, and returns a prediction result.

Main application files:

- `app.py`
- `fraud_detection/app.py`
- `fraud_detection/templates/`
- `fraud_detection/static/`

## Setup

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r fraud_detection/requirements.txt
```

## Run The Project

Run from the project root:

```bash
python app.py
```

Or run the main application directly:

```bash
python fraud_detection/app.py
```

After the server starts, open the local Flask URL shown in the terminal.

## Files Not Pushed To GitHub

The `.gitignore` file excludes files that should usually stay local, including:

- virtual environments
- secret files
- large datasets
- generated output folders
- trained model artifacts
- logs and cache files

This keeps the GitHub repository clean and avoids uploading large or sensitive files.

## Project Purpose

This project is useful for learning how fraud detection systems are built end to end: from data analysis and model training to deploying a working prediction interface. It demonstrates both the data science workflow and the application layer needed to make a model usable.

