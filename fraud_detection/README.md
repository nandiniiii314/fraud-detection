  # 🛡 FraudGuard — Credit Card Fraud Detection

## ✅ Dataset is already included
`creditcard.csv` is bundled in this folder. No download needed!

---

## 🚀 Setup & Run

### Step 1 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
python app.py
```

### Step 3 — Open browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
fraud_detection/
├── app.py                    ← Flask backend + ML pipeline
├── creditcard.csv            ← Dataset (284,807 transactions)
├── requirements.txt
├── README.md
│
├── r_scripts/
│   ├── eda.R                 ← EDA with ggplot2 (5 plots)
│   ├── modeling.R            ← R LR + RandomForest models
│   └── report.R              ← R Markdown HTML report
│
├── r_output/                 ← Auto-created when R runs
│   ├── r_class_distribution.png
│   ├── r_roc_curves.png
│   └── fraud_report.html     ← R Markdown output
│
├── models/                   ← Auto-created after training
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## 🔬 Python ML Pipeline

| Step | Detail |
|------|--------|
| StandardScaler | Normalize all 30 features |
| PCA | Reduce to N components (default: 10) |
| Train/Test Split | Configurable (default: 80/20) |
| SMOTE | Balance fraud/legit classes |
| Logistic Regression | sklearn, max_iter=1000 |
| Random Forest | 100 trees, sklearn |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC |

---

## R Language Integration

| Script | Packages | Output |
|--------|----------|--------|
| `eda.R` | ggplot2, dplyr, scales | 5 dark-themed plots |
| `modeling.R` | randomForest, caret, pROC, ROSE | ROC, importance, metrics plots |
| `report.R` | rmarkdown, knitr, kableExtra | HTML report |

### Install R (if not installed)
- Windows: https://cran.r-project.org/bin/windows/base/
- Mac: https://cran.r-project.org/bin/macosx/
- Linux: `sudo apt install r-base`

### R packages auto-install on first run!

---

## ❗ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 5000 in use | Change last line of app.py to `port=5001` |
| R not found | Install R from r-project.org, restart VS Code |
| R packages fail | Run `Rscript r_scripts/eda.R` manually in terminal |
