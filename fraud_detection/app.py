from flask import Flask, render_template, request, jsonify, send_file, url_for
import pandas as pd
import numpy as np
import json, os, io, base64, subprocess, warnings, shutil, threading, webbrowser
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# ── Global State ────────────────────────────────────────
trained_models  = {}
scaler_global   = None
pca_global      = None
feature_cols    = []
feature_means_global = {}
df_global       = None
results_global  = {}
BASE_DIR        = os.path.dirname(__file__)
DATASET_PATH    = os.path.join(BASE_DIR, "creditcard.csv")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
PCA_PATH        = os.path.join(MODELS_DIR, "pca.pkl")
MODEL_PATHS     = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
}
MODEL_ALIASES   = {
    "random forest": "Random Forest",
    "smart pattern recognition": "Random Forest",
    "logistic regression": "Logistic Regression",
    "simple probability check": "Logistic Regression",
}
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
R_SCRIPTS       = {
    "eda": "eda.R",
    "modeling": "modeling_fast.R",
    "report": "report.R",
}
R_TIMEOUT_SECONDS = 900
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def reset_training_state():
    global trained_models, scaler_global, pca_global, results_global, feature_means_global
    trained_models = {}
    scaler_global = None
    pca_global = None
    results_global = {}
    feature_means_global = {}

# ── Helpers ─────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def dark_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return fig, ax

def find_rscript():
    env_path = os.environ.get("RSCRIPT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    in_path = shutil.which("Rscript")
    if in_path:
        return in_path

    candidates = []
    if os.name == "nt":
        search_roots = [
            Path("C:/Program Files/R"),
            Path("C:/Program Files (x86)/R"),
            Path.home() / "AppData/Local/Programs/R",
        ]
        patterns = ("R-*/bin/Rscript.exe", "R-*/bin/x64/Rscript.exe")
        for root in search_roots:
            if not root.exists():
                continue
            for pattern in patterns:
                candidates.extend(root.glob(pattern))

    if not candidates:
        return None

    candidates = sorted({str(path.resolve()) for path in candidates}, reverse=True)
    return candidates[0]

def run_rscript(args, timeout=R_TIMEOUT_SECONDS):
    rscript = find_rscript()
    if not rscript:
        raise FileNotFoundError("Rscript not found")
    return subprocess.run(
        [rscript, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.path.dirname(__file__)
    )

def set_feature_context(df):
    global df_global, feature_cols, feature_means_global
    df_global = df
    feature_cols = [c for c in df.columns if c != "Class"]
    feature_means_global = {c: float(df[c].mean()) for c in feature_cols}

def load_feature_context_from_disk():
    if not os.path.exists(DATASET_PATH):
        return False
    set_feature_context(pd.read_csv(DATASET_PATH))
    return True

def ensure_prediction_assets():
    global scaler_global, pca_global, trained_models, feature_cols, feature_means_global

    if trained_models and scaler_global is not None and pca_global is not None and feature_cols and feature_means_global:
        return True, None

    metadata = load_training_metadata()
    if metadata:
        feature_cols = metadata.get("feature_cols", feature_cols)
        feature_means_global = {
            key: float(value)
            for key, value in (metadata.get("feature_means") or {}).items()
        }

    if (not feature_cols or not feature_means_global) and not load_feature_context_from_disk():
        return False, "creditcard.csv not found in project folder."

    if scaler_global is None and os.path.exists(SCALER_PATH):
        scaler_global = joblib.load(SCALER_PATH)
    if pca_global is None and os.path.exists(PCA_PATH):
        pca_global = joblib.load(PCA_PATH)

    for model_name, model_path in MODEL_PATHS.items():
        if model_name not in trained_models and os.path.exists(model_path):
            trained_models[model_name] = joblib.load(model_path)

    if trained_models and scaler_global is not None and pca_global is not None:
        return True, None

    return False, "Train models first."

def resolve_model_name(model_name):
    raw_name = str(model_name or "").strip()
    if raw_name in trained_models:
        return raw_name
    return MODEL_ALIASES.get(raw_name.lower())

def asset_mtime(*relative_parts):
    path = os.path.join(BASE_DIR, *relative_parts)
    try:
        return int(os.path.getmtime(path))
    except OSError:
        return 0

def load_training_metadata():
    if not os.path.exists(MODEL_METADATA_PATH):
        return {}
    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

def open_dashboard_in_edge(url):
    edge_candidates = []

    for candidate in (
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
    ):
        if not candidate:
            continue
        if candidate.lower().endswith(".exe"):
            edge_candidates.append(candidate)
        else:
            edge_candidates.append(os.path.join(candidate, "Microsoft", "Edge", "Application", "msedge.exe"))

    for edge_path in edge_candidates:
        if edge_path and os.path.exists(edge_path):
            try:
                subprocess.Popen(
                    [edge_path, "--new-window", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True
            except OSError:
                continue

    try:
        return webbrowser.open(url, new=2)
    except Exception:
        return False

def schedule_dashboard_open(host, port):
    if os.environ.get("FRAUDGUARD_OPEN_BROWSER", "1").strip().lower() in {"0", "false", "no"}:
        return

    url = f"http://{host}:{port}/"

    def _open():
        open_dashboard_in_edge(url)

    threading.Timer(1.2, _open).start()

# ── Routes ───────────────────────────────────────────────
@app.after_request
def add_no_cache_headers(response):
    if request.path == "/" or request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    if request.path.startswith("/api/"):
        origin = request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Vary"] = "Origin"
    return response

@app.route("/")
def index():
    return render_template(
        "index.html",
        dataset_exists=os.path.exists(DATASET_PATH),
        style_version=asset_mtime("static", "css", "style.css"),
        script_version=asset_mtime("static", "js", "app.js"),
    )

@app.route("/api/load_dataset", methods=["POST"])
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"success": False, "error": "creditcard.csv not found in project folder."})
    reset_training_state()
    set_feature_context(pd.read_csv(DATASET_PATH))
    fraud  = int(df_global["Class"].sum())
    legit  = int((df_global["Class"] == 0).sum())
    null_c = int(df_global.isnull().sum().sum())
    return jsonify({
        "success": True,
        "rows": len(df_global),
        "columns": len(df_global.columns),
        "fraud": fraud,
        "legit": legit,
        "fraud_pct": round(fraud / len(df_global) * 100, 3),
        "nulls": null_c,
        "amount_mean": round(float(df_global["Amount"].mean()), 2),
        "amount_max":  round(float(df_global["Amount"].max()), 2),
    })

@app.route("/api/clean_data", methods=["POST"])
def clean_data():
    if df_global is None:
        return jsonify({"success": False, "error": "Load dataset first."})

    df_cleaned = df_global.drop_duplicates().copy()
    removed_duplicates = len(df_global) - len(df_cleaned)

    # Preserve fraud rows and only trim extreme Amount outliers from legitimate data.
    fraud_rows = df_cleaned[df_cleaned["Class"] == 1]
    legit_rows = df_cleaned[df_cleaned["Class"] == 0]

    amount_threshold = legit_rows["Amount"].quantile(0.99)
    legit_rows = legit_rows[legit_rows["Amount"] <= amount_threshold]

    df_cleaned = pd.concat([legit_rows, fraud_rows], ignore_index=True)
    df_cleaned = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save cleaned dataset
    cleaned_path = os.path.join(os.path.dirname(__file__), "creditcard clean.csv")
    df_cleaned.to_csv(cleaned_path, index=False)

    # Calculate row counts before and after cleaning
    original_rows = len(df_global)
    cleaned_rows  = len(df_cleaned)
    rows_removed  = original_rows - cleaned_rows

    reset_training_state()
    set_feature_context(df_cleaned)

    return jsonify({
        "success": True,
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "rows_removed": rows_removed,
        "cleaned_path": cleaned_path,
        "message": (
            "Cleaned dataset saved as creditcard clean.csv. "
            f"{rows_removed} rows removed ({removed_duplicates} duplicates and "
            f"{rows_removed - removed_duplicates} extreme legitimate Amount outliers)."
        )
    })

@app.route("/api/train", methods=["POST"])
def train():
    global trained_models, scaler_global, pca_global, results_global, df_global, feature_means_global
    if df_global is None:
        return jsonify({"success": False, "error": "Load dataset first."})

    cfg        = request.json or {}
    use_smote  = cfg.get("smote", True)
    try:
        pca_comps = int(cfg.get("pca_components", 10))
        test_size = float(cfg.get("test_size", 0.2))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid training configuration."}), 400

    df = df_global.copy()
    X  = df[feature_cols].values
    y  = df["Class"].values
    feature_means_global = {c: float(df[c].mean()) for c in feature_cols}

    max_pca_components = min(len(feature_cols), len(df) - 1)
    if not 1 <= pca_comps <= max_pca_components:
        return jsonify({
            "success": False,
            "error": f"pca_components must be between 1 and {max_pca_components}."
        }), 400
    if not 0.05 <= test_size <= 0.5:
        return jsonify({"success": False, "error": "test_size must be between 0.05 and 0.5."}), 400
    if len(np.unique(y)) < 2:
        return jsonify({"success": False, "error": "Dataset must contain both fraud and legitimate rows."}), 400

    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(X)
    scaler_global = scaler

    pca          = PCA(n_components=pca_comps, random_state=42)
    X_pca        = pca.fit_transform(X_scaled)
    pca_global   = pca

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_pca, y, test_size=test_size, random_state=42, stratify=y)

    if use_smote:
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results_global = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        cm     = confusion_matrix(y_te, y_pred)
        fpr, tpr, _ = roc_curve(y_te, y_prob)

        results_global[name] = {
            "accuracy":         round(accuracy_score(y_te, y_pred)  * 100, 2),
            "precision":        round(precision_score(y_te, y_pred, zero_division=0) * 100, 2),
            "recall":           round(recall_score(y_te, y_pred, zero_division=0)    * 100, 2),
            "f1":               round(f1_score(y_te, y_pred, zero_division=0)        * 100, 2),
            "roc_auc":          round(roc_auc_score(y_te, y_prob) * 100, 2),
            "confusion_matrix": cm.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "tn": int(cm[0,0]), "fp": int(cm[0,1]),
            "fn": int(cm[1,0]), "tp": int(cm[1,1])
        }
        trained_models[name] = model

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler_global, SCALER_PATH)
    joblib.dump(pca_global, PCA_PATH)
    for name, model in trained_models.items():
        joblib.dump(model, MODEL_PATHS[name])
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "feature_cols": feature_cols,
            "feature_means": feature_means_global,
            "results": results_global,
            "pca_variance": [round(v * 100, 2) for v in pca.explained_variance_ratio_],
            "pca_total_variance": round(sum(pca.explained_variance_ratio_) * 100, 2),
        }, fh, indent=2)

    return jsonify({
        "success": True,
        "results": results_global,
        "pca_variance":       [round(v*100,2) for v in pca.explained_variance_ratio_],
        "pca_total_variance": round(sum(pca.explained_variance_ratio_)*100, 2)
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        ready, error = ensure_prediction_assets()
        if not ready:
            return jsonify({"success": False, "error": error})
        cfg        = request.json or {}
        features   = cfg.get("features", {})
        model_name = resolve_model_name(cfg.get("model", "Random Forest"))
        if model_name not in trained_models:
            return jsonify({"success": False, "error": "Model not found."})

        # Create full feature vector using provided values and averages for missing V features
        row = []
        for c in feature_cols:
            if c in features:
                row.append(float(features[c]))
            else:
                # Use saved averages for the anonymized features the UI does not expose.
                row.append(float(feature_means_global[c]))

        X   = np.array(row).reshape(1, -1)
        X_s = scaler_global.transform(X)
        X_p = pca_global.transform(X_s)
        pred = int(trained_models[model_name].predict(X_p)[0])
        prob = float(trained_models[model_name].predict_proba(X_p)[0][1])

        return jsonify({
            "success": True,
            "prediction":  pred,
            "probability": round(prob * 100, 2),
            "label":       "FRAUD" if pred == 1 else "LEGITIMATE"
        })
    except Exception as exc:
        app.logger.exception("Prediction request failed")
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500

@app.route("/api/state")
def app_state():
    metadata = load_training_metadata()
    ready, _ = ensure_prediction_assets()
    results = results_global or metadata.get("results") or {}
    return jsonify({
        "success": True,
        "dataset_loaded": df_global is not None,
        "trained": bool(ready and results),
        "available_models": list(trained_models.keys()),
        "results": results,
        "pca_variance": metadata.get("pca_variance", []),
        "pca_total_variance": metadata.get("pca_total_variance"),
    })

# ── Python Charts ────────────────────────────────────────
@app.route("/api/charts/distribution")
def chart_dist():
    if df_global is None: return jsonify({"error": "Load data first"})
    fig, ax = dark_fig(7, 4)
    counts = df_global["Class"].value_counts()
    bars   = ax.bar(["Legitimate","Fraud"], counts.values,
                    color=["#238636","#f85149"], width=0.5, edgecolor="#30363d")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+500,
                f"{v:,}", ha="center", color="#e6edf3", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14, pad=12)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

@app.route("/api/charts/amount")
def chart_amount():
    if df_global is None: return jsonify({"error": "Load data first"})
    fig, ax = dark_fig(8, 4)
    fraud = df_global[df_global["Class"]==1]["Amount"]
    legit = df_global[df_global["Class"]==0]["Amount"]
    ax.hist(legit[legit<2000], bins=60, alpha=0.7, color="#238636", label="Legitimate", edgecolor="none")
    ax.hist(fraud[fraud<2000], bins=60, alpha=0.9, color="#f85149", label="Fraud",      edgecolor="none")
    ax.set_title("Transaction Amount Distribution (< $2000)", fontsize=13, pad=12)
    ax.set_xlabel("Amount ($)"); ax.set_ylabel("Frequency")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

@app.route("/api/charts/correlation")
def chart_corr():
    if df_global is None: return jsonify({"error": "Load data first"})
    sample = df_global.sample(min(5000, len(df_global)), random_state=42)
    cols   = ["Time","Amount","V1","V2","V3","V4","V5","Class"]
    corr   = sample[cols].corr()
    fig, ax = dark_fig(8, 6)
    sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                annot_kws={"size":8,"color":"#e6edf3"},
                linewidths=0.5, linecolor="#30363d", cbar_kws={"shrink":0.8})
    ax.set_title("Correlation Heatmap (Key Features)", fontsize=13, pad=12)
    ax.tick_params(colors="#8b949e", labelsize=9)
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

@app.route("/api/charts/pca")
def chart_pca():
    if pca_global is None: return jsonify({"error": "Train first"})
    var    = pca_global.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    fig, ax = dark_fig(8, 4)
    ax.bar(range(1, len(var)+1), var, color="#1f6feb", alpha=0.8, edgecolor="none")
    ax2 = ax.twinx()
    ax2.plot(range(1,len(cumvar)+1), cumvar, color="#e3b341", marker="o", markersize=4, lw=2)
    ax2.set_ylabel("Cumulative Variance (%)", color="#e3b341")
    ax2.tick_params(colors="#e3b341")
    ax2.set_facecolor("#161b22")
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
    ax.set_title("PCA Explained Variance", fontsize=13, pad=12)
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance (%)")
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

@app.route("/api/charts/roc")
def chart_roc():
    if not results_global: return jsonify({"error": "Train first"})
    fig, ax = dark_fig(7, 5)
    colors = {"Logistic Regression":"#1f6feb","Random Forest":"#3fb950"}
    for name, r in results_global.items():
        ax.plot(r["fpr"], r["tpr"], color=colors.get(name,"#e3b341"),
                lw=2, label=f"{name} (AUC={r['roc_auc']}%)")
    ax.plot([0,1],[0,1],"--",color="#484f58",lw=1)
    ax.set_title("ROC Curve Comparison", fontsize=13, pad=12)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

@app.route("/api/charts/confusion/<path:model_name>")
def chart_confusion(model_name):
    if model_name not in results_global: return jsonify({"error": "Train first"})
    cm  = np.array(results_global[model_name]["confusion_matrix"])
    fig, ax = dark_fig(5, 4)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap=sns.color_palette(["#0d1117","#1f6feb"], as_cmap=True),
                linewidths=1, linecolor="#30363d",
                annot_kws={"size":16,"color":"#e6edf3"},
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"], cbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

@app.route("/api/charts/importance")
def chart_importance():
    if "Random Forest" not in trained_models: return jsonify({"error": "Train first"})
    imp    = trained_models["Random Forest"].feature_importances_
    labels = [f"PC{i+1}" for i in range(len(imp))]
    idx    = np.argsort(imp)[::-1]
    fig, ax = dark_fig(8, 4)
    ax.bar([labels[i] for i in idx], imp[idx], color="#1f6feb", alpha=0.85, edgecolor="none")
    ax.set_title("Random Forest — PCA Feature Importances", fontsize=13, pad=12)
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Importance")
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

@app.route("/api/charts/time_fraud")
def chart_time():
    if df_global is None: return jsonify({"error": "Load data first"})
    fig, ax = dark_fig(8, 4)
    fraud = df_global[df_global["Class"]==1]["Time"] / 3600
    legit = df_global[df_global["Class"]==0]["Time"] / 3600
    ax.hist(legit, bins=48, alpha=0.6, color="#238636", label="Legitimate", edgecolor="none", density=True)
    ax.hist(fraud, bins=48, alpha=0.9, color="#f85149", label="Fraud",      edgecolor="none", density=True)
    ax.set_title("Transaction Time Distribution (Hours)", fontsize=13, pad=12)
    ax.set_xlabel("Hour"); ax.set_ylabel("Density")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    fig.tight_layout()
    return jsonify({"image": fig_to_b64(fig)})

# ── R Integration ────────────────────────────────────────
@app.route("/api/r/run", methods=["POST"])
def run_r():
    payload = request.get_json(silent=True) or {}
    script = payload.get("script", "eda")
    if script not in R_SCRIPTS:
        return jsonify({"success": False, "error": f"Unsupported R script '{script}'."}), 400

    r_file = os.path.join(os.path.dirname(__file__), "r_scripts", R_SCRIPTS[script])
    if not os.path.exists(r_file):
        return jsonify({"success": False, "error": f"R script '{script}' not found."})

    out_dir = os.path.join(os.path.dirname(__file__), "r_output")
    before_mtime = {}
    if os.path.exists(out_dir):
        for name in os.listdir(out_dir):
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                before_mtime[name] = os.path.getmtime(path)

    try:
        result = run_rscript([r_file, DATASET_PATH], timeout=R_TIMEOUT_SECONDS)
        output = (result.stdout or "") + (result.stderr or "")

        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"R script failed with exit code {result.returncode}.",
                "output": output
            }), 500

        images = {}
        generated_files = []
        if os.path.exists(out_dir):
            for f in os.listdir(out_dir):
                path = os.path.join(out_dir, f)
                if not os.path.isfile(path):
                    continue
                modified = os.path.getmtime(path)
                if f.endswith(".png"):
                    with open(path, "rb") as fp:
                        images[f] = base64.b64encode(fp.read()).decode("utf-8")
                if before_mtime.get(f) != modified:
                    generated_files.append(f)

        report_path = os.path.join(out_dir, "fraud_report.html")
        report_url = url_for("download_r_report") if os.path.exists(report_path) else None

        return jsonify({
            "success": True,
            "output": output,
            "images": images,
            "generated_files": sorted(generated_files),
            "report_url": report_url
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": f"R script timed out ({R_TIMEOUT_SECONDS}s)."})
    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error": "Rscript not found. Install R from https://www.r-project.org/ or set RSCRIPT_PATH."
        }), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/r/check")
def check_r():
    try:
        rscript = find_rscript()
        if not rscript:
            return jsonify({"available": False})
        result = subprocess.run([rscript, "--version"], capture_output=True, text=True, timeout=5)
        version = (result.stdout + result.stderr).strip()
        return jsonify({"available": True, "version": version, "path": rscript})
    except:
        return jsonify({"available": False})

@app.route("/api/r/report")
def download_r_report():
    report_path = os.path.join(os.path.dirname(__file__), "r_output", "fraud_report.html")
    if not os.path.exists(report_path):
        return jsonify({"success": False, "error": "R report not found. Generate it first."}), 404
    return send_file(report_path, mimetype="text/html", as_attachment=True, download_name="fraud_report.html")

# ── Report ───────────────────────────────────────────────
@app.route("/api/report")
def report():
    if not results_global:
        return jsonify({"error": "Train first"})
    lines = [
        "=" * 55,
        "   CREDIT CARD FRAUD DETECTION - REPORT",
        "=" * 55, "",
        f"Dataset : creditcard.csv",
        f"Rows    : {len(df_global):,}",
        f"Fraud   : {int(df_global['Class'].sum()):,}  ({round(df_global['Class'].mean()*100,3)}%)",
        f"Legit   : {int((df_global['Class']==0).sum()):,}",
        ""
    ]
    for name, r in results_global.items():
        lines += [
            "-" * 40,
            f"Model     : {name}",
            f"Accuracy  : {r['accuracy']}%",
            f"Precision : {r['precision']}%",
            f"Recall    : {r['recall']}%",
            f"F1-Score  : {r['f1']}%",
            f"ROC-AUC   : {r['roc_auc']}%",
            f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}",
            ""
        ]
    buf = io.BytesIO("\n".join(lines).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="text/plain",
                     as_attachment=True, download_name="fraud_detection_report.txt")

if __name__ == "__main__":
    host = os.environ.get("FRAUDGUARD_HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", os.environ.get("FRAUDGUARD_PORT", DEFAULT_PORT)))
    schedule_dashboard_open(host, port)
    app.run(debug=False, host=host, port=port)
