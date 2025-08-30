import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from django.conf import settings
from django.http import HttpRequest
from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

# Optional imports
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None


# -------- session helpers --------
RESET_KEYS = ["df", "columns", "plot_url", "target_column", "task"]

def _load_df_from_session(request: HttpRequest) -> Optional[pd.DataFrame]:
    raw = request.session.get("df")
    if not raw:
        return None
    try:
        return pd.read_json(raw, orient="split")
    except Exception:
        return None

def _smart_cast(value: Optional[str]):
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    low = v.lower()
    if low in {"none", "null", "na", "n/a"}:
        return None
    if low in {"on", "true", "false"}:
        return low in {"on", "true"}
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

def _extract_hyperparams(post_dict) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for k, v in post_dict.items():
        if not k.startswith("hp_"):
            continue
        name = k[3:]
        casted = _smart_cast(v)
        if casted is not None:
            params[name] = casted
    return params


@dataclass
class EvalResult:
    ok: bool
    message: str
    metrics: Dict[str, Any]
    conf_matrix: Optional[list]
    class_report: Optional[list]


def index(request: HttpRequest):
    # default context (fresh page shows only upload section)
    context: Dict[str, Any] = {
        "data_preview": None,
        "plot_url": None,
        "columns": [],
        "evaluation_result": None,
        "conf_matrix": None,
        "class_report": None,
        "task": None,
        "show_visualize": False,
        "show_train_form": False,
        "x_feature": None,
        "y_feature": None,
        "target_column": None,
        "model_label": "",
        "error_msg": None,
        "locked": False,
    }

    # --------------------------
    # GET => always reset to initial state
    # --------------------------
    if request.method == "GET":
        for k in RESET_KEYS:
            request.session.pop(k, None)
        # render initial page (only upload visible)
        return render(request, "index.html", context)

    # --------------------------
    # POST handling
    # --------------------------

    # optional manual reset button
    if "reset" in request.POST:
        for k in RESET_KEYS:
            request.session.pop(k, None)
        return render(request, "index.html", context)

    # 1) UPLOAD
    if "csv_file" in request.FILES:
        try:
            df = pd.read_csv(request.FILES["csv_file"])
        except Exception as e:
            context["error_msg"] = f"Failed to read CSV: {e}"
            return render(request, "index.html", context)

        request.session["df"] = df.to_json(orient="split")
        request.session["columns"] = df.columns.tolist()

        context["data_preview"] = df.head().to_html(classes="table dataframe")
        context["columns"] = df.columns.tolist()
        context["show_visualize"] = True
        context["show_train_form"] = True
        context["locked"] = False
        return render(request, "index.html", context)

    # 2) VISUALIZE
    if "visualize" in request.POST:
        df = _load_df_from_session(request)
        if df is None:
            context["error_msg"] = "No dataset in session. Please upload a CSV again."
            return render(request, "index.html", context)

        context["columns"] = request.session.get("columns", [])
        context["data_preview"] = df.head().to_html(classes="table dataframe")

        target_column = request.POST.get("target_column")
        task = request.POST.get("task")
        x_feature = request.POST.get("x_feature")
        y_feature = request.POST.get("y_feature")

        if task == "regression" and not y_feature:
            y_feature = target_column

        request.session["target_column"] = target_column
        request.session["task"] = task

        context["target_column"] = target_column
        context["task"] = task
        context["x_feature"] = x_feature
        context["y_feature"] = y_feature

        if not all(col in df.columns for col in [x_feature, target_column]):
            context["error_msg"] = "Selected features/target not found in the dataset."
            context["show_visualize"] = True
            context["show_train_form"] = True
            return render(request, "index.html", context)

        try:
            plt.figure(figsize=(8, 6))
            if task == "classification":
                if y_feature and y_feature in df.columns:
                    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=target_column)
                    plt.title(f"{y_feature} vs {x_feature} (colored by {target_column})")
                else:
                    raise ValueError("Y feature is required for a classification scatter plot.")
            elif task == "regression":
                if y_feature and y_feature != target_column:
                    sns.scatterplot(data=df, x=x_feature, y=y_feature)
                    plt.title(f"{y_feature} vs {x_feature}")
                else:
                    sns.scatterplot(data=df, x=x_feature, y=target_column)
                    plt.title(f"{target_column} vs {x_feature}")

            filename = f"{uuid.uuid4()}.png"
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            path = os.path.join(settings.MEDIA_ROOT, filename)
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            plot_url = settings.MEDIA_URL + filename
            request.session["plot_url"] = plot_url
            context["plot_url"] = plot_url
            context["show_visualize"] = True
            context["show_train_form"] = True
        except Exception as e:
            context["error_msg"] = f"Plot generation failed: {e}"
            context["show_visualize"] = True
            context["show_train_form"] = True

        return render(request, "index.html", context)

    # 3) TRAIN
    if "train" in request.POST:
        df = _load_df_from_session(request)
        if df is None:
            context["error_msg"] = "No dataset in session. Please upload a CSV again."
            return render(request, "index.html", context)

        context["columns"] = request.session.get("columns", [])
        context["plot_url"] = request.session.get("plot_url")
        context["show_visualize"] = True
        context["show_train_form"] = True

        target_column = request.POST.get("target_column") or request.session.get("target_column")
        task = request.POST.get("task") or request.session.get("task")
        model_type = request.POST.get("model_type")

        context["target_column"] = target_column
        context["task"] = task

        df = df.dropna()
        if not target_column or target_column not in df.columns:
            context["error_msg"] = "Target column not found after cleaning. Recheck your selection."
            return render(request, "index.html", context)

        X = df.drop(columns=[target_column]).select_dtypes(include="number")
        y = df[target_column]
        if X.shape[1] == 0:
            context["error_msg"] = "No numeric features available for modeling."
            return render(request, "index.html", context)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        hp = _extract_hyperparams(request.POST)
        model = None
        model_label = ""

        if task == "classification":
            if model_type == "logistic":
                model = LogisticRegression(**({"max_iter": 1000} | hp))
                model_label = "Logistic Regression Classifier"
            elif model_type == "dt":
                model = DecisionTreeClassifier(**hp)
                model_label = "Decision Tree Classifier"
            elif model_type == "rf":
                model = RandomForestClassifier(random_state=42, **hp)
                model_label = "Random Forest Classifier"
            elif model_type == "knn":
                model = KNeighborsClassifier(**hp)
                model_label = "K-Nearest Neighbors Classifier"
            elif model_type == "svm":
                model = SVC(**hp)
                model_label = "Support Vector Machine Classifier"
            elif model_type == "nb":
                model = GaussianNB(**hp)
                model_label = "Naive Bayes Classifier"
        else:
            if model_type == "lasso":
                model = Lasso(**hp)
                model_label = "Lasso Regression"
            elif model_type == "ridge":
                model = Ridge(**hp)
                model_label = "Ridge Regression"
            elif model_type == "xgb" and XGBRegressor:
                model = XGBRegressor(**hp)
                model_label = "XGBoost Regressor"
            elif model_type == "lgbm" and LGBMRegressor:
                model = LGBMRegressor(**hp)
                model_label = "LightGBM Regressor"
            elif model_type == "rf":
                model = RandomForestRegressor(random_state=42, **hp)
                model_label = "Random Forest Regressor"
            elif model_type == "svm":
                model = SVR(**hp)
                model_label = "Support Vector Regressor"
            elif model_type == "knn":
                model = KNeighborsRegressor(**hp)
                model_label = "K-Nearest Neighbors Regressor"

        if model is None:
            context["error_msg"] = "Please choose a valid model for the selected task."
            return render(request, "index.html", context)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "classification":
                acc = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred).tolist()
                raw_report = classification_report(y_test, y_pred, output_dict=True)
                macro = raw_report.get("macro avg", {})
                context["evaluation_result"] = {
                    "accuracy": f"{acc:.2f}",
                    "precision": f"{macro.get('precision', 0):.2f}",
                    "recall": f"{macro.get('recall', 0):.2f}",
                    "f1_score": f"{macro.get('f1-score', 0):.2f}",
                }
                context["conf_matrix"] = conf_matrix
                context["class_report"] = [
                    {
                        "label": label,
                        "precision": f"{m.get('precision', 0):.2f}",
                        "recall": f"{m.get('recall', 0):.2f}",
                        "f1_score": f"{m.get('f1-score', 0):.2f}",
                        "support": m.get("support", 0),
                    }
                    for label, m in raw_report.items() if isinstance(m, dict)
                ]
            else:
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                context["evaluation_result"] = {
                    "mae": f"{mae:.2f}",
                    "mse": f"{mse:.2f}",
                    "rmse": f"{rmse:.2f}",
                    "r2": f"{r2:.2f}",
                }

            context["model_label"] = model_label
            context["data_preview"] = df.head().to_html(classes="table dataframe")
            context["locked"] = True  # freeze controls after success

        except Exception as e:
            context["error_msg"] = f"Training failed: {e}"
            context["locked"] = False

        return render(request, "index.html", context)

    # Fallback POST
    context["error_msg"] = "Unknown action."
    return render(request, "index.html", context)
