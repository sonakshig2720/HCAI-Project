import os
import io
import uuid
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpRequest

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

# Optional libs
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None  # type: ignore


# -----------------------------
# Helpers (session + casting)
# -----------------------------
def _store_df_in_session(request: HttpRequest, df: pd.DataFrame) -> None:
    request.session["df"] = df.to_json(orient="split")
    request.session["columns"] = df.columns.tolist()


def _load_df_from_session(request: HttpRequest) -> Optional[pd.DataFrame]:
    raw = request.session.get("df")
    if not raw:
        return None
    try:
        return pd.read_json(raw, orient="split")
    except Exception:
        return None


def _smart_cast(v: Optional[str]):
    if v is None:
        return None
    s = v.strip()
    if s == "":
        return None
    low = s.lower()
    if low in {"none", "null", "na", "n/a"}:
        return None
    if low in {"true", "on"}:
        return True
    if low in {"false", "off"}:
        return False
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        pass
    return s


def _extract_hyperparams(post: Dict[str, str]) -> Dict[str, Any]:
    hp: Dict[str, Any] = {}
    for k, v in post.items():
        if k.startswith("hp_"):
            name = k[3:]
            val = _smart_cast(v)
            if val is not None:
                hp[name] = val
    return hp


# -----------------------------
# Validation (incompatibility)
# -----------------------------
def _check_logistic_compatibility(hp: Dict[str, Any]) -> (bool, str):
    """
    Validate LogisticRegression hyperparameters.
    Returns (ok, message). If not ok, message explains the incompatibility.
    """
    penalty = str(hp.get("penalty", "l2")).lower()
    solver = str(hp.get("solver", "lbfgs")).lower()

    # Allowed combos from sklearn docs:
    # - lbfgs: l2 only
    # - liblinear: l1 or l2
    # - saga: l1, l2, elasticnet (elasticnet requires l1_ratio)
    allowed = {
        "lbfgs": {"l2"},
        "liblinear": {"l1", "l2"},
        "saga": {"l1", "l2", "elasticnet"},
    }
    if solver not in allowed:
        return False, f"solver='{solver}' is not supported. Use lbfgs, liblinear or saga."
    if penalty not in allowed[solver]:
        return False, f"penalty='{penalty}' is incompatible with solver='{solver}'."

    if penalty == "elasticnet":
        if solver != "saga":
            return False, "elasticnet penalty requires solver='saga'."
        if "l1_ratio" not in hp:
            return False, "elasticnet requires 'l1_ratio' in [0,1]."

    # C must be positive if provided
    if "C" in hp:
        try:
            c = float(hp["C"])
            if c <= 0:
                return False, "C must be > 0."
        except Exception:
            return False, "C must be a number."

    # l1_ratio bounds if provided
    if "l1_ratio" in hp:
        try:
            r = float(hp["l1_ratio"])
            if not (0.0 <= r <= 1.0):
                return False, "l1_ratio must be between 0 and 1."
        except Exception:
            return False, "l1_ratio must be a number."

    return True, ""


# -----------------------------
# View
# -----------------------------
@csrf_exempt
def index(request: HttpRequest):
    # Reset: start fresh (no persisted values)
    if request.method == "POST" and request.POST.get("reset") == "1":
        for k in ["df", "columns", "plot_url", "task", "target_column",
                  "selected_model_type", "selected_hp_json"]:
            if k in request.session:
                del request.session[k]

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
        "locked": False,  # lock UI after successful training
        # prefill hooks
        "selected_model_type": request.session.get("selected_model_type", ""),
        "selected_hp_json": request.session.get("selected_hp_json", "{}"),
    }

    # ---------- GET ----------
    if request.method == "GET":
        # Keep page clean (fresh) unless you specifically want persistence on GET.
        return render(request, "index.html", context)

    # ---------- POST ----------
    # 1) CSV UPLOAD
    if "csv_file" in request.FILES:
        try:
            df = pd.read_csv(request.FILES["csv_file"])
            _store_df_in_session(request, df)
            context["data_preview"] = df.head().to_html(classes="table dataframe")
            context["columns"] = df.columns.tolist()
            context["show_visualize"] = True
        except Exception as e:
            context["error_msg"] = f"Failed to read CSV: {e}"
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

        if not all(c in df.columns for c in [x_feature, target_column]):
            context["error_msg"] = "Selected features/target not found in the dataset."
            context["show_visualize"] = True
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
            path = os.path.join(settings.MEDIA_ROOT, filename)
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
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

        return render(request, "index.html", context)

    # 3) TRAIN
    if "train" in request.POST:
        df = _load_df_from_session(request)
        if df is None:
            context["error_msg"] = "No dataset in session. Please upload a CSV again."
            return render(request, "index.html", context)

        context["columns"] = request.session.get("columns", [])
        context["plot_url"] = request.session.get("plot_url")

        target_column = request.POST.get("target_column") or request.session.get("target_column")
        task = request.POST.get("task") or request.session.get("task")
        model_type = request.POST.get("model_type")

        context["target_column"] = target_column
        context["task"] = task
        context["show_visualize"] = True
        context["show_train_form"] = True

        # Clean & prepare
        df = df.dropna()
        if target_column not in df.columns:
            context["error_msg"] = "Target column not found after cleaning. Recheck your selection."
            return render(request, "index.html", context)

        X = df.drop(columns=[target_column]).select_dtypes(include="number")
        y = df[target_column]
        if X.shape[1] == 0:
            context["error_msg"] = "No numeric features available for modeling."
            return render(request, "index.html", context)

        # Splits
        split_mode = request.POST.get("split_mode", "tt")
        test_size = float(_smart_cast(request.POST.get("test_size")) or 0.2)
        val_size = float(_smart_cast(request.POST.get("val_size")) or 0.0)
        stratify = True if request.POST.get("stratify") is not None else False
        random_state = int(_smart_cast(request.POST.get("random_state")) or 42)

        try:
            if split_mode == "tt":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if (task == "classification" and stratify) else None,
                    shuffle=True
                )
                X_val = y_val = None
            else:
                X_tmp, X_test, y_tmp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if (task == "classification" and stratify) else None,
                    shuffle=True
                )
                rel_val = val_size / max(1e-9, (1.0 - test_size))
                X_train, X_val, y_train, y_val = train_test_split(
                    X_tmp, y_tmp, test_size=rel_val, random_state=random_state,
                    stratify=y_tmp if (task == "classification" and stratify) else None,
                    shuffle=True
                )
        except Exception as e:
            context["error_msg"] = f"Split failed: {e}"
            return render(request, "index.html", context)

        # Hyperparams
        hp = _extract_hyperparams(request.POST)

        # Build model with compatibility checks
        model = None
        model_label = ""

        try:
            if task == "classification":
                if model_type == "logistic":
                    ok, msg = _check_logistic_compatibility(hp)
                    if not ok:
                        context["error_msg"] = f"Incompatible hyperparameters: {msg}"
                        return render(request, "index.html", context)
                    model = LogisticRegression(**hp)
                    model_label = "Logistic Regression"

                elif model_type == "dt":
                    model = DecisionTreeClassifier(**hp)
                    model_label = "Decision Tree Classifier"

                elif model_type == "rf":
                    model = RandomForestClassifier(random_state=random_state, **hp)
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
                    context["error_msg"] = f"Unknown classification model '{model_type}'."
                    return render(request, "index.html", context)

            else:  # regression
                if model_type == "lasso":
                    model = Lasso(**hp)
                    model_label = "Lasso Regression"

                elif model_type == "ridge":
                    model = Ridge(**hp)
                    model_label = "Ridge Regression"

                elif model_type == "rf":
                    model = RandomForestRegressor(random_state=random_state, **hp)
                    model_label = "Random Forest Regressor"

                elif model_type == "xgb":
                    if XGBRegressor is None:
                        context["error_msg"] = "XGBoost is not installed in this environment."
                        return render(request, "index.html", context)
                    model = XGBRegressor(**hp)
                    model_label = "XGBoost Regressor"

                elif model_type == "lgbm":
                    if LGBMRegressor is None:
                        context["error_msg"] = "LightGBM is not installed in this environment."
                        return render(request, "index.html", context)
                    model = LGBMRegressor(**hp)
                    model_label = "LightGBM Regressor"

                elif model_type == "svm":
                    model = SVR(**hp)
                    model_label = "Support Vector Regressor"

                elif model_type == "knn":
                    model = KNeighborsRegressor(**hp)
                    model_label = "K-Nearest Neighbors Regressor"

                else:
                    context["error_msg"] = f"Unknown regression model '{model_type}'."
                    return render(request, "index.html", context)

            # --- Fit & evaluate
            model.fit(X_train, y_train)

            if task == "classification":
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                class_report = []
                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        class_report.append({
                            "label": label,
                            "precision": f"{metrics.get('precision', 0):.2f}",
                            "recall": f"{metrics.get('recall', 0):.2f}",
                            "f1_score": f"{metrics.get('f1-score', 0):.2f}",
                            "support": metrics.get("support", 0),
                        })
                macro = report.get("macro avg", {})
                evaluation_result = {
                    "accuracy": f"{acc:.2f}",
                    "precision": f"{macro.get('precision', 0):.2f}",
                    "recall": f"{macro.get('recall', 0):.2f}",
                    "f1_score": f"{macro.get('f1-score', 0):.2f}",
                }
                context["conf_matrix"] = confusion_matrix(y_test, y_pred).tolist()
                context["class_report"] = class_report
                context["evaluation_result"] = evaluation_result
            else:
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = float(np.sqrt(mse))
                r2 = r2_score(y_test, y_pred)
                evaluation_result = {
                    "mae": f"{mae:.2f}",
                    "mse": f"{mse:.2f}",
                    "rmse": f"{rmse:.2f}",
                    "r2": f"{r2:.2f}",
                }
                context["evaluation_result"] = evaluation_result

            context["model_label"] = model_label
            context["data_preview"] = df.head().to_html(classes="table dataframe")
            context["locked"] = True  # lock UI after successful run

            # >>> Preserve chosen model + hyperparams for prefill <<<
            request.session["selected_model_type"] = model_type
            request.session["selected_hp_json"] = json.dumps(hp)
            context["selected_model_type"] = model_type
            context["selected_hp_json"] = json.dumps(hp)

        except Exception as e:
            context["error_msg"] = f"Training failed: {type(e).__name__}: {e}"

        return render(request, "index.html", context)

    # Fallback
    return render(request, "index.html", context)