import os
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from django.conf import settings
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
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


def _load_df_from_session(request):
    """Load the DataFrame from session with a stable orient."""
    raw = request.session.get("df")
    if not raw:
        return None
    try:
        return pd.read_json(raw, orient="split")
    except Exception:
        # fall back: session may be stale or malformed
        return None


def index(request):
    # Defaults for template
    context = {
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
    }

    if request.method == "POST":
        # --- 1) CSV UPLOAD ---
        if "csv_file" in request.FILES:
            try:
                df = pd.read_csv(request.FILES["csv_file"])
            except Exception as e:
                context["error_msg"] = f"Failed to read CSV: {e}"
                return render(request, "index.html", context)

            # Persist to session with stable orient
            request.session["df"] = df.to_json(orient="split")
            request.session["columns"] = df.columns.tolist()

            context["data_preview"] = df.head().to_html(classes="table dataframe")
            context["columns"] = df.columns.tolist()
            context["show_visualize"] = True

        # --- 2) VISUALIZE ---
        elif "visualize" in request.POST:
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

            # For regression, if no y_feature chosen, use target for y
            if task == "regression" and not y_feature:
                y_feature = target_column

            # Store task/target for later
            request.session["target_column"] = target_column
            request.session["task"] = task

            context["target_column"] = target_column
            context["task"] = task
            context["x_feature"] = x_feature
            context["y_feature"] = y_feature

            # Basic sanity checks
            if not all(col in df.columns for col in [x_feature, target_column]):
                context["error_msg"] = "Selected features/target not found in the dataset."
                context["show_visualize"] = True
                return render(request, "index.html", context)

            try:
                plt.figure(figsize=(8, 6))
                if task == "classification":
                    # Require y_feature for classification plot
                    if y_feature and y_feature in df.columns:
                        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=target_column)
                        plt.title(f"{y_feature} vs {x_feature} (colored by {target_column})")
                    else:
                        raise ValueError("Y feature is required for a classification scatter plot.")
                elif task == "regression":
                    # If y_feature provided and different, plot x vs y; else x vs target
                    if y_feature and y_feature != target_column:
                        sns.scatterplot(data=df, x=x_feature, y=y_feature)
                        plt.title(f"{y_feature} vs {x_feature}")
                    else:
                        sns.scatterplot(data=df, x=x_feature, y=target_column)
                        plt.title(f"{target_column} vs {x_feature}")

                filename = f"{uuid.uuid4()}.png"
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

        # --- 3) TRAIN ---
        elif "train" in request.POST:
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

            # Clean and split
            df = df.dropna()
            if target_column not in df.columns:
                context["error_msg"] = "Target column not found after cleaning. Recheck your selection."
                return render(request, "index.html", context)

            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Use numeric features only
            X = X.select_dtypes(include="number")
            if X.shape[1] == 0:
                context["error_msg"] = "No numeric features available for modeling."
                return render(request, "index.html", context)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Choose model
            model = None
            model_label = ""

            if task == "classification":
                if model_type == "logistic":
                    model = LogisticRegression(max_iter=1000)
                    model_label = "Logistic Regression Classifier"
                elif model_type == "dt":
                    model = DecisionTreeClassifier()
                    model_label = "Decision Tree Classifier"
                elif model_type == "rf":
                    model = RandomForestClassifier(random_state=42)
                    model_label = "Random Forest Classifier"
                elif model_type == "knn":
                    model = KNeighborsClassifier()
                    model_label = "K-Nearest Neighbors Classifier"
                elif model_type == "svm":
                    model = SVC()
                    model_label = "Support Vector Machine Classifier"
                elif model_type == "nb":
                    model = GaussianNB()
                    model_label = "Naive Bayes Classifier"
            else:
                if model_type == "lasso":
                    model = Lasso()
                    model_label = "Lasso Regression"
                elif model_type == "ridge":
                    model = Ridge()
                    model_label = "Ridge Regression"
                elif model_type == "xgb" and XGBRegressor:
                    model = XGBRegressor()
                    model_label = "XGBoost Regressor"
                elif model_type == "lgbm" and LGBMRegressor:
                    model = LGBMRegressor()
                    model_label = "LightGBM Regressor"
                elif model_type == "rf":
                    model = RandomForestRegressor(random_state=42)
                    model_label = "Random Forest Regressor"
                elif model_type == "svm":
                    model = SVR()
                    model_label = "Support Vector Regressor"
                elif model_type == "knn":
                    model = KNeighborsRegressor()
                    model_label = "K-Nearest Neighbors Regressor"

            # IMPORTANT: do not use truthiness on sklearn models
            if model is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if task == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
                    raw_report = classification_report(y_test, y_pred, output_dict=True)

                    class_report = []
                    for label, metrics in raw_report.items():
                        if isinstance(metrics, dict):
                            class_report.append({
                                "label": label,
                                "precision": f"{metrics.get('precision', 0):.2f}",
                                "recall": f"{metrics.get('recall', 0):.2f}",
                                "f1_score": f"{metrics.get('f1-score', 0):.2f}",
                                "support": metrics.get("support", 0),
                            })

                    macro = raw_report.get("macro avg", {})
                    evaluation_result = {
                        "accuracy": f"{acc:.2f}",
                        "precision": f"{macro.get('precision', 0):.2f}",
                        "recall": f"{macro.get('recall', 0):.2f}",
                        "f1_score": f"{macro.get('f1-score', 0):.2f}",
                    }
                    context["conf_matrix"] = conf_matrix
                    context["class_report"] = class_report
                    context["evaluation_result"] = evaluation_result
                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
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
                context["show_visualize"] = True
                context["show_train_form"] = True
            else:
                context["error_msg"] = "Please choose a valid model for the selected task."
                context["show_visualize"] = True

    # GET or render after POST
    if not context["columns"]:
        context["columns"] = request.session.get("columns", [])
    return render(request, "index.html", context)
