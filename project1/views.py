import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, uuid
import numpy as np

from django.shortcuts import render
from django.conf import settings
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


def index(request):
    data_preview = None
    plot_url = None
    columns = []
    evaluation_result = None
    conf_matrix = None
    class_report = None
    task = None
    show_visualize = False
    show_train_form = False
    target_column = None
    x_feature = None
    y_feature = None
    model_label = ""


    if request.method == "POST":
        if "csv_file" in request.FILES:
            df = pd.read_csv(request.FILES["csv_file"])
            request.session["df"] = df.to_json()
            request.session["columns"] = df.columns.tolist()
            data_preview = df.head().to_html(classes="table table-bordered")
            columns = df.columns.tolist()
            show_visualize = True

        elif "visualize" in request.POST:
            df = pd.read_json(request.session.get("df"))
            columns = request.session.get("columns", [])
            data_preview = df.head().to_html(classes="table table-bordered")

            target_column = request.POST.get("target_column")
            task = request.POST.get("task")
            x_feature = request.POST.get("x_feature")
            y_feature = request.POST.get("y_feature")
            target_column = request.POST.get("target_column")
            task = request.POST.get("task")

            # Default y_feature to target_column if regression and none is selected
            if task == "regression" and not y_feature:
                y_feature = target_column

            request.session["target_column"] = target_column
            request.session["task"] = task

            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            if all(col in df.columns for col in [x_feature, target_column]):

                try:
                    plt.figure(figsize=(8, 6))

                    if task == "classification":
                        # Classification: color by target
                        if y_feature and y_feature in df.columns:
                            sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=target_column)
                            plt.title(f"{y_feature} vs {x_feature} (colored by {target_column})")
                        else:
                            raise ValueError("Y feature required for classification plot")

                    elif task == "regression":
                        # Regression: either scatter x vs target or x vs y
                        if y_feature and y_feature != target_column:
                            sns.scatterplot(data=df, x=x_feature, y=y_feature)
                            plt.title(f"{y_feature} vs {x_feature}")
                        else:
                            sns.scatterplot(data=df, x=x_feature, y=target_column)
                            plt.title(f"{target_column} vs {x_feature}")

                    filename = f"{uuid.uuid4()}.png"
                    path = os.path.join(settings.MEDIA_ROOT, filename)
                    plt.savefig(path)
                    plt.close()

                    plot_url = settings.MEDIA_URL + filename
                    request.session["plot_url"] = plot_url
                    show_train_form = True

                except Exception as e:
                    print("Plot generation failed:", e)

        elif "train" in request.POST:
            df = pd.read_json(request.session.get("df"))
            columns = request.session.get("columns", [])
            plot_url = request.session.get("plot_url")
            target_column = request.POST.get("target_column")
            model_type = request.POST.get("model_type")
            task = request.POST.get("task")
            df = df.dropna()

            X = df.drop(columns=[target_column])
            y = df[target_column]
            X = X.select_dtypes(include="number")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Initialize model = None by default
            model = None

            if task == "classification":
                if model_type == "logistic":
                    model = LogisticRegression(max_iter=1000)
                    model_label = "Logistic Regression Classifier"
                elif model_type == "dt":
                    model = DecisionTreeClassifier()
                    model_label = "Decision Tree Classifier"
                elif model_type == "rf":
                    model = RandomForestClassifier()
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
                    model = RandomForestRegressor()
                    model_label = "Random Forest Regressor"
                elif model_type == "svm":
                    model = SVR()
                    model_label = "Support Vector Regressor"
                elif model_type == "knn":
                    model = KNeighborsRegressor()
                    model_label = "K-Nearest Neighbors Regressor"



            if model:
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
                                'label': label,
                                'precision': f"{metrics.get('precision', 0):.2f}",
                                'recall': f"{metrics.get('recall', 0):.2f}",
                                'f1_score': f"{metrics.get('f1-score', 0):.2f}",
                                'support': metrics.get('support', 0),
                            })

                    macro = raw_report.get("macro avg", {})
                    evaluation_result = {
                        "accuracy": f"{acc:.2f}",
                        "precision": f"{macro.get('precision', 0):.2f}",
                        "recall": f"{macro.get('recall', 0):.2f}",
                        "f1_score": f"{macro.get('f1-score', 0):.2f}",
                    }
                    
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

                data_preview = df.head().to_html(classes="table")

        show_visualize = True  
        show_train_form = True  

    return render(request, "index.html", {
        "data_preview": data_preview,
        "plot_url": plot_url,
        "columns": columns,
        "evaluation_result": evaluation_result,
        "conf_matrix": conf_matrix,
        "class_report": class_report,
        "task": task,
        "show_visualize": show_visualize,
        "show_train_form": show_train_form,
        "x_feature": x_feature,
        "y_feature": y_feature,
        "target_column": target_column,
        "task": task,
        "model_label": model_label,

    })
