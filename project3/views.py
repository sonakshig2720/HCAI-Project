import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from palmerpenguins import load_penguins
from sklearn.metrics import pairwise_distances


# Load and preprocess data
def load_data():
    df = load_penguins()
    df = df.dropna()
    df = pd.get_dummies(df, columns=['sex', 'island'], drop_first=True)
    X = df.drop('species', axis=1)
    y = df['species']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Convert matplotlib figure to base64 string
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# Index view for project navigation
def index(request):
    links = [
        {"name": "Decision Tree", "url": "project3:decision_tree"},
        {"name": "Logistic Regression", "url": "project3:logistic_regression"},
        {"name": "Counterfactual Explanation", "url": "project3:counterfactual"},
    ]
    return render(request, 'project3/index.html', {'links': links})

# Decision Tree View
def decision_tree_view(request):
    X_train, X_test, y_train, y_test = load_data()
    accuracy = None
    leaves = None
    tree_image = None

    if request.method == 'POST':
        depth = int(request.POST.get('depth', 3))
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = round(100 * (y_pred == y_test).mean(), 2)
        leaves = model.get_n_leaves()

        # Create a fresh figure for every request
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_tree(
            model,
            feature_names=list(X_train.columns),
            class_names=list(model.classes_),
            filled=True,
            ax=ax
        )

        # Convert to base64 and close to avoid overlap
        tree_image = fig_to_base64(fig)
        plt.close(fig)

    return render(request, 'project3/decision_tree.html', {
        'accuracy': accuracy,
        'leaves': leaves,
        'tree_image': tree_image,
        'depth': depth if request.method == 'POST' else '',
    })




# Logistic Regression View
# Task 2: Sparse Tree with GOSDT
def sparse_tree_view(request):
    X_train, X_test, y_train, y_test = load_data()
    accuracy = None
    leaves = None
    tree_image = None
    depth = ''
    lambda_val = 1.0

    if request.method == 'POST':
        depth = int(request.POST.get('depth', 3))
        lambda_val = float(request.POST.get('lambda', 1.0))

        model = GOSDT(config={
            "regularization": lambda_val,
            "depth_budget": depth,
            "time_limit": 10
        })
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = round(100 * (y_pred == y_test).mean(), 2)
        leaves = len(model.model["rules"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.set_title("GOSDT: Sparse Decision Tree")
        ax.text(0.1, 0.5, f"{leaves} rules (leaves)", fontsize=12)
        tree_image = fig_to_base64(fig)
        plt.close(fig)

    return render(request, 'project3/sparse_tree.html', {
        'accuracy': accuracy,
        'leaves': leaves,
        'tree_image': tree_image,
        'depth': depth,
        'lambda': lambda_val,
    })

# Task 3: Logistic Regression with L1 Regularization (Sparsity)
def logistic_regression_view(request):
    X_train, X_test, y_train, y_test = load_data()
    accuracy = None
    features_used = None
    lambda_val = 0.1

    if request.method == 'POST':
        lambda_val = float(request.POST.get('lambda', 0.1))

    if lambda_val == 0:
        error = "Please enter a valid non-zero λ value."
    else:
        # Logistic Regression with L1 (sparsity) penalty
        model = LogisticRegression(penalty='l1', solver='saga', C=1/lambda_val, max_iter=10000)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = round(100 * (y_pred == y_test).mean(), 2)
        features_used = np.sum(np.any(model.coef_ != 0, axis=0))


    return render(request, 'project3/logistic_regression.html', {
        'accuracy': accuracy,
        'features_used': features_used,
        'lambda': lambda_val,
        'error': error if lambda_val == 0 else None,
    })



# Counterfactual Explanation View
# Task 4: Counterfactual Explanation
def mad_weights(X):
    return np.median(np.abs(X - np.median(X, axis=0)), axis=0)

def counterfactual_view(request):
    df = load_penguins().dropna()
    df_encoded = pd.get_dummies(df.drop(columns='species'), drop_first=True)
    X = df_encoded
    y = df['species'].astype('category').cat.codes
    class_names = df['species'].astype('category').cat.categories

    instance_options = list(enumerate(df.index))  # [(0, orig_index), ...]
    label_options = list(enumerate(class_names))

    instance_index = 0
    target_class = 0
    counterfactuals = []
    prediction = None
    original_instance = {}

    k = 3 
    if request.method == 'POST':
        instance_index = int(request.POST.get('instance', 0))
        target_class = int(request.POST.get('target', 0))
        k = int(request.POST.get('k', 3))  # Get from user input

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        instance = X_scaled[instance_index]

        model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
        model.fit(X_scaled, y)
        prediction = class_names[int(model.predict([instance])[0])]

        # Nearby points
        noise = np.random.normal(0, 0.1, size=(500, X.shape[1]))
        candidates = instance + noise
        candidates = scaler.inverse_transform(candidates)
        candidates_df = pd.DataFrame(candidates, columns=X.columns)
        candidates_scaled = scaler.transform(candidates_df)

        preds = model.predict(candidates_scaled)
        valid = candidates_df[preds == target_class]

        if not valid.empty:
            mad = mad_weights(X.values)
            mad[mad == 0] = 1e-6

            scaled_instance = scaler.inverse_transform([instance])[0]
            diffs = np.abs(valid.values - scaled_instance) / mad
            dists = np.nansum(diffs, axis=1)

            valid['distance'] = dists.astype(float)
            top_k = valid.nsmallest(k, 'distance').drop(columns='distance')

            # Decode one-hot columns
            decoded = top_k.copy()

            # Island
            decoded["island"] = decoded[["island_Dream", "island_Torgersen"]].apply(
                lambda row: "Dream" if row["island_Dream"] > 0.5 else ("Torgersen" if row["island_Torgersen"] > 0.5 else "Biscoe"),
                axis=1
            )
            decoded.drop(columns=["island_Dream", "island_Torgersen"], inplace=True)

            # Sex
            decoded["sex"] = decoded["sex_male"].apply(lambda v: "Male" if v > 0.5 else "Female")
            decoded.drop(columns=["sex_male"], inplace=True)


            decoded = decoded.rename(columns={
                "bill_length_mm": "Bill Length (mm)",
                "bill_depth_mm": "Bill Depth (mm)",
                "flipper_length_mm": "Flipper Length (mm)",
                "body_mass_g": "Body Mass (g)",
                "year": "Year",
            })


            counterfactuals = [{"CF #": i + 1, **row} for i, row in decoded.reset_index(drop=True).iterrows()]


        original_instance = dict(zip(X.columns, scaler.inverse_transform([instance])[0]))

    return render(request, 'project3/counterfactual.html', {
        'instance_options': instance_options,
        'label_options': label_options,
        'selected_instance': instance_index,
        'selected_label': target_class,
        'original': original_instance,
        'prediction': prediction,
        'target': class_names[target_class],
        'counterfactuals': counterfactuals,
        'k':k,
        'instance_number': instance_index,

    })
