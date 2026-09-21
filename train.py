"""Train and evaluate churn classifiers on the synthetic dataset.

Compares logistic regression vs. random forest, prints precision/recall/F1,
and saves a confusion matrix + feature importance chart to images/.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import generate_churn_data

NUMERIC = ["tenure_months", "monthly_charges", "total_charges",
           "support_calls", "senior_citizen"]
CATEGORICAL = ["contract_type", "internet_service", "tech_support",
               "paperless_billing"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def plot_confusion_matrix(y_true, y_pred, title, path):
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                                  display_labels=["stayed", "churned"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, path, top_n=12):
    importances = model.named_steps["clf"].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    feat_imp.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} features — Random Forest")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs("images", exist_ok=True)

    df = generate_churn_data(n_customers=5000, seed=42)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  "
          f"Churn rate: {y.mean():.1%}\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    preprocessor = build_preprocessor()
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(f"=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
        print(classification_report(y_test, pred,
                                    target_names=["stayed", "churned"]))
        plot_confusion_matrix(
            y_test, pred, f"Confusion matrix — {name}",
            f"images/confusion_matrix_{name.lower().replace(' ', '_')}.png",
        )
        if name == "Random Forest":
            feature_names = pipe.named_steps["prep"].get_feature_names_out()
            plot_feature_importance(pipe, feature_names,
                                    "images/feature_importance.png")

    print("Charts saved to images/")


if __name__ == "__main__":
    main()
