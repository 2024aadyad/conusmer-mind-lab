

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Load the shopping-session dataset
DATA_PATH = "data/raw/online_shoppers.csv"
MODEL_PATH = "models/purchase_model.pkl"

print("LOADING CONSUMER SHOPPING DATA...")
df = pd.read_csv(DATA_PATH)

# Predictors available before the purchase outcome is known.
# Revenue is deliberately excluded to prevent target leakage.
numeric_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
categorical_features = ["VisitorType", "Weekend"]
all_features = numeric_features + categorical_features

X = df[all_features].copy()
y = df["Revenue"].astype(int)

# Split before training so evaluation uses unseen sessions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Encode categorical inputs; numeric features can be used directly by a forest.
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("TRAINING RANDOM FOREST MODEL...")
model.fit(X_train, y_train)

# Evaluate on the held-out test set.
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nMODEL EVALUATION")
print("----------------")
print(f"Test sessions: {len(X_test):,}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save the complete preprocessing + model pipeline for the Streamlit app.
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nMODEL SAVED TO: {MODEL_PATH}")
print("PURCHASE PREDICTION MODEL TRAINING COMPLETE!")