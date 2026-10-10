"""
Student Performance Prediction System - Intermediate ML project
Dataset: UCI Student Performance (student-mat.csv or student-por.csv)
Target: G3 (final grade, 0-20)
Run: python src_train_model.py
"""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student-mat.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Download student-mat.csv from the UCI Student Performance page and place it beside this script."
    )

df = pd.read_csv(DATA_PATH, sep=";")
print("Original shape:", df.shape)
print("Missing values before cleaning:\n", df.isna().sum().sort_values(ascending=False).head(10))
print("Duplicate rows:", int(df.duplicated().sum()))

df = df.drop_duplicates().copy()
if "G3" not in df.columns:
    raise ValueError("Dataset must contain target column G3.")

# G3=0 is a valid observed grade in this dataset; do not remove it automatically.
df = df.dropna(subset=["G3"]).copy()
df["G3"] = pd.to_numeric(df["G3"], errors="coerce")
df = df.dropna(subset=["G3"]).copy()

# Save basic data-quality summary
quality = {
    "rows_after_cleaning": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "duplicate_rows_remaining": int(df.duplicated().sum()),
    "missing_cells_remaining": int(df.isna().sum().sum()),
    "target_min": float(df["G3"].min()),
    "target_max": float(df["G3"].max()),
}
(OUT / "data_quality.json").write_text(json.dumps(quality, indent=2))

# EDA plots
sns.set_theme()
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=15, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final grade (0-20)")
plt.tight_layout()
plt.savefig(OUT / "grade_distribution.png", dpi=160)
plt.close()

numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=160)
plt.close()

# Exclude target from features. For a realistic early-warning use case, also exclude G1/G2
# to measure prediction before period grades are known. Compare both scenarios.
scenarios = {
    "early_warning": [c for c in df.columns if c not in ["G3", "G1", "G2"]],
    "with_period_grades": [c for c in df.columns if c != "G3"],
}
all_results = []
best_model = None
best_pipeline = None
best_scenario = None
best_mae = float("inf")

for scenario_name, feature_cols in scenarios.items():
    X = df[feature_cols].copy()
    y = df["G3"].copy()

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Baseline_Median": DummyRegressor(strategy="median"),
        "Linear_Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
    }

    for model_name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        # CV uses training data only; report average CV MAE as a stability check.
        cv_neg_mae = cross_val_score(
            pipe, X_train, y_train, cv=5,
            scoring="neg_mean_absolute_error", n_jobs=None
        )
        cv_mae = float(-cv_neg_mae.mean())
        all_results.append({
            "scenario": scenario_name,
            "model": model_name,
            "MAE": float(mae),
            "RMSE": rmse,
            "R2": float(r2),
            "CV_MAE_5fold": cv_mae,
            "test_rows": int(len(y_test)),
            "features_count": int(len(feature_cols)),
        })
        if model_name != "Baseline_Median" and mae < best_mae:
            best_mae, best_model, best_pipeline, best_scenario = mae, model_name, pipe, scenario_name

results = pd.DataFrame(all_results).sort_values(["scenario", "MAE"])
results.to_csv(OUT / "model_comparison.csv", index=False)
print("\nModel comparison:\n", results.to_string(index=False))

# Save actual vs predicted for selected best model and its scenario.
best_features = scenarios[best_scenario]
X = df[best_features].copy()
y = df["G3"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_pipeline.fit(X_train, y_train)
pred = best_pipeline.predict(X_test)
predictions = pd.DataFrame({"actual_G3": y_test.to_numpy(), "predicted_G3": pred})
predictions["absolute_error"] = abs(predictions["actual_G3"] - predictions["predicted_G3"])
predictions.to_csv(OUT / "test_predictions.csv", index=False)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.75)
plt.plot([0, 20], [0, 20], linestyle="--")
plt.xlabel("Actual final grade")
plt.ylabel("Predicted final grade")
plt.title(f"Actual vs Predicted ({best_model}, {best_scenario})")
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

# Save model with its feature names and metadata
bundle = {
    "pipeline": best_pipeline,
    "features": best_features,
    "model_name": best_model,
    "scenario": best_scenario,
    "target": "G3",
    "dataset_file": DATA_PATH.name,
}
joblib.dump(bundle, ROOT / "student_performance_model.joblib")

summary = {
    "best_model": best_model,
    "best_scenario": best_scenario,
    "test_MAE": float(mean_absolute_error(y_test, pred)),
    "test_RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "test_R2": float(r2_score(y_test, pred)),
    "interpretation_note": (
        "Metrics are from a single 80/20 holdout split. They are not a guarantee of future performance. "
        "Compare the early_warning and with_period_grades scenarios before claiming the model is useful."
    ),
}
(OUT / "best_model_summary.json").write_text(json.dumps(summary, indent=2))
print("\nSaved model:", ROOT / "student_performance_model.joblib")
print("Best model summary:\n", json.dumps(summary, indent=2))
print("\nNext: run `streamlit run app.py`")
