"""AI-powered Student Performance Analytics & Early-Warning Dashboard.

Run:
python -m pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
python -m streamlit run app.py
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_performance_model.joblib"
DATA_PATH = ROOT / "student-mat.csv"
OUTPUTS = ROOT / "outputs"
METRICS_PATH = OUTPUTS / "model_comparison.csv"
SUMMARY_PATH = OUTPUTS / "best_model_summary.json"

st.set_page_config(
    page_title="Student Performance AI Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Styling -------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
.hero {padding: 1.3rem 1.5rem; border-radius: 14px;
       background: linear-gradient(120deg, #102a43, #243b53); color: white;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: .5rem 0 0 0; color: #e6edf5;}
.small-note {font-size: .88rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Student Performance AI Analytics</h1>
  <p>Machine-learning prediction • Student risk signals • Model evaluation • Evidence-based insights</p>
</div>
""", unsafe_allow_html=True)
st.caption("Portfolio project for educational demonstration. Predictions may be wrong and must never be used alone to label, penalize, or make high-stakes decisions about students.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Put student_performance_model.joblib in this folder and run src_train_model.py first.")
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    features = list(bundle["features"])
    scenario = bundle.get("scenario", "unknown")
    model_name = bundle.get("model_name", "Saved machine-learning model")
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}")
    st.stop()

def load_csv(path):
    try:
        return pd.read_csv(path, sep=";")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

data = load_csv(DATA_PATH) if DATA_PATH.exists() else None
metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None
summary = {}
if SUMMARY_PATH.exists():
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        summary = {}

with st.sidebar:
    st.header("Project controls")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Scenario:** {scenario.replace('_', ' ').title()}")
    st.divider()
    st.markdown("**Prediction mode**")
    if "G1" in features or "G2" in features:
        st.warning("This model uses previous period grades (G1/G2). It is a later-stage prediction, not a pre-assessment early-warning model.")
    else:
        st.success("This model excludes G1/G2, making it more suitable for an earlier warning.")
    st.markdown("---")
    st.caption("Dataset: UCI Student Performance (Math course) if the local CSV is present.")

# ------------------------- Dataset preparation -------------------------
target_col = "G3"
if data is not None and target_col in data.columns:
    data = data.copy()
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype(str).str.strip()
    dataset_available = True
else:
    dataset_available = False

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict a student",
    "📊 Dataset insights",
    "🧪 Model performance",
    "🧠 Explainability & responsible AI",
])

# ------------------------- Tab 1: Prediction -------------------------
with tab1:
    st.subheader("Individual student assessment")
    st.write("Enter a hypothetical student's information. The app uses only the features expected by the saved model.")

    defaults = {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
        "Pstatus": "T", "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other",
        "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 2,
        "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
        "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1,
        "Walc": 2, "health": 3, "absences": 2, "G1": 10, "G2": 11,
    }
    categorical_options = {
        "school": ["GP", "MS"], "sex": ["F", "M"], "address": ["U", "R"],
        "famsize": ["GT3", "LE3"], "Pstatus": ["T", "A"],
        "Mjob": ["teacher", "health", "services", "at_home", "other"],
        "Fjob": ["teacher", "health", "services", "at_home", "other"],
        "reason": ["home", "reputation", "course", "other"],
        "guardian": ["mother", "father", "other"],
        "schoolsup": ["yes", "no"], "famsup": ["yes", "no"], "paid": ["yes", "no"],
        "activities": ["yes", "no"], "nursery": ["yes", "no"], "higher": ["yes", "no"],
        "internet": ["yes", "no"], "romantic": ["yes", "no"],
    }
    ranges = {
        "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4), "traveltime": (1, 4),
        "studytime": (1, 4), "failures": (0, 4), "famrel": (1, 5),
        "freetime": (1, 5), "goout": (1, 5), "Dalc": (1, 5), "Walc": (1, 5),
        "health": (1, 5), "absences": (0, 100), "G1": (0, 20), "G2": (0, 20),
    }
    with st.form("student_prediction_form"):
        cols = st.columns(3)
        values = {}
        for i, feature in enumerate(features):
            with cols[i % 3]:
                if feature in categorical_options:
                    options = categorical_options[feature]
                    default = defaults.get(feature, options[0])
                    values[feature] = st.selectbox(
                        feature, options,
                        index=options.index(default) if default in options else 0,
                        help=f"Feature used by the saved model: {feature}",
                    )
                elif feature in ranges:
                    low, high = ranges[feature]
                    default = int(np.clip(defaults.get(feature, low), low, high))
                    values[feature] = st.number_input(
                        feature, min_value=low, max_value=high, value=default, step=1
                    )
                else:
                    values[feature] = st.number_input(
                        feature, value=float(defaults.get(feature, 0.0))
                    )
        submitted = st.form_submit_button("✨ Predict final grade", type="primary", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([{f: values[f] for f in features}])
            prediction = float(np.clip(pipeline.predict(input_df)[0], 0, 20))
            st.markdown("### Prediction summary")
            a, b, c = st.columns(3)
            a.metric("Estimated final grade", f"{prediction:.1f} / 20")
            a.progress(float(prediction / 20))
            if prediction < 10:
                category = "Potential support needed"
                advice = "Consider a supportive teacher conversation, topic-level revision, and a follow-up assessment."
                c.metric("Project signal", "Higher support priority")
            elif prediction < 14:
                category = "Developing"
                advice = "Continue regular revision, practice questions, and check progress in the next assessment."
                c.metric("Project signal", "Monitor progress")
            elif prediction < 17:
                category = "Good progress"
                advice = "Maintain effective study routines and review mistakes from assessments."
                c.metric("Project signal", "On track")
            else:
                category = "Strong predicted performance"
                advice = "Maintain learning habits and offer appropriately challenging work."
                c.metric("Project signal", "Positive estimate")
            b.metric("Illustrative category", category)
            st.info(f"**Suggested supportive action:** {advice}")
            st.caption("These categories are project-defined, not official grading bands or validated intervention rules.")
            with st.expander("View the exact input submitted to the model"):
                st.dataframe(input_df, use_container_width=True)
            st.download_button(
                "Download this prediction as CSV",
                data=input_df.assign(predicted_G3=round(prediction, 2), project_category=category).to_csv(index=False).encode("utf-8"),
                file_name="student_prediction.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ------------------------- Tab 2: Dataset insights -------------------------
with tab2:
    st.subheader("Explore the data")
    if not dataset_available:
        st.warning("Dataset analytics require student-mat.csv in the same folder as app.py. The prediction tab can still work without it.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Student records", f"{len(data):,}")
        m2.metric("Input columns", f"{data.shape[1] - 1}")
        m3.metric("Average final grade", f"{data['G3'].mean():.2f}/20")
        m4.metric("Median absences", f"{data['absences'].median():.0f}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Final grade distribution (G3)**")
            fig, ax = plt.subplots()
            ax.hist(data["G3"], bins=range(0, 22), edgecolor="white")
            ax.set_xlabel("Final grade (0–20)")
            ax.set_ylabel("Number of students")
            ax.set_title("Distribution of final grades")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("**Study time vs average final grade**")
            if "studytime" in data.columns:
                grouped = data.groupby("studytime", as_index=False)["G3"].mean()
                fig, ax = plt.subplots()
                ax.bar(grouped["studytime"].astype(str), grouped["G3"])
                ax.set_xlabel("Study time category (1–4)")
                ax.set_ylabel("Average final grade")
                ax.set_title("Association, not proof of causation")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        if "absences" in data.columns:
            st.markdown("**Absences and final grade**")
            fig, ax = plt.subplots()
            ax.scatter(data["absences"], data["G3"], alpha=0.55)
            ax.set_xlabel("Absences")
            ax.set_ylabel("Final grade (G3)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with st.expander("Preview dataset"):
            st.dataframe(data.head(30), use_container_width=True)
        st.download_button("Download dataset summary", data=data.describe(include="all").transpose().to_csv().encode("utf-8"),
                           file_name="dataset_summary.csv", mime="text/csv")

# ------------------------- Tab 3: Model evaluation -------------------------
with tab3:
    st.subheader("Compare model results")
    if metrics is not None and not metrics.empty:
        st.caption("These are results saved by the training script. Check that the evaluation protocol is appropriate before reporting them.")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        numeric_metrics = [c for c in ["MAE", "RMSE", "R2", "CV_MAE_5fold"] if c in metrics.columns]
        if "model" in metrics.columns and "MAE" in metrics.columns:
            plot_df = metrics.copy()
            label_col = "scenario" if "scenario" in plot_df.columns else None
            plot_df["model_label"] = plot_df["model"].astype(str)
            if label_col:
                plot_df["model_label"] = plot_df[label_col].astype(str) + " — " + plot_df["model"].astype(str)
            plot_df = plot_df.sort_values("MAE", ascending=True)
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(plot_df))))
            ax.barh(plot_df["model_label"], plot_df["MAE"])
            ax.set_xlabel("MAE (lower is better)")
            ax.set_title("Test-set mean absolute error")
            ax.invert_yaxis()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        if summary:
            st.markdown("**Selected model summary**")
            s1, s2, s3 = st.columns(3)
            s1.metric("Selected model", str(summary.get("best_model", model_name)))
            s2.metric("Test MAE", f"{float(summary.get('test_MAE', float('nan'))):.2f}")
            s3.metric("Test R²", f"{float(summary.get('test_R2', float('nan'))):.3f}")
        st.download_button("Download model comparison CSV", data=metrics.to_csv(index=False).encode("utf-8"),
                           file_name="model_comparison.csv", mime="text/csv")
    else:
        st.warning("Model comparison file not found. Run src_train_model.py to generate outputs/model_comparison.csv.")
        if summary:
            st.json(summary)
    st.markdown("**Metric guide**")
    st.write("- **MAE:** average absolute error in grade points; lower is better.")
    st.write("- **RMSE:** penalizes large errors more strongly; lower is better.")
    st.write("- **R²:** relative to a mean-baseline model; higher is generally better, but can be negative.")
    st.write("- **Cross-validation MAE:** average error across folds; useful for checking stability.")

# ------------------------- Tab 4: Explainability and responsible AI -------------------------
with tab4:
    st.subheader("Explainability, limitations and next steps")
    st.markdown("**What makes this an AI/ML project?** The model learns patterns from historical examples rather than using only hand-written rules. The dashboard turns the model into an interactive decision-support prototype.")
    if dataset_available:
        st.markdown("**Dataset-level signals (correlations)**")
        numeric = data.select_dtypes(include=[np.number])
        if "G3" in numeric.columns:
            corr = numeric.corr(numeric_only=True)["G3"].drop("G3").sort_values()
            low = corr.head(5)
            high = corr.tail(5)
            corr_show = pd.concat([low, high]).drop_duplicates()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(corr_show.index.astype(str), corr_show.values)
            ax.set_xlabel("Pearson correlation with G3")
            ax.set_title("Top positive and negative numeric associations")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Correlation is descriptive, not causal. These associations are not automatically the model's feature importance.")
    st.markdown("**Known limitations to discuss in your report**")
    st.markdown("""
- The dataset is relatively small and comes from a specific educational context; performance may not transfer to other schools.
- A random train/test split does not guarantee performance on future cohorts or different schools.
- The `with_period_grades` scenario uses G1/G2, which may not be available for an early-warning intervention.
- Correlation does not show that a feature causes a grade outcome.
- Family and demographic variables can reflect structural disadvantage. Do not use this prototype to punish, rank, or deny opportunities to students.
- Before deployment, evaluate fairness across relevant groups, calibrate uncertainty, validate on new data, and involve educators.
""")
    st.markdown("**Strong next upgrades for a portfolio version**")
    st.markdown("""
1. Compare an early-warning model against a later-stage model using identical, documented test splits.
2. Add permutation importance or SHAP explanations on held-out data.
3. Add uncertainty estimates and clearly flag low-confidence predictions.
4. Create a reproducible training pipeline with saved metrics, versioned dataset, and GitHub README.
5. Validate with a separate school dataset before claiming real-world readiness.
""")
    st.warning("This is a machine-learning decision-support prototype, not a deployed student assessment product. Do not describe it as a fully autonomous AI system.")
