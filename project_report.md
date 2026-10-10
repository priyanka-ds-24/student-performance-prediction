# Student Performance Prediction System — Project Report

## 1. Abstract
This project investigates supervised machine learning for estimating students' final mathematics grades from student-related data. It includes data cleaning, exploratory analysis, feature preprocessing, model comparison, regression evaluation, and an interactive prediction interface.

## 2. Problem statement
Can historical student and academic information be used to estimate final grade `G3`? The project compares a scenario using only non-grade features with a scenario that also includes first- and second-period grades.

## 3. Dataset and source
Source: UCI Student Performance dataset, https://archive.ics.uci.edu/dataset/320/student+performance. The data concerns secondary-school students in two Portuguese schools. Record the exact file used, row/column counts, and license/attribution details.

## 4. Methodology
1. Load the semicolon-separated CSV.
2. Inspect shape, data types, missing values, duplicates and summary statistics.
3. Remove duplicate rows and records with missing target values; preserve valid grade values such as zero.
4. Explore final-grade distribution and numeric correlations.
5. Split into training (80%) and testing (20%) data using a fixed random seed.
6. Impute numeric and categorical missing values within a scikit-learn pipeline; scale numeric values and one-hot encode categorical values.
7. Compare a median baseline, Linear Regression, Ridge, Random Forest and Gradient Boosting.
8. Use 5-fold cross-validation on the training set and evaluate on the held-out test set.
9. Save the selected model and predictions; run the Streamlit interface.

## 5. Results
Complete this section after running the training script. Copy the real metrics from `outputs/model_comparison.csv` and `outputs/best_model_summary.json`.

| Scenario | Model | MAE | RMSE | R² | 5-fold CV MAE |
|---|---|---:|---:|---:|---:|
| Fill from model_comparison.csv | | | | | |

Do not invent or pre-fill model scores. Discuss the trade-offs between early-warning predictions and predictions that use prior grades.

## 6. Visualizations
Insert and explain:
- `outputs/grade_distribution.png`
- `outputs/correlation_heatmap.png`
- `outputs/actual_vs_predicted.png`

## 7. Limitations and ethics
The dataset is limited in size and geography. The model estimates grades, not student potential. Correlation does not prove causation. Avoid high-stakes use, and do not use predictions to deny students opportunities or support. Consider fairness, privacy and data minimization.

## 8. Conclusion
Write a conclusion based on measured results only. State which model/scenario performed best on the held-out set, how large its typical error was, and what further validation is needed.

## 9. Future improvements
- Repeat evaluation with multiple random splits or nested cross-validation.
- Test on a separate dataset or school before any real-world claim.
- Add permutation importance or SHAP with careful interpretation.
- Build a teacher-facing interface that emphasizes support rather than ranking.
