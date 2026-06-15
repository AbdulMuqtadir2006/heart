import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Page setup
# =========================================================

APP_VERSION = "final_v4"

st.set_page_config(
    page_title="CardioGuard AI",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .section-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(150,150,150,0.25);
        background: rgba(120,120,120,0.08);
        margin-bottom: 18px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(120,120,120,0.08);
        border: 1px solid rgba(150,150,150,0.18);
        padding: 18px;
        border-radius: 16px;
    }

    .low {
        color: #41d67c;
        font-weight: 800;
    }

    .moderate {
        color: #f4c542;
        font-weight: 800;
    }

    .high {
        color: #ff6b6b;
        font-weight: 800;
    }

    .critical {
        color: #ff3333;
        font-weight: 900;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">CardioGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Cardiovascular risk prediction, model training, patient analysis, and live monitoring simulation.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# Feature information
# =========================================================

FEATURE_INFO = {
    "age": [
        "Age",
        "Important",
        "Older age can increase cardiovascular risk.",
    ],
    "sex": [
        "Sex",
        "Medium",
        "Biological sex is used as one of the clinical risk factors in this dataset.",
    ],
    "cp": [
        "Chest Pain Type",
        "Very important",
        "Type of chest pain. Chest pain pattern is strongly related to cardiac symptoms.",
    ],
    "trestbps": [
        "Resting Blood Pressure",
        "Important",
        "Blood pressure while resting. High values can increase cardiovascular risk.",
    ],
    "chol": [
        "Serum Cholesterol",
        "Important",
        "Cholesterol level in blood. High cholesterol can increase heart risk.",
    ],
    "fbs": [
        "Fasting Blood Sugar",
        "Medium",
        "Shows whether fasting blood sugar is high.",
    ],
    "restecg": [
        "Resting Electrocardiogram Result",
        "Important",
        "ECG result while resting. Abnormal ECG patterns can show heart problems.",
    ],
    "thalach": [
        "Maximum Heart Rate Achieved",
        "Very important",
        "Maximum heart rate during exercise. Lower exercise performance can indicate higher risk.",
    ],
    "exang": [
        "Exercise-Induced Angina",
        "Very important",
        "Chest pain caused by exercise. This is a strong warning signal.",
    ],
    "oldpeak": [
        "ST Depression Induced by Exercise",
        "Very important",
        "ECG stress-test value. Higher values often indicate higher heart risk.",
    ],
    "slope": [
        "Slope of Peak Exercise ST Segment",
        "Important",
        "ECG stress-test pattern that helps detect abnormal heart response.",
    ],
    "ca": [
        "Number of Major Vessels Colored by Fluoroscopy",
        "Very important",
        "Shows affected major blood vessels. More affected vessels usually means higher risk.",
    ],
    "thal": [
        "Thalassemia / Thallium Stress Test Result",
        "Very important",
        "Stress-test related heart blood-flow indicator.",
    ],
}


REQUIRED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


# =========================================================
# Data loading
# =========================================================

@st.cache_data
def load_local_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        st.error(
            f"Could not find '{csv_path}'. Keep heart_dummy_100.csv in the same GitHub folder as heart.py."
        )
        st.stop()

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")
    df = df.drop_duplicates()

    return df


def validate_data(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in CSV: {missing_columns}")
        st.stop()

    if df["target"].nunique() < 2:
        st.error("The target column must contain both 0 and 1 values.")
        st.stop()


def safe_roc_auc(y_true, probabilities):
    try:
        return roc_auc_score(y_true, probabilities)
    except Exception:
        return 0.0


# =========================================================
# Model training
# =========================================================

def train_ai(df: pd.DataFrame):
    df = clean_data(df)
    validate_data(df)

    X = df.drop(columns=["target"])
    y = (df["target"] > 0).astype(int)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=7,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    threshold = 0.45
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        auc = safe_roc_auc(y_test, probabilities)

        results.append(
            {
                "model": name,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4),
                "threshold": threshold,
            }
        )

        try:
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                scoring="roc_auc",
                n_repeats=8,
                random_state=42,
            )

            importance_values = perm.importances_mean

        except Exception:
            importance_values = np.zeros(len(feature_names))

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        )

        importance_df["importance"] = importance_df["importance"].clip(lower=0)
        importance_df = importance_df.sort_values("importance", ascending=False)

        trained_models[name] = {
            "schema_version": APP_VERSION,
            "model": model,
            "features": feature_names,
            "threshold": threshold,
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "importance_df": importance_df,
            "test_probabilities": probabilities,
            "test_predictions": predictions,
            "y_test": y_test.reset_index(drop=True),
        }

    leaderboard = pd.DataFrame(results).sort_values(
        by=["recall", "roc_auc", "f1_score"],
        ascending=False,
    )

    best_model_name = leaderboard.iloc[0]["model"]
    best_model_bundle = trained_models[best_model_name]

    return leaderboard, best_model_name, best_model_bundle


def train_and_store_model(df: pd.DataFrame):
    leaderboard, best_model_name, best_model = train_ai(df)

    st.session_state["leaderboard"] = leaderboard
    st.session_state["best_model_name"] = best_model_name
    st.session_state["best_model"] = best_model
    st.session_state["app_version"] = APP_VERSION

    joblib.dump(best_model, "cardioguard_model.joblib")


def model_needs_training() -> bool:
    if st.session_state.get("app_version") != APP_VERSION:
        return True

    if "leaderboard" not in st.session_state:
        return True

    if "best_model_name" not in st.session_state:
        return True

    if "best_model" not in st.session_state:
        return True

    best_model_check = st.session_state["best_model"]

    required_keys = [
        "schema_version",
        "model",
        "features",
        "threshold",
        "confusion_matrix",
        "importance_df",
        "test_probabilities",
        "test_predictions",
        "y_test",
    ]

    for key in required_keys:
        if key not in best_model_check:
            return True

    if best_model_check.get("schema_version") != APP_VERSION:
        return True

    return False


# =========================================================
# Prediction helpers
# =========================================================

def predict_risk(model_bundle, patient_data: dict):
    model = model_bundle["model"]
    features = model_bundle["features"]
    threshold = model_bundle["threshold"]

    patient_df = pd.DataFrame([patient_data])
    patient_df = patient_df[features]

    probability = float(model.predict_proba(patient_df)[:, 1][0])

    if probability >= 0.80:
        level = "CRITICAL"
    elif probability >= threshold:
        level = "HIGH"
    elif probability >= 0.35:
        level = "MODERATE"
    else:
        level = "LOW"

    alert = "YES" if level in ["HIGH", "CRITICAL"] else "NO"

    return probability, level, alert


def get_level_class(level: str) -> str:
    if level == "LOW":
        return "low"
    if level == "MODERATE":
        return "moderate"
    if level == "HIGH":
        return "high"
    return "critical"


def build_patient_comparison(df: pd.DataFrame, patient_data: dict) -> pd.DataFrame:
    rows = []

    for feature, value in patient_data.items():
        dataset_median = float(df[feature].median())
        dataset_min = float(df[feature].min())
        dataset_max = float(df[feature].max())

        if dataset_max == dataset_min:
            normalized_patient = 0
            normalized_median = 0
        else:
            normalized_patient = (value - dataset_min) / (dataset_max - dataset_min)
            normalized_median = (dataset_median - dataset_min) / (dataset_max - dataset_min)

        rows.append(
            {
                "feature": feature,
                "patient_profile": normalized_patient,
                "dataset_median": normalized_median,
            }
        )

    return pd.DataFrame(rows)


def create_risk_factor_chart(df: pd.DataFrame, patient_data: dict) -> pd.DataFrame:
    selected_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "ca",
        "thal",
    ]

    rows = []

    for feature in selected_features:
        if feature in patient_data:
            rows.append(
                {
                    "feature": feature,
                    "patient": float(patient_data[feature]),
                    "dataset_median": float(df[feature].median()),
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.header("Controls")

csv_path = st.sidebar.text_input(
    "CSV file path",
    value="heart_dummy_100.csv",
    help="Keep this CSV file in the same GitHub folder as heart.py.",
)

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload another CSV",
    type=["csv"],
)

st.sidebar.divider()

train_button = st.sidebar.button("Train / Re-train Model", use_container_width=True)

st.sidebar.caption("Data source: local CSV file in your GitHub repository.")


# =========================================================
# Load data
# =========================================================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_local_data(csv_path)

df = clean_data(df)
validate_data(df)


# =========================================================
# Safe auto-training
# =========================================================

if train_button or model_needs_training():
    with st.spinner("Training model..."):
        train_and_store_model(df)


leaderboard = st.session_state.get("leaderboard")
best_model_name = st.session_state.get("best_model_name")
best_model = st.session_state.get("best_model")

if leaderboard is None or best_model_name is None or best_model is None:
    st.error("Model training did not complete. Click Train / Re-train Model again.")
    st.stop()


# =========================================================
# Tabs
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset", "Train Model", "Predict Risk", "Live Simulation"]
)


# =========================================================
# Tab 1: Dataset
# =========================================================

with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", len(df))
    c2.metric("Input Features", len(df.columns) - 1)
    c3.metric("Risk Cases", int(df["target"].sum()))
    c4.metric("No-Risk Cases", int((df["target"] == 0).sum()))

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

    with right:
        st.subheader("Risk Class Distribution")

        distribution_df = pd.DataFrame(
            {
                "class": ["No Risk", "Risk"],
                "count": [
                    int((df["target"] == 0).sum()),
                    int((df["target"] == 1).sum()),
                ],
            }
        )

        st.bar_chart(distribution_df.set_index("class")["count"])

        st.subheader("Age Distribution")

        age_bins = pd.cut(df["age"], bins=6)
        age_distribution = df.groupby(age_bins, observed=False).size()
        age_distribution.index = age_distribution.index.astype(str)

        st.bar_chart(age_distribution)

    st.subheader("Important Medical Features")

    feature_table = pd.DataFrame(
        [
            {
                "Short Name": key,
                "Full Form": value[0],
                "Importance": value[1],
                "Meaning": value[2],
            }
            for key, value in FEATURE_INFO.items()
            if key not in ["age", "sex"]
        ]
    )

    st.dataframe(feature_table, use_container_width=True)

    st.subheader("Average Feature Values by Class")

    average_df = df.groupby("target").mean(numeric_only=True).T
    average_df.columns = ["No Risk Average", "Risk Average"]

    st.dataframe(average_df, use_container_width=True)
    st.bar_chart(average_df[["No Risk Average", "Risk Average"]])


# =========================================================
# Tab 2: Train Model
# =========================================================

with tab2:
    st.subheader("Model Training and Testing")

    st.dataframe(leaderboard, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Selected Model", best_model_name)
    c2.metric("Decision Threshold", best_model["threshold"])
    c3.metric("Saved Model File", "cardioguard_model.joblib")

    st.subheader("Model Performance Comparison")

    metric_chart = leaderboard.set_index("model")[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]

    st.bar_chart(metric_chart)

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")

        cm = best_model["confusion_matrix"]

        cm_df = pd.DataFrame(
            cm,
            index=["Actual No Risk", "Actual Risk"],
            columns=["Predicted No Risk", "Predicted Risk"],
        )

        st.dataframe(cm_df, use_container_width=True)

    with right:
        st.subheader("Test Prediction Distribution")

        test_probabilities = best_model.get("test_probabilities")

        if test_probabilities is None:
            st.info("Prediction distribution is not available. Re-train the model.")
        else:
            probability_bins = pd.cut(test_probabilities, bins=6)
            probability_distribution = pd.Series(test_probabilities).groupby(
                probability_bins,
                observed=False,
            ).size()

            probability_distribution.index = probability_distribution.index.astype(str)

            st.bar_chart(probability_distribution)

    st.subheader("Feature Importance")

    importance_df = best_model.get("importance_df")

    if importance_df is None or importance_df.empty:
        st.info("Feature importance is not available. Re-train the model.")
    else:
        importance_df = importance_df.copy()
        importance_df["importance"] = importance_df["importance"].clip(lower=0)

        st.dataframe(importance_df, use_container_width=True)

        if importance_df["importance"].sum() == 0:
            st.info("Feature importance values are close to zero for this test split.")
        else:
            st.bar_chart(importance_df.set_index("feature")["importance"])


# =========================================================
# Tab 3: Predict Risk
# =========================================================

with tab3:
    st.subheader("Patient Risk Prediction")

    features = best_model["features"]
    patient_data = {}

    st.markdown("Adjust the values below to create a patient profile.")

    col_a, col_b, col_c = st.columns(3)
    layout_cols = [col_a, col_b, col_c]

    for index, feature in enumerate(features):
        minimum = float(df[feature].min())
        maximum = float(df[feature].max())
        median = float(df[feature].median())

        if minimum == maximum:
            maximum = minimum + 1

        help_text = FEATURE_INFO.get(feature, ["", "", ""])[2]

        with layout_cols[index % 3]:
            patient_data[feature] = st.slider(
                label=feature,
                min_value=minimum,
                max_value=maximum,
                value=median,
                help=help_text,
            )

    risk_probability, risk_level, emergency_alert = predict_risk(best_model, patient_data)
    risk_percent = risk_probability * 100

    st.divider()

    m1, m2, m3 = st.columns(3)

    m1.metric("Risk Score", f"{risk_percent:.2f}%")
    m2.metric("Risk Level", risk_level)
    m3.metric("Emergency Alert", emergency_alert)

    st.progress(min(max(risk_probability, 0), 1))

    level_class = get_level_class(risk_level)

    st.markdown(
        f"""
        <div class="section-card">
            <h3>Current Assessment</h3>
            <p>Predicted risk level: <span class="{level_class}">{risk_level}</span></p>
            <p>Risk score: <b>{risk_percent:.2f}%</b></p>
            <p>Emergency alert status: <b>{emergency_alert}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Patient vs Dataset Median")

        comparison_df = create_risk_factor_chart(df, patient_data)

        if not comparison_df.empty:
            comparison_chart = comparison_df.set_index("feature")[
                ["patient", "dataset_median"]
            ]

            st.bar_chart(comparison_chart)

    with right:
        st.subheader("Normalized Patient Profile")

        normalized_df = build_patient_comparison(df, patient_data)

        if not normalized_df.empty:
            normalized_chart = normalized_df.set_index("feature")[
                ["patient_profile", "dataset_median"]
            ]

            st.bar_chart(normalized_chart)

    st.subheader("Risk Level Reference")

    risk_reference = pd.DataFrame(
        {
            "Risk Level": ["LOW", "MODERATE", "HIGH", "CRITICAL"],
            "Minimum Score": [0, 35, 45, 80],
            "Maximum Score": [34.99, 44.99, 79.99, 100],
        }
    )

    st.dataframe(risk_reference, use_container_width=True)

    st.subheader("Feature Explanation")

    explanation_rows = []

    for feature in features:
        full_form = FEATURE_INFO.get(feature, [feature, "Not listed", ""])[0]
        importance = FEATURE_INFO.get(feature, [feature, "Not listed", ""])[1]
        meaning = FEATURE_INFO.get(feature, [feature, "Not listed", ""])[2]

        explanation_rows.append(
            {
                "Feature": feature,
                "Full Form": full_form,
                "Importance": importance,
                "Current Value": round(float(patient_data[feature]), 2),
                "Meaning": meaning,
            }
        )

    st.dataframe(pd.DataFrame(explanation_rows), use_container_width=True)

    if risk_level == "CRITICAL":
        st.error("Critical alert: very high predicted heart risk.")
    elif risk_level == "HIGH":
        st.warning("High alert: high predicted heart risk.")
    elif risk_level == "MODERATE":
        st.info("Moderate risk: continue monitoring.")
    else:
        st.success("Low predicted risk.")


# =========================================================
# Tab 4: Live Simulation
# =========================================================

with tab4:
    st.subheader("Live Monitoring Simulation")

    features = best_model["features"]

    st.write(
        "This section simulates continuous incoming data from a wearable device. "
        "Heart-rate related values, blood pressure, and ECG stress indicator are changed over time."
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        simulation_seconds = st.slider("Simulation length in seconds", 5, 60, 25)

    with c2:
        st.markdown(
            """
            The simulation starts from the median patient profile in the CSV.  
            Every step recalculates risk and updates the monitoring chart.
            """
        )

    base_patient = {feature: float(df[feature].median()) for feature in features}

    if st.button("Start Live Simulation", use_container_width=True):
        live_box = st.empty()
        chart_box = st.empty()
        table_box = st.empty()

        logs = []

        for second in range(simulation_seconds):
            live_patient = base_patient.copy()

            if "thalach" in live_patient:
                live_patient["thalach"] = max(
                    60,
                    live_patient["thalach"] + np.random.randint(-12, 16),
                )

            if "trestbps" in live_patient:
                live_patient["trestbps"] = max(
                    80,
                    live_patient["trestbps"] + np.random.randint(-6, 9),
                )

            if "oldpeak" in live_patient:
                live_patient["oldpeak"] = max(
                    0,
                    live_patient["oldpeak"] + np.random.normal(0, 0.25),
                )

            risk_probability, risk_level, emergency_alert = predict_risk(
                best_model,
                live_patient,
            )

            logs.append(
                {
                    "second": second + 1,
                    "risk_score": round(risk_probability * 100, 2),
                    "risk_level": risk_level,
                    "emergency_alert": emergency_alert,
                    "thalach": round(live_patient.get("thalach", 0), 2),
                    "trestbps": round(live_patient.get("trestbps", 0), 2),
                    "oldpeak": round(live_patient.get("oldpeak", 0), 2),
                }
            )

            logs_df = pd.DataFrame(logs)

            with live_box.container():
                a, b, c, d = st.columns(4)

                a.metric("Second", second + 1)
                b.metric("Risk Score", f"{risk_probability * 100:.2f}%")
                c.metric("Risk Level", risk_level)
                d.metric("Alert", emergency_alert)

                if emergency_alert == "YES":
                    st.error("Demo emergency alert triggered.")
                else:
                    st.success("Monitoring normal.")

            chart_data = logs_df.set_index("second")[
                ["risk_score", "thalach", "trestbps", "oldpeak"]
            ]

            chart_box.line_chart(chart_data)

            table_box.dataframe(logs_df.tail(8), use_container_width=True)

            time.sleep(0.4)

        st.subheader("Simulation Log")

        st.dataframe(logs_df, use_container_width=True)

        csv = logs_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Simulation Log",
            data=csv,
            file_name="cardioguard_simulation_log.csv",
            mime="text/csv",
        )
