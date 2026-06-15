import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="❤️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        opacity: 0.78;
        margin-bottom: 25px;
    }
    .section-box {
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.08);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">❤️ CardioGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Heart-risk prediction, AI model training, wearable simulation, and emergency alert demo.</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Educational prototype only. This is not a medical device and must not be used for real diagnosis."
)


# ---------------------------------------------------------
# Feature information
# ---------------------------------------------------------

FEATURE_INFO = {
    "cp": [
        "Chest Pain Type",
        "Very important",
        "Represents the type of chest pain. Chest pain pattern is strongly related to cardiac symptoms.",
    ],
    "trestbps": [
        "Resting Blood Pressure",
        "Important",
        "Blood pressure while resting. High values can increase cardiovascular risk.",
    ],
    "chol": [
        "Serum Cholesterol",
        "Important",
        "Cholesterol level in the blood. High cholesterol can increase heart risk.",
    ],
    "fbs": [
        "Fasting Blood Sugar",
        "Medium",
        "Shows whether fasting blood sugar is high.",
    ],
    "restecg": [
        "Resting Electrocardiogram Result",
        "Important",
        "ECG result while resting. Abnormal ECG patterns can indicate heart problems.",
    ],
    "thalach": [
        "Maximum Heart Rate Achieved",
        "Very important",
        "Maximum heart rate reached during exercise. Lower exercise performance can suggest risk.",
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


# ---------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------

@st.cache_data
def load_local_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        st.error(
            f"Could not find '{csv_path}'. Put heart_dummy_100.csv in the same GitHub folder as heart.py."
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
    required_columns = [
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

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in CSV: {missing_columns}")
        st.stop()

    if df["target"].nunique() < 2:
        st.error("The target column must contain both 0 and 1 values.")
        st.stop()


# ---------------------------------------------------------
# AI model training
# ---------------------------------------------------------

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
                        n_estimators=300,
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

        results.append(
            {
                "model": name,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
                "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
                "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
                "threshold": threshold,
            }
        )

        trained_models[name] = {
            "model": model,
            "features": feature_names,
            "threshold": threshold,
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }

    leaderboard = pd.DataFrame(results).sort_values(
        by=["recall", "roc_auc", "f1_score"],
        ascending=False,
    )

    best_model_name = leaderboard.iloc[0]["model"]
    best_model_bundle = trained_models[best_model_name]

    return leaderboard, best_model_name, best_model_bundle


# ---------------------------------------------------------
# Prediction and feature importance
# ---------------------------------------------------------

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


def get_feature_importance(model_bundle):
    model = model_bundle["model"]
    features = model_bundle["features"]
    final_model = model.steps[-1][1]

    if hasattr(final_model, "feature_importances_"):
        values = final_model.feature_importances_
    else:
        values = np.zeros(len(features))

    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)

    return importance_df


def train_and_store_model(df: pd.DataFrame):
    leaderboard, best_model_name, best_model = train_ai(df)

    st.session_state["leaderboard"] = leaderboard
    st.session_state["best_model_name"] = best_model_name
    st.session_state["best_model"] = best_model

    joblib.dump(best_model, "cardioguard_model.joblib")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("⚙️ Controls")

csv_path = st.sidebar.text_input(
    "CSV file path",
    value="heart_dummy_100.csv",
    help="Keep this CSV file in the same GitHub folder as heart.py.",
)

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload a different CSV",
    type=["csv"],
)

st.sidebar.divider()

train_button = st.sidebar.button("🚀 Train / Re-train AI Model", use_container_width=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_local_data(csv_path)

df = clean_data(df)
validate_data(df)


# ---------------------------------------------------------
# Auto-train safely
# ---------------------------------------------------------

if (
    train_button
    or "leaderboard" not in st.session_state
    or "best_model_name" not in st.session_state
    or "best_model" not in st.session_state
):
    with st.spinner("Training AI model..."):
        train_and_store_model(df)


leaderboard = st.session_state.get("leaderboard")
best_model_name = st.session_state.get("best_model_name")
best_model = st.session_state.get("best_model")

if leaderboard is None or best_model_name is None or best_model is None:
    st.error("Model training did not complete. Click Train / Re-train AI Model again.")
    st.stop()


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset", "🧠 Train Model", "🩺 Predict Risk", "⌚ Live Simulation"]
)


# ---------------------------------------------------------
# Tab 1: Dataset
# ---------------------------------------------------------

with tab1:
    st.subheader("Dataset Preview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Features", len(df.columns) - 1)
    c3.metric("Risk Cases", int(df["target"].sum()))
    c4.metric("No-Risk Cases", int((df["target"] == 0).sum()))

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Important Medical Features")

    feature_table = pd.DataFrame(
        [
            {
                "Short name": key,
                "Full form": value[0],
                "Importance": value[1],
                "Why it matters": value[2],
            }
            for key, value in FEATURE_INFO.items()
        ]
    )

    st.dataframe(feature_table, use_container_width=True)

    st.info(
        "The CSV is used as local training data. For a real medical system, this dummy data should be replaced with clinically validated patient or wearable data."
    )


# ---------------------------------------------------------
# Tab 2: Training
# ---------------------------------------------------------

with tab2:
    st.subheader("AI Training and Testing")

    st.dataframe(leaderboard, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_model_name)
    c2.metric("Decision Threshold", best_model["threshold"])
    c3.metric("Model Saved", "cardioguard_model.joblib")

    st.subheader("Confusion Matrix")

    cm = best_model["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Risk", "Actual Risk"],
        columns=["Predicted No Risk", "Predicted Risk"],
    )

    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Feature Importance")

    importance_df = get_feature_importance(best_model)
    st.dataframe(importance_df, use_container_width=True)
    st.bar_chart(importance_df.set_index("feature")["importance"])


# ---------------------------------------------------------
# Tab 3: Prediction
# ---------------------------------------------------------

with tab3:
    st.subheader("Patient Risk Prediction")

    features = best_model["features"]
    patient_data = {}

    st.markdown("Adjust the patient values below. The AI will calculate a risk score.")

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

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Score", f"{risk_probability * 100:.2f}%")
    m2.metric("Risk Level", risk_level)
    m3.metric("Emergency Alert", emergency_alert)

    if risk_level == "CRITICAL":
        st.error("🚨 CRITICAL ALERT: Very high predicted heart risk.")
    elif risk_level == "HIGH":
        st.warning("⚠️ HIGH ALERT: High predicted heart risk.")
    elif risk_level == "MODERATE":
        st.info("Moderate risk. Continue monitoring.")
    else:
        st.success("Low predicted risk.")


# ---------------------------------------------------------
# Tab 4: Live Simulation
# ---------------------------------------------------------

with tab4:
    st.subheader("Live Wearable Simulation")

    features = best_model["features"]

    st.write(
        "This simulates a wearable band by changing heart-related values such as heart rate, "
        "blood pressure, and ECG stress indicator over time."
    )

    simulation_seconds = st.slider("Simulation length in seconds", 5, 60, 20)

    base_patient = {feature: float(df[feature].median()) for feature in features}

    if st.button("▶️ Start Live Simulation", use_container_width=True):
        live_box = st.empty()
        chart_box = st.empty()

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

            risk_probability, risk_level, emergency_alert = predict_risk(best_model, live_patient)

            logs.append(
                {
                    "second": second + 1,
                    "risk_score": round(risk_probability * 100, 2),
                    "risk_level": risk_level,
                    "emergency_alert": emergency_alert,
                }
            )

            logs_df = pd.DataFrame(logs)

            with live_box.container():
                a, b, c = st.columns(3)
                a.metric("Second", second + 1)
                b.metric("Risk Score", f"{risk_probability * 100:.2f}%")
                c.metric("Alert", emergency_alert)

                if emergency_alert == "YES":
                    st.error("🚨 Demo emergency alert triggered.")
                else:
                    st.success("Monitoring normal.")

            chart_box.line_chart(logs_df.set_index("second")["risk_score"])

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
