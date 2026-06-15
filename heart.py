import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

st.set_page_config(
    page_title="CardioGuard AI Agent",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ CardioGuard AI Agent")
st.caption("AI prototype for cardiovascular risk prediction, wearable simulation, and demo emergency alerts.")

st.warning(
    "Medical safety note: This is an educational prototype only. "
    "It is not a real medical device and must not be used for diagnosis or emergency decisions."
)


@st.cache_data
def load_uci_heart_data():
    try:
        from ucimlrepo import fetch_ucirepo

        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        y = heart.data.targets.copy()

        target_col = y.columns[0]
        df = pd.concat([X, y], axis=1)
        df = df.rename(columns={target_col: "target"})

        # UCI target: 0 = no disease, 1-4 = disease presence
        df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)
        return df

    except Exception as e:
        st.error("Could not download UCI data. Upload your own CSV instead.")
        st.exception(e)
        return None


def clean_dataframe(df):
    df = df.copy()

    # Convert columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove empty rows
    df = df.dropna(how="all")

    return df


def get_target_column(df):
    possible_targets = ["target", "num", "heart_disease", "condition", "output"]

    for col in possible_targets:
        if col in df.columns:
            return col

    return df.columns[-1]


def choose_threshold(y_true, probabilities, desired_recall=0.85):
    best_threshold = 0.50
    best_f1 = -1

    thresholds = np.arange(0.10, 0.91, 0.01)

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # In medical screening demos, prioritize recall
        if rec >= desired_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold)


def train_models(df, target_col):
    df = clean_dataframe(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert target to binary if needed
    y = (pd.to_numeric(y, errors="coerce") > 0).astype(int)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", numeric_preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    max_depth=6
                ))
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42))
            ]
        )
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
        else:
            probabilities = model.predict(X_test)

        threshold = choose_threshold(y_test, probabilities)
        preds = (probabilities >= threshold).astype(int)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "threshold": threshold,
        }

        results.append(row)
        trained[name] = {
            "pipeline": model,
            "threshold": threshold,
            "features": feature_names,
            "confusion_matrix": confusion_matrix(y_test, preds),
        }

    results_df = pd.DataFrame(results).sort_values(
        by=["recall", "roc_auc", "f1"],
        ascending=False
    )

    best_name = results_df.iloc[0]["model"]
    best_bundle = trained[best_name]

    return results_df, best_name, best_bundle


def predict_risk(model_bundle, input_row):
    model = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]
    features = model_bundle["features"]

    input_df = pd.DataFrame([input_row])
    input_df = input_df[features]

    probability = float(model.predict_proba(input_df)[:, 1][0])
    label = int(probability >= threshold)

    if probability >= 0.80:
        level = "CRITICAL"
    elif probability >= threshold:
        level = "HIGH"
    elif probability >= 0.40:
        level = "MODERATE"
    else:
        level = "LOW"

    return probability, label, level


def get_feature_importance(model_bundle):
    pipeline = model_bundle["pipeline"]
    features = model_bundle["features"]

    last_model = pipeline.steps[-1][1]

    if hasattr(last_model, "feature_importances_"):
        importances = last_model.feature_importances_
        return pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values("importance", ascending=False)

    return pd.DataFrame({
        "feature": features,
        "importance": np.zeros(len(features))
    })


def simulate_patient(base_patient, model_bundle, seconds=20):
    logs = []

    for t in range(seconds):
        patient = base_patient.copy()

        # Simulate wearable-like changes using available UCI columns
        if "thalach" in patient:
            patient["thalach"] = max(70, patient["thalach"] + np.random.randint(-8, 12))

        if "trestbps" in patient:
            patient["trestbps"] = max(80, patient["trestbps"] + np.random.randint(-5, 7))

        if "oldpeak" in patient:
            patient["oldpeak"] = max(0, patient["oldpeak"] + np.random.normal(0, 0.15))

        probability, label, level = predict_risk(model_bundle, patient)

        logs.append({
            "second": t + 1,
            "risk_score": probability,
            "risk_percent": round(probability * 100, 2),
            "alert_level": level,
            "emergency_alert": "YES" if level in ["HIGH", "CRITICAL"] else "NO",
        })

    return pd.DataFrame(logs)


# -----------------------------
# Sidebar: data source
# -----------------------------

st.sidebar.header("1. Data Source")

data_choice = st.sidebar.radio(
    "Choose data source",
    ["Use UCI Heart Disease Dataset", "Upload my own CSV"]
)

df = None

if data_choice == "Use UCI Heart Disease Dataset":
    df = load_uci_heart_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_col = get_target_column(df)

    st.sidebar.header("2. Training")
    target_col = st.sidebar.selectbox(
        "Target column",
        options=df.columns,
        index=list(df.columns).index(target_col)
    )

    train_button = st.sidebar.button("Train AI Agent")

    if train_button:
        with st.spinner("Training models and selecting best agent..."):
            results_df, best_name, best_bundle = train_models(df, target_col)

            st.session_state["results_df"] = results_df
            st.session_state["best_name"] = best_name
            st.session_state["best_bundle"] = best_bundle

            joblib.dump(best_bundle, "cardioguard_model.joblib")

        st.success(f"Training complete. Best model: {best_name}")

    if "best_bundle" in st.session_state:
        results_df = st.session_state["results_df"]
        best_name = st.session_state["best_name"]
        best_bundle = st.session_state["best_bundle"]

        st.subheader("Model Leaderboard")
        st.dataframe(results_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Selected Model", best_name)
            st.metric("Risk Threshold", round(best_bundle["threshold"], 2))

        with col2:
            st.write("Confusion Matrix")
            st.write(best_bundle["confusion_matrix"])

        st.subheader("Feature Importance / Explanation")
        importance_df = get_feature_importance(best_bundle)
        st.dataframe(importance_df.head(10), use_container_width=True)
        st.bar_chart(importance_df.set_index("feature").head(10))

        st.divider()
        st.subheader("3. Play With Patient Data")

        features = best_bundle["features"]
        clean_df = clean_dataframe(df)

        input_row = {}

        cols = st.columns(3)

        for i, feature in enumerate(features):
            series = clean_df[feature]
            median = float(series.median())
            min_val = float(series.min())
            max_val = float(series.max())

            if min_val == max_val:
                max_val = min_val + 1

            with cols[i % 3]:
                input_row[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=median
                )

        probability, label, level = predict_risk(best_bundle, input_row)

        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{probability * 100:.2f}%")
        c2.metric("Risk Level", level)
        c3.metric("Emergency Demo Alert", "YES" if level in ["HIGH", "CRITICAL"] else "NO")

        if level == "CRITICAL":
            st.error("🚨 CRITICAL DEMO ALERT: Very high predicted risk. In a real system, emergency contacts would be notified.")
        elif level == "HIGH":
            st.warning("⚠️ HIGH DEMO ALERT: The patient should seek medical attention.")
        elif level == "MODERATE":
            st.info("Moderate risk. Continue monitoring.")
        else:
            st.success("Low predicted risk.")

        st.divider()
        st.subheader("4. Live Wearable Simulation")

        sim_seconds = st.slider("Simulation length in seconds", 5, 60, 20)

        if st.button("Start Live Simulation"):
            placeholder = st.empty()
            chart_placeholder = st.empty()

            logs = []

            for t in range(sim_seconds):
                patient = input_row.copy()

                if "thalach" in patient:
                    patient["thalach"] = max(70, patient["thalach"] + np.random.randint(-8, 12))

                if "trestbps" in patient:
                    patient["trestbps"] = max(80, patient["trestbps"] + np.random.randint(-5, 7))

                if "oldpeak" in patient:
                    patient["oldpeak"] = max(0, patient["oldpeak"] + np.random.normal(0, 0.15))

                probability, label, level = predict_risk(best_bundle, patient)

                logs.append({
                    "second": t + 1,
                    "risk_score": probability,
                    "risk_percent": round(probability * 100, 2),
                    "alert_level": level,
                    "emergency_alert": "YES" if level in ["HIGH", "CRITICAL"] else "NO",
                })

                latest = logs[-1]

                with placeholder.container():
                    st.write(f"Second: {latest['second']}")
                    st.write(f"Risk: {latest['risk_percent']}%")
                    st.write(f"Alert level: {latest['alert_level']}")

                    if latest["emergency_alert"] == "YES":
                        st.error("🚨 Demo emergency alert triggered.")
                    else:
                        st.success("Monitoring normal.")

                chart_df = pd.DataFrame(logs)
                chart_placeholder.line_chart(chart_df.set_index("second")["risk_percent"])

                time.sleep(0.5)

            final_logs = pd.DataFrame(logs)
            st.subheader("Simulation Log")
            st.dataframe(final_logs, use_container_width=True)

            csv = final_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Simulation Log",
                data=csv,
                file_name="cardioguard_simulation_log.csv",
                mime="text/csv"
            )

else:
    st.info("Load a dataset or upload a CSV to begin.")
