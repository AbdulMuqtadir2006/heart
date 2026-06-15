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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


st.set_page_config(
    page_title="CardioGuard AI Agent",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ CardioGuard AI Agent")
st.write("AI model for heart-risk prediction, wearable simulation, and emergency alert demo.")

st.warning(
    "This is an educational prototype only. It is not a real medical diagnosis system."
)


@st.cache_data
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo

        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        y = heart.data.targets.copy()

        df = pd.concat([X, y], axis=1)
        df = df.rename(columns={y.columns[0]: "target"})

        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df["target"] = (df["target"] > 0).astype(int)

        return df

    except Exception:
        st.error("Could not download online data. Please upload a heart disease CSV file.")
        return None


def clean_data(df):
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")
    return df


def train_ai(df):
    df = clean_data(df)

    target_col = "target"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    y = (pd.to_numeric(y, errors="coerce") > 0).astype(int)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=42))
        ])
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        probability = model.predict_proba(X_test)[:, 1]
        threshold = 0.45
        prediction = (probability >= threshold).astype(int)

        result = {
            "model": name,
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1_score": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
            "threshold": threshold
        }

        results.append(result)

        trained_models[name] = {
            "model": model,
            "features": feature_names,
            "threshold": threshold,
            "confusion_matrix": confusion_matrix(y_test, prediction)
        }

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["recall", "roc_auc", "f1_score"],
        ascending=False
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    return results_df, best_model_name, best_model


def predict_risk(model_bundle, patient_data):
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

    return probability, level


def feature_importance(model_bundle):
    model = model_bundle["model"]
    features = model_bundle["features"]
    final_model = model.steps[-1][1]

    if hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
    else:
        importance = np.zeros(len(features))

    return pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=False)


# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("Step 1: Data")

data_option = st.sidebar.radio(
    "Choose data source",
    ["Auto-download UCI Heart Disease Data", "Upload CSV"]
)

df = None

if data_option == "Auto-download UCI Heart Disease Data":
    df = load_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    if "target" not in df.columns:
        st.error("Your CSV must contain a column named 'target'.")
        st.stop()

    st.sidebar.header("Step 2: Train")

    if st.sidebar.button("Train AI Model"):
        with st.spinner("Training AI models..."):
            results_df, best_model_name, best_model = train_ai(df)

            st.session_state["results_df"] = results_df
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_model"] = best_model

            joblib.dump(best_model, "cardioguard_model.joblib")

        st.success("Training complete!")

    if "best_model" in st.session_state:
        results_df = st.session_state["results_df"]
        best_model_name = st.session_state["best_model_name"]
        best_model = st.session_state["best_model"]

        st.subheader("AI Model Leaderboard")
        st.dataframe(results_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Best Model", best_model_name)
            st.metric("Risk Threshold", best_model["threshold"])

        with col2:
            st.write("Confusion Matrix")
            st.write(best_model["confusion_matrix"])

        st.subheader("Feature Importance")
        importance_df = feature_importance(best_model)
        st.dataframe(importance_df.head(10), use_container_width=True)
        st.bar_chart(importance_df.set_index("Feature").head(10))

        st.divider()

        st.subheader("Patient Risk Prediction")

        clean_df = clean_data(df)
        features = best_model["features"]

        patient_data = {}
        columns = st.columns(3)

        for i, feature in enumerate(features):
            minimum = float(clean_df[feature].min())
            maximum = float(clean_df[feature].max())
            median = float(clean_df[feature].median())

            if minimum == maximum:
                maximum = minimum + 1

            with columns[i % 3]:
                patient_data[feature] = st.slider(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=median
                )

        risk_probability, risk_level = predict_risk(best_model, patient_data)

        c1, c2, c3 = st.columns(3)

        c1.metric("Risk Score", f"{risk_probability * 100:.2f}%")
        c2.metric("Risk Level", risk_level)

        if risk_level in ["HIGH", "CRITICAL"]:
            c3.metric("Emergency Alert", "YES")
        else:
            c3.metric("Emergency Alert", "NO")

        if risk_level == "CRITICAL":
            st.error("🚨 CRITICAL ALERT: Very high predicted heart risk.")
        elif risk_level == "HIGH":
            st.warning("⚠️ HIGH ALERT: High predicted heart risk.")
        elif risk_level == "MODERATE":
            st.info("Moderate predicted risk. Continue monitoring.")
        else:
            st.success("Low predicted risk.")

        st.divider()

        st.subheader("Live Wearable Simulation")

        simulation_seconds = st.slider("Simulation length", 5, 60, 20)

        if st.button("Start Simulation"):
            placeholder = st.empty()
            chart_placeholder = st.empty()

            logs = []

            for second in range(simulation_seconds):
                live_patient = patient_data.copy()

                if "thalach" in live_patient:
                    live_patient["thalach"] = max(
                        60,
                        live_patient["thalach"] + np.random.randint(-10, 15)
                    )

                if "trestbps" in live_patient:
                    live_patient["trestbps"] = max(
                        80,
                        live_patient["trestbps"] + np.random.randint(-5, 8)
                    )

                if "oldpeak" in live_patient:
                    live_patient["oldpeak"] = max(
                        0,
                        live_patient["oldpeak"] + np.random.normal(0, 0.15)
                    )

                risk_probability, risk_level = predict_risk(best_model, live_patient)

                emergency = "YES" if risk_level in ["HIGH", "CRITICAL"] else "NO"

                logs.append({
                    "second": second + 1,
                    "risk_score": round(risk_probability * 100, 2),
                    "risk_level": risk_level,
                    "emergency_alert": emergency
                })

                logs_df = pd.DataFrame(logs)

                with placeholder.container():
                    st.write(f"Second: {second + 1}")
                    st.write(f"Risk Score: {risk_probability * 100:.2f}%")
                    st.write(f"Risk Level: {risk_level}")

                    if emergency == "YES":
                        st.error("🚨 Demo emergency alert triggered.")
                    else:
                        st.success("Monitoring normal.")

                chart_placeholder.line_chart(
                    logs_df.set_index("second")["risk_score"]
                )

                time.sleep(0.5)

            st.subheader("Simulation Log")
            st.dataframe(logs_df, use_container_width=True)

            csv = logs_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Simulation Log",
                data=csv,
                file_name="cardioguard_simulation_log.csv",
                mime="text/csv"
            )

else:
    st.info("Load data first.")
