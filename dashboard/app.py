import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "preprocessor.pkl")
feature_path = os.path.join(BASE_DIR, "feature_columns.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_columns = joblib.load(feature_path)

# =========================================================
# LOAD DATA
# =========================================================

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    import os

    BASE_DIR = os.path.dirname(__file__)

    data_path = os.path.join(
        BASE_DIR,
        "..",
        "data",
        "train_transaction.csv"
    )

    df = pd.read_csv(data_path)

    return df

# LOAD DATAFRAME
df = load_data()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💳 Fraud Detection System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Transaction Explorer",
        "Risk Analytics",
        "SHAP Explainer"
    ]
)

st.sidebar.markdown("---")

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.title("📊 Fraud Detection Dashboard")

    total_transactions = len(df)
    total_fraud = df["isFraud"].sum()
    fraud_rate = (total_fraud / total_transactions) * 100
    avg_fraud_amount = df[df["isFraud"] == 1]["TransactionAmt"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Cases",
        f"{total_fraud:,}"
    )

    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    col4.metric(
        "Avg Fraud Amount",
        f"${avg_fraud_amount:.2f}"
    )

    st.markdown("---")

    st.subheader("Fraud Distribution")

    fraud_counts = df["isFraud"].value_counts()

    fig = px.pie(
        values=fraud_counts.values,
        names=["Non Fraud", "Fraud"],
        hole=0.5,
        title="Fraud vs Non-Fraud Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction Amount Distribution")

    fig2 = px.histogram(
        df,
        x="TransactionAmt",
        color="isFraud",
        nbins=100,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# TRANSACTION EXPLORER
# =========================================================

elif page == "Transaction Explorer":

    st.title("🔍 Transaction Explorer")

    search_id = st.text_input(
        "Enter Transaction ID"
    )

    min_amt = st.slider(
        "Minimum Transaction Amount",
        0,
        int(df["TransactionAmt"].max()),
        0
    )

    filtered_df = df[
        df["TransactionAmt"] >= min_amt
    ]

    if search_id:

        try:
            transaction = filtered_df[
                filtered_df["TransactionID"] == int(search_id)
            ]

            if len(transaction) > 0:

                st.success("Transaction Found")

                st.dataframe(transaction)

            else:
                st.error("Transaction ID Not Found")

        except:
            st.error("Invalid Transaction ID")

    st.markdown("### Sample Transactions")

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )

# =========================================================
# RISK ANALYTICS
# =========================================================

elif page == "Risk Analytics":

    st.title("⚠️ Risk Analytics")

    # Simulated probabilities
    np.random.seed(42)

    df["FraudProbability"] = np.random.rand(len(df))

    df["RiskTier"] = pd.cut(

    df["FraudProbability"],

    bins=[0, 0.4, 0.75, 1],

    labels=[
        "Clear",
        "Suspicious",
        "Critical"
    ]
)

    risk_counts = df["RiskTier"].value_counts()

    st.subheader("Risk Tier Distribution")

    fig = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        hole=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Tier vs Transaction Amount")

    fig2 = px.box(
        df,
        x="RiskTier",
        y="TransactionAmt",
        color="RiskTier"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Fraud by Hour")

    if "TransactionDT" in df.columns:

        df["Hour"] = (
            (df["TransactionDT"] // 3600) % 24
        )

        hourly = df.groupby("Hour")["isFraud"].mean().reset_index()

        fig3 = px.line(
            hourly,
            x="Hour",
            y="isFraud",
            markers=True,
            title="Fraud Rate by Hour"
        )

        st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# SHAP EXPLAINER
# =========================================================

elif page == "SHAP Explainer":

    st.title("🧠 Explainable AI — SHAP")

    st.markdown("""
    Enter a transaction row number to generate
    SHAP explanation for fraud prediction.
    """)

    row_number = st.number_input(
        "Transaction Row Number",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    if st.button("Generate Explanation"):

        try:

            input_df = df.drop(columns=["isFraud"])

            input_df = input_df.select_dtypes(
                include=np.number
            )

            input_df = input_df.fillna(0)

            input_df = input_df.reindex(
                columns=feature_columns,
                fill_value=0
            )

            scaled_input = scaler.transform(input_df)

            prediction_prob = model.predict_proba(
                scaled_input
            )[:, 1]

            st.subheader("Fraud Probability")

            st.metric(
                "Fraud Score",
                f"{prediction_prob[row_number]:.4f}"
            )

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(
                scaled_input[row_number:row_number+1]
            )

            st.subheader("SHAP Waterfall Plot")

            fig, ax = plt.subplots(figsize=(10, 5))

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_df.iloc[row_number],
                    feature_names=input_df.columns
                ),
                show=False
            )

            st.pyplot(fig)

            st.success("""
            Explanation Generated Successfully.
            Features pushing prediction higher
            indicate fraud risk signals.
            """)

        except Exception as e:

            st.error(f"Error: {e}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    ### Real-Time Fraud Detection System
    Built using:
    - LightGBM
    - XGBoost
    - SHAP Explainability
    - Streamlit
    """
)