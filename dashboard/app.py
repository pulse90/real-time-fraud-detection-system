import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use("Agg")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield — Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS — Dark Industrial Theme
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

/* ---- Global Reset ---- */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0c10;
    color: #e0e4ef;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #7a85a3;
    padding: 6px 0;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #00e5ff;
}

/* ---- Main area background ---- */
.main .block-container {
    background: #0a0c10;
    padding-top: 2rem;
}

/* ---- Page Title ---- */
h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #00e5ff, #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #c8cfe8;
    letter-spacing: -0.01em;
}

/* ---- Metric Cards ---- */
div[data-testid="stMetric"] {
    background: #111520;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00e5ff, #7c4dff);
    border-radius: 3px 0 0 3px;
}
div[data-testid="stMetricLabel"] > div {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a5270;
}
div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.9rem;
    color: #e8ecff;
}

/* ---- DataFrames ---- */
div[data-testid="stDataFrame"] {
    border: 1px solid #1e2535;
    border-radius: 10px;
    overflow: hidden;
}

/* ---- Buttons ---- */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00e5ff22, #7c4dff22);
    border: 1px solid #00e5ff55;
    border-radius: 8px;
    color: #00e5ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 0.6rem 1.8rem;
    transition: all 0.25s;
    letter-spacing: 0.05em;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #00e5ff44, #7c4dff44);
    border-color: #00e5ff;
    box-shadow: 0 0 16px #00e5ff44;
    color: #ffffff;
}

/* ---- Dividers ---- */
hr {
    border: none;
    border-top: 1px solid #1e2330;
    margin: 1.5rem 0;
}

/* ---- Slider ---- */
div[data-testid="stSlider"] label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #4a5270;
    letter-spacing: 0.05em;
}

/* ---- Text input ---- */
div[data-testid="stTextInput"] input {
    background: #111520;
    border: 1px solid #1e2535;
    border-radius: 8px;
    color: #e0e4ef;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}

/* ---- Sidebar brand ---- */
.brand-header {
    padding: 1rem 0 0.5rem 0;
    text-align: center;
}
.brand-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    background: linear-gradient(90deg, #00e5ff, #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #2e3650;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ---- Status badge ---- */
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00ff88;
    margin-right: 6px;
    box-shadow: 0 0 6px #00ff88;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---- Risk Tier Badge ---- */
.risk-critical { color: #ff4444; font-weight: 700; }
.risk-suspicious { color: #ffaa00; font-weight: 700; }
.risk-clear { color: #00e5a0; font-weight: 700; }

/* ---- Section subheader line ---- */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2e3650;
    margin-bottom: 0.8rem;
    border-left: 3px solid #7c4dff;
    padding-left: 10px;
}

/* ---- Success / Error alerts ---- */
div[data-testid="stAlert"] {
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

/* ---- Matplotlib figure background ---- */
.stPyplot > div {
    background: #111520 !important;
    border-radius: 10px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model_artifacts()

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "..", "data", "train_transaction.csv")
    return pd.read_csv(data_path)

df = load_data()

# =========================================================
# PREPARE NUMERIC INPUT DATAFRAME (reused across pages)
# =========================================================

@st.cache_data
def prepare_input_df(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Drops the target column, selects numeric columns, fills NaNs,
    and reindexes to match training feature columns.
    """
    drop_cols = [c for c in ["isFraud"] if c in df.columns]
    input_df = df.drop(columns=drop_cols)
    input_df = input_df.select_dtypes(include=np.number)
    input_df = input_df.fillna(0)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df

# =========================================================
# PLOTLY TEMPLATE — matches dark theme
# =========================================================

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#111520",
        plot_bgcolor="#111520",
        font=dict(family="Syne, sans-serif", color="#7a85a3", size=12),
        title=dict(font=dict(color="#c8cfe8", size=16, family="Syne, sans-serif")),
        xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", tickcolor="#4a5270"),
        yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", tickcolor="#4a5270"),
        colorway=["#00e5ff", "#7c4dff", "#ff4444", "#00ff88", "#ffaa00"],
        legend=dict(bgcolor="#0f1117", bordercolor="#1e2535", borderwidth=1),
    )
)

def apply_theme(fig):
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">🛡️ FraudShield</div>
        <div class="brand-sub">Detection Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Transaction Explorer", "Risk Analytics", "SHAP Explainer"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    total = len(df)
    frauds = int(df["isFraud"].sum())
    rate = frauds / total * 100

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#2e3650;
                letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">
        Live Stats
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#4a5270; line-height:2">
        <span class="status-dot"></span>System Online<br>
        📦 {total:,} records loaded<br>
        🚨 {frauds:,} fraud cases<br>
        📈 {rate:.2f}% fraud rate
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.title("Fraud Detection Dashboard")
    st.markdown('<div class="section-label">Real-time transaction intelligence</div>', unsafe_allow_html=True)

    total_transactions = len(df)
    total_fraud = int(df["isFraud"].sum())
    fraud_rate = (total_fraud / total_transactions) * 100
    avg_fraud_amount = df[df["isFraud"] == 1]["TransactionAmt"].mean()
    total_value_at_risk = df[df["isFraud"] == 1]["TransactionAmt"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{total_transactions:,}")
    c2.metric("Fraud Cases", f"{total_fraud:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")
    c5.metric("Value at Risk", f"${total_value_at_risk:,.0f}")

    st.markdown("---")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Transaction Split")
        fraud_counts = df["isFraud"].value_counts()
        fig_pie = px.pie(
            values=fraud_counts.values,
            names=["Non-Fraud", "Fraud"],
            hole=0.6,
            color_discrete_sequence=["#00e5ff", "#ff4444"],
        )
        fig_pie.update_traces(textfont_family="JetBrains Mono", textfont_size=11)
        fig_pie = apply_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Transaction Amount Distribution")
        fig_hist = px.histogram(
            df,
            x="TransactionAmt",
            color="isFraud",
            nbins=80,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={0: "#00e5ff", 1: "#ff4444"},
            labels={"isFraud": "Label", "TransactionAmt": "Amount ($)"},
        )
        fig_hist.update_layout(legend=dict(
            title="",
            itemsizing="constant",
        ))
        fig_hist.for_each_trace(lambda t: t.update(
            name="Non-Fraud" if t.name == "0" else "Fraud"
        ))
        fig_hist = apply_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Card type breakdown if available
    if "card4" in df.columns:
        st.markdown("---")
        st.subheader("Fraud by Card Network")
        card_fraud = df.groupby("card4")["isFraud"].agg(["sum", "count"]).reset_index()
        card_fraud.columns = ["Card Network", "Fraud Count", "Total"]
        card_fraud["Fraud Rate (%)"] = (card_fraud["Fraud Count"] / card_fraud["Total"] * 100).round(2)
        card_fraud = card_fraud.sort_values("Fraud Rate (%)", ascending=False)

        fig_bar = px.bar(
            card_fraud,
            x="Card Network",
            y="Fraud Rate (%)",
            color="Fraud Rate (%)",
            color_continuous_scale=["#00e5ff", "#7c4dff", "#ff4444"],
            text="Fraud Rate (%)",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar = apply_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# TRANSACTION EXPLORER
# =========================================================

elif page == "Transaction Explorer":

    st.title("Transaction Explorer")
    st.markdown('<div class="section-label">Search · Filter · Inspect</div>', unsafe_allow_html=True)

    col_search, col_slider = st.columns([1, 2])

    with col_search:
        search_id = st.text_input("🔎  Transaction ID", placeholder="e.g. 2987004")

    with col_slider:
        max_val = int(df["TransactionAmt"].max())
        min_amt, max_amt = st.slider(
            "Transaction Amount Range ($)",
            0, max_val, (0, max_val)
        )

    filtered_df = df[
        (df["TransactionAmt"] >= min_amt) &
        (df["TransactionAmt"] <= max_amt)
    ]

    # Color-code isFraud
    if "isFraud" in filtered_df.columns:
        n_fraud = int(filtered_df["isFraud"].sum())
        n_total = len(filtered_df)
        frate = n_fraud / n_total * 100 if n_total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Filtered Records", f"{n_total:,}")
        m2.metric("Fraud in Filter", f"{n_fraud:,}")
        m3.metric("Filtered Fraud Rate", f"{frate:.2f}%")

    st.markdown("---")

    if search_id:
        try:
            tid = int(search_id)
            transaction = filtered_df[filtered_df["TransactionID"] == tid]
            if len(transaction) > 0:
                st.success(f"✅ Transaction {tid} found")
                st.dataframe(transaction, use_container_width=True)
            else:
                st.error(f"❌ Transaction ID {tid} not found in current filter")
        except ValueError:
            st.error("⚠️  Please enter a valid numeric Transaction ID")

    st.subheader("Sample Records")
    st.markdown('<div class="section-label">First 200 rows of filtered dataset</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered_df.head(200).reset_index(drop=True),
        use_container_width=True,
        height=420
    )

# =========================================================
# RISK ANALYTICS
# =========================================================

elif page == "Risk Analytics":

    st.title("Risk Analytics")
    st.markdown('<div class="section-label">Scoring · Tiers · Temporal Patterns</div>', unsafe_allow_html=True)

    # Use real model probabilities on a sample for performance
    sample_size = min(5000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42).copy()

    input_df = prepare_input_df(sample_df, feature_columns)
    scaled = scaler.transform(input_df)
    probs = model.predict_proba(scaled)[:, 1]
    sample_df = sample_df.reset_index(drop=True)
    sample_df["FraudProbability"] = probs

    sample_df["RiskTier"] = pd.cut(
        sample_df["FraudProbability"],
        bins=[0, 0.4, 0.75, 1.0],
        labels=["Clear", "Suspicious", "Critical"],
        include_lowest=True
    )

    # KPI row
    tier_counts = sample_df["RiskTier"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢  Clear", f"{tier_counts.get('Clear', 0):,}")
    c2.metric("🟡  Suspicious", f"{tier_counts.get('Suspicious', 0):,}")
    c3.metric("🔴  Critical", f"{tier_counts.get('Critical', 0):,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Tier Distribution")
        fig_pie = px.pie(
            names=tier_counts.index,
            values=tier_counts.values,
            hole=0.55,
            color=tier_counts.index,
            color_discrete_map={
                "Clear": "#00e5a0",
                "Suspicious": "#ffaa00",
                "Critical": "#ff4444"
            },
        )
        fig_pie.update_traces(textfont_family="JetBrains Mono", textfont_size=11)
        fig_pie = apply_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Amount by Risk Tier")
        fig_box = px.box(
            sample_df,
            x="RiskTier",
            y="TransactionAmt",
            color="RiskTier",
            color_discrete_map={
                "Clear": "#00e5a0",
                "Suspicious": "#ffaa00",
                "Critical": "#ff4444"
            },
            category_orders={"RiskTier": ["Clear", "Suspicious", "Critical"]},
            labels={"TransactionAmt": "Amount ($)", "RiskTier": "Tier"},
        )
        fig_box = apply_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    # Fraud by Hour
    if "TransactionDT" in df.columns:
        st.markdown("---")
        st.subheader("Fraud Rate by Hour of Day")

        hourly_df = df.copy()
        hourly_df["Hour"] = (hourly_df["TransactionDT"] // 3600) % 24
        hourly = hourly_df.groupby("Hour")["isFraud"].mean().reset_index()
        hourly.columns = ["Hour", "Fraud Rate"]
        hourly["Fraud Rate %"] = hourly["Fraud Rate"] * 100

        fig_line = px.line(
            hourly,
            x="Hour",
            y="Fraud Rate %",
            markers=True,
            labels={"Hour": "Hour of Day (UTC)", "Fraud Rate %": "Fraud Rate (%)"},
            color_discrete_sequence=["#7c4dff"],
        )
        fig_line.add_scatter(
            x=hourly["Hour"],
            y=hourly["Fraud Rate %"],
            mode="markers",
            marker=dict(size=8, color="#00e5ff", symbol="circle"),
            name="Data Point"
        )
        fig_line.update_traces(line=dict(width=2.5))
        fig_line = apply_theme(fig_line)
        st.plotly_chart(fig_line, use_container_width=True)

    # Fraud probability histogram
    st.markdown("---")
    st.subheader("Model Score Distribution")
    fig_score = px.histogram(
        sample_df,
        x="FraudProbability",
        color="RiskTier",
        nbins=60,
        barmode="stack",
        color_discrete_map={
            "Clear": "#00e5a0",
            "Suspicious": "#ffaa00",
            "Critical": "#ff4444"
        },
        labels={"FraudProbability": "Fraud Probability Score"},
        category_orders={"RiskTier": ["Clear", "Suspicious", "Critical"]},
    )
    fig_score = apply_theme(fig_score)
    st.plotly_chart(fig_score, use_container_width=True)

# =========================================================
# SHAP EXPLAINER
# =========================================================

elif page == "SHAP Explainer":

    st.title("Explainable AI — SHAP")
    st.markdown('<div class="section-label">Model transparency · Feature attribution</div>', unsafe_allow_html=True)

    st.info(
        "Select a transaction row and click **Generate Explanation** to see "
        "which features drove the model's fraud prediction for that record.",
        icon="🧠"
    )

    row_number = st.slider(
        "Transaction Row Number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        help="Index into the loaded dataset"
    )

    if st.button("⚡  Generate Explanation"):

        with st.spinner("Computing SHAP values — this may take a moment…"):

            try:
                # -------------------------------------------------
                # 1. Prepare the full input matrix (needed for SHAP
                #    background) and isolate the selected row.
                # -------------------------------------------------
                input_df = prepare_input_df(df, feature_columns)

                # Scale all rows
                scaled_all = scaler.transform(input_df)

                # Selected row (as 2-D array for predict)
                row_scaled = scaled_all[row_number:row_number + 1]
                row_raw    = input_df.iloc[row_number]

                # -------------------------------------------------
                # 2. Fraud probability for the selected row
                # -------------------------------------------------
                fraud_prob = model.predict_proba(row_scaled)[0, 1]

                col_prob, col_label = st.columns(2)
                col_prob.metric("🔴  Fraud Score", f"{fraud_prob:.4f}")
                col_label.metric(
                    "Verdict",
                    "FRAUD" if fraud_prob >= 0.5 else "LEGITIMATE",
                )

                st.markdown("---")

                # -------------------------------------------------
                # 3. Build SHAP explainer using a background sample
                #    to keep computation fast.
                # -------------------------------------------------
                background_size = min(200, len(scaled_all))
                rng = np.random.default_rng(42)
                bg_idx = rng.choice(len(scaled_all), size=background_size, replace=False)
                background = scaled_all[bg_idx]

                # TreeExplainer is preferred for tree-based models
                # (XGBoost, LightGBM, RandomForest, etc.)
                try:
                    explainer = shap.TreeExplainer(
                        model,
                        data=background,
                        feature_perturbation="interventional",
                        model_output="probability",   # output in probability space
                    )
                    shap_values = explainer.shap_values(row_scaled)
                    expected_value = explainer.expected_value
                except Exception:
                    # Fallback: KernelExplainer (model-agnostic, slower)
                    st.warning("TreeExplainer unavailable — using KernelExplainer (slower).")
                    explainer = shap.KernelExplainer(
                        lambda x: model.predict_proba(x)[:, 1],
                        background[:50],
                    )
                    shap_values = explainer.shap_values(row_scaled, nsamples=100)
                    expected_value = explainer.expected_value

                # -------------------------------------------------
                # 4. Normalise shap_values shape:
                #    Tree models for binary classification can return
                #    list-of-arrays (one per class) or a 2-D array.
                #    We always want the POSITIVE CLASS (fraud) values
                #    for the single selected row → shape (n_features,)
                # -------------------------------------------------
                if isinstance(shap_values, list):
                    # Multi-class / two-class list → take class 1
                    sv_row = shap_values[1][0]
                    ev = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
                elif shap_values.ndim == 3:
                    # (1, n_features, n_classes) — some tree models
                    sv_row = shap_values[0, :, 1]
                    ev = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
                elif shap_values.ndim == 2:
                    # (1, n_features) — already positive-class or single output
                    sv_row = shap_values[0]
                    ev = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
                else:
                    sv_row = shap_values.ravel()
                    ev = float(np.atleast_1d(expected_value)[-1])

                ev = float(np.atleast_1d(ev)[-1])

                # -------------------------------------------------
                # 5. SHAP Waterfall Plot
                # -------------------------------------------------
                st.subheader("SHAP Waterfall Plot")
                st.caption("Each bar shows how much a feature pushed the score up (red) or down (blue) from the baseline.")

                explanation = shap.Explanation(
                    values=sv_row,
                    base_values=ev,
                    data=row_raw.values,
                    feature_names=list(input_df.columns),
                )

                # Dark-mode matplotlib style
                plt.rcParams.update({
                    "figure.facecolor":  "#111520",
                    "axes.facecolor":    "#111520",
                    "axes.edgecolor":    "#1e2535",
                    "axes.labelcolor":   "#7a85a3",
                    "xtick.color":       "#4a5270",
                    "ytick.color":       "#4a5270",
                    "text.color":        "#c8cfe8",
                    "font.family":       "monospace",
                })

                fig_wf, ax_wf = plt.subplots(figsize=(12, 7))
                shap.plots.waterfall(explanation, max_display=15, show=False)
                fig_wf.patch.set_facecolor("#111520")
                plt.tight_layout()
                st.pyplot(fig_wf)
                plt.close(fig_wf)

                # -------------------------------------------------
                # 6. Bar chart — top contributing features
                # -------------------------------------------------
                st.subheader("Top Feature Contributions")
                n_top = 15
                top_idx = np.argsort(np.abs(sv_row))[-n_top:][::-1]
                top_features = [input_df.columns[i] for i in top_idx]
                top_values   = [sv_row[i] for i in top_idx]

                bar_colors = ["#ff4444" if v > 0 else "#00e5ff" for v in top_values]

                fig_bar = go.Figure(go.Bar(
                    x=top_values,
                    y=top_features,
                    orientation="h",
                    marker_color=bar_colors,
                    marker_line_width=0,
                    text=[f"{v:+.4f}" for v in top_values],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=11, color="#c8cfe8"),
                ))
                fig_bar.update_layout(
                    template=PLOTLY_TEMPLATE,
                    xaxis_title="SHAP Value (impact on fraud probability)",
                    yaxis=dict(autorange="reversed"),
                    height=420,
                    margin=dict(l=10, r=40, t=10, b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # -------------------------------------------------
                # 7. Plain-English Summary
                # -------------------------------------------------
                st.subheader("Plain-English Explanation")

                top5_idx = np.argsort(np.abs(sv_row))[-5:][::-1]
                lines = []
                for i in top5_idx:
                    fname = input_df.columns[i]
                    fval  = row_raw.iloc[i]
                    sval  = sv_row[i]
                    direction = "increased" if sval > 0 else "decreased"
                    lines.append(
                        f"• **{fname}** = `{fval:.4g}` → {direction} fraud risk "
                        f"by `{abs(sval):.4f}` SHAP units"
                    )

                verdict_color = "🔴" if fraud_prob >= 0.5 else "🟢"
                st.markdown(
                    f"{verdict_color} **Model baseline fraud probability:** `{ev:.4f}`  \n"
                    f"{verdict_color} **Predicted fraud probability:** `{fraud_prob:.4f}`\n\n"
                    "**Top drivers for this transaction:**\n\n" + "\n".join(lines)
                )

                st.success("✅ SHAP explanation generated successfully.")

            except Exception as e:
                st.error(f"❌ Error generating SHAP explanation: {e}")
                st.exception(e)