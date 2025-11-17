
import streamlit as st, joblib, json, pandas as pd, numpy as np, shap, plotly.express as px, matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Retention Risk Dashboard")
st.title("📈 Retention Risk Dashboard")

art = Path("artifacts")
if not (art/"model.pkl").exists():
    st.warning("No model found. Run `python -m src.train` first.")
else:
    model = joblib.load(art/"model.pkl")
    st.success("Model loaded.")

    st.subheader("Upload CSV for scoring (optional)")
    file = st.file_uploader("CSV with columns f0...f9", type="csv")
    if file:
        df = pd.read_csv(file)
    else:
        st.caption("Using random demo data")
        df = pd.DataFrame(np.random.randn(200, 10), columns=[f"f{i}" for i in range(10)])

    proba = model.predict_proba(df)[:,1]
    df_scores = df.copy(); df_scores["risk_score"] = proba
    st.write(df_scores.head())

    fig = plt.figure()
    plt.hist(df_scores["risk_score"], bins=30)
    plt.title("Risk Score Distribution")
    st.pyplot(fig)

    st.subheader("Explainability (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_scores[[c for c in df_scores.columns if c.startswith('f')]].head(200))
    st.caption("Summary plot for the first 200 rows")
    st.pyplot(shap.summary_plot(shap_values[1], df_scores[[c for c in df_scores.columns if c.startswith('f')]].head(200), show=False))
