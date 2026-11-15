from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data import generate_transactions, load_transactions
from src.graph import build_transaction_graph
from src.pipeline import train_and_evaluate

st.set_page_config(page_title="GraphShield", page_icon="🛡️", layout="wide")
st.title("🛡️ GraphShield")
st.caption("Graph-based transaction network and fraud intelligence | Synthetic-data demo")
uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
if uploaded:
    df = load_transactions(uploaded)
else:
    sample_path = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
    df = load_transactions(sample_path) if sample_path.exists() else generate_transactions()
st.sidebar.metric("Transactions", f"{len(df):,}")
st.sidebar.metric("Fraud labels", f"{int(df.is_fraud.sum()):,}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Transactions", f"{len(df):,}")
c2.metric("Accounts", f"{len(set(df.sender_id) | set(df.receiver_id)):,}")
c3.metric("Fraud rate", f"{df.is_fraud.mean():.1%}")
c4.metric("Total volume", f"{df.amount.sum():,.0f}")
left,right = st.columns(2)
with left:
    st.subheader("Transaction volume by channel")
    st.plotly_chart(px.histogram(df, x="channel", color="is_fraud", barmode="group"), use_container_width=True)
with right:
    st.subheader("Transaction amount distribution")
    st.plotly_chart(px.histogram(df, x="amount", color="is_fraud", nbins=35), use_container_width=True)
st.subheader("Account network overview")
g = build_transaction_graph(df)
st.write(f"Directed network: **{g.number_of_nodes():,} accounts** and **{g.number_of_edges():,} unique directed relationships**.")
st.dataframe(df.head(100), use_container_width=True)
if st.button("Train models and generate risk scores", type="primary"):
    with st.spinner("Training models..."):
        metrics, scored = train_and_evaluate(df)
    st.success("Training completed. Metrics are saved in reports/metrics.json.")
    st.subheader("Holdout metrics")
    st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
    st.subheader("Highest-scoring holdout transactions")
    st.dataframe(scored.sort_values("graphshield_probability", ascending=False).head(25),
                 use_container_width=True)
st.info("Prototype only. Synthetic labels and scores are not suitable for real payment decisions.")
