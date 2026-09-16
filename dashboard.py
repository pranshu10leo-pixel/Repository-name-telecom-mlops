import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = Path("data/telecom_master.csv")

st.title("📊 Telecom Churn Dashboard")
st.write("Overview of the telecom customer churn dataset.")

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Customers", len(df))

with col2:
    st.metric("Features", len(df.columns) - 2)

with col3:
    st.metric("Churned", int(df["churn"].sum()))

with col4:
    st.metric("Churn Rate", f"{df['churn'].mean() * 100:.1f}%")

st.subheader("Churn Distribution")

churn_counts = df["churn"].value_counts().sort_index()
churn_counts.index = ["Stayed", "Churned"]

st.bar_chart(churn_counts)

st.subheader("Customer Data")

st.dataframe(df, width='stretch')

st.subheader("Numerical Feature Summary")

st.dataframe(df.describe(), width='stretch')
