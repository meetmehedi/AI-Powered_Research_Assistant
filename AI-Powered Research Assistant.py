# AI-Powered Research Assistant for Scientists

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="AI-Powered Research Assistant", layout="wide")

st.title("SciGenie 🧠 AI-Powered Research Assistant for Scientists")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
    
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("🧹 Data Cleaning Summary")
    st.write(f"Shape: {df.shape}")
    st.write("Missing Values per Column:")
    st.write(df.isnull().sum())

    # Optional: Auto drop columns with all nulls
    df.dropna(axis=1, how='all', inplace=True)

    # Step 2: Exploratory Data Analysis
    st.subheader("📈 Exploratory Data Analysis")
    col = st.selectbox("Select numeric column for histogram:", df.select_dtypes(include='number').columns)
    st.pyplot(sns.histplot(df[col], kde=True).figure)

    # Correlation heatmap
    st.subheader("📌 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # Step 3: Basic Model Suggestion
    st.subheader("🧠 Suggested Model Type")
    target = st.selectbox("Select Target Column:", df.columns)
    if df[target].dtype == "object" or df[target].nunique() < 10:
        st.success("✅ This looks like a classification task.")
    else:
        st.info("📈 This seems to be a regression problem.")
