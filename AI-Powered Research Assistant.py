# AI-Powered Research Assistant for Scientists

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
from fpdf import FPDF

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import autosklearn.classification
    import autosklearn.regression
except ImportError:
    autosklearn = None

import os
import openai

st.set_page_config(page_title="AI-Powered Research Assistant", layout="wide")

st.title(" SciGene 🧠")
st.caption("AI-Powered Research Assistant for Scientists")

# --- Helper Functions ---
def auto_clean(df):
    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()
    df.fillna(method='ffill', inplace=True)
    return df

def train_models(X, y, task):
    results = {}

    if task == 'classification':
        rf = RandomForestClassifier()
        rf.fit(X, y)
        results['RandomForest'] = (rf, accuracy_score(y, rf.predict(X)))

        if xgb:
            xgb_model = xgb.XGBClassifier()
            xgb_model.fit(X, y)
            results['XGBoost'] = (xgb_model, accuracy_score(y, xgb_model.predict(X)))

        if autosklearn:
            ask = autosklearn.classification.AutoSklearnClassifier()
            ask.fit(X, y)
            results['AutoSklearn'] = (ask, accuracy_score(y, ask.predict(X)))

    else:
        rf = RandomForestRegressor()
        rf.fit(X, y)
        results['RandomForest'] = (rf, r2_score(y, rf.predict(X)))

        if xgb:
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X, y)
            results['XGBoost'] = (xgb_model, r2_score(y, xgb_model.predict(X)))

        if autosklearn:
            ask = autosklearn.regression.AutoSklearnRegressor()
            ask.fit(X, y)
            results['AutoSklearn'] = (ask, r2_score(y, ask.predict(X)))

    best_model_name = max(results, key=lambda k: results[k][1])
    return best_model_name, results[best_model_name][0], results[best_model_name][1]

def generate_abstract(df, task, model_name):
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
        prompt = f"""
        Write an academic abstract for a research project based on the following:
        - Dataset with columns: {list(df.columns)}
        - Task type: {task}
        - Model used: {model_name}
        - Key insights from EDA and modeling.
        Keep it within 150 words.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    return "GPT-4 abstract not generated. Set OPENAI_API_KEY."

def make_pdf(summary, abstract):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.ln()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Abstract", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, abstract)
    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# --- Main App ---
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)

    df = auto_clean(df)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("🧹 Data Cleaning Summary")
    st.write(f"Shape: {df.shape}")
    st.write("Missing Values per Column:")
    st.write(df.isnull().sum())

    st.subheader("📈 Exploratory Data Analysis")
    col = st.selectbox("Select numeric column for histogram:", df.select_dtypes(include='number').columns)
    st.pyplot(sns.histplot(df[col], kde=True).figure)

    st.subheader("📌 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    st.subheader("🧠 Suggested Model Type")
    target = st.selectbox("Select Target Column:", df.columns)
    if df[target].dtype == "object" or df[target].nunique() < 10:
        task = 'classification'
        st.success("✅ This looks like a classification task.")
    else:
        task = 'regression'
        st.info("📈 This seems to be a regression problem.")

    if st.button("🚀 Run AutoML"):
        X = df.drop(columns=[target])
        X = pd.get_dummies(X)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_name, model, score = train_models(X_train, y_train, task)
        st.success(f"✅ Best model: {model_name} | Score: {score:.4f}")

        y_pred = model.predict(X_test)
        if task == 'classification':
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        else:
            st.write(f"R² Score on Test Set: {r2_score(y_test, y_pred):.4f}")

        joblib.dump(model, "trained_model.pkl")
        st.download_button("📥 Download Model (.pkl)", data=open("trained_model.pkl", "rb"), file_name="model.pkl")

        abstract = generate_abstract(df, task, model_name)
        st.subheader("📝 Abstract")
        st.write(abstract)

        summary_text = f"Target: {target}\nModel: {model_name}\nTask: {task}\nScore: {score:.4f}"
        pdf_data = make_pdf(summary_text, abstract)
        st.download_button("📄 Download PDF Report", data=pdf_data, file_name="report.pdf")

    st.subheader("📚 Zotero BibTeX Export")
    bib_refs = st.text_area("Paste reference titles/DOIs/URLs (one per line):")
    if st.button("📥 Generate BibTeX"):
        bibtex = "\n".join([f"@misc{{ref{i},\ntitle={{ {line.strip()} }},\n}}" for i, line in enumerate(bib_refs.strip().splitlines()) if line.strip()])
        st.download_button("📤 Download .bib", data=bibtex, file_name="references.bib")

st.markdown("---")
st.markdown("Developed by **Md. Mehedi Hasan @2025**")
