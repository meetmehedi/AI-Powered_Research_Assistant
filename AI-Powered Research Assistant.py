# AI-Powered Research Assistant for Scientists

import streamlit as st
# Configure Streamlit page
st.set_page_config(
    page_title="AI-Powered Research Assistant",
    layout="wide",
    page_icon="🧠"
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    st.warning("XGBoost not installed - some features may be limited")

try:
    import autosklearn.classification
    import autosklearn.regression
except ImportError:
    autosklearn = None
    st.warning("Auto-sklearn not installed - some features may be limited")

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


st.title("SciGenie 🧠")
st.caption("AI-Powered Research Assistant for Scientists")

# --- Helper Functions ---
def auto_clean(df):
    """Automatically clean the dataframe"""
    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Forward fill missing values
    df.ffill(inplace=True)
    
    # Convert object columns with few unique values to categorical
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() < 10:
            df[col] = df[col].astype('category')
    
    return df

def train_models(X, y, task):
    """Train and evaluate multiple models for the given task"""
    results = {}
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == 'classification':
        # Logistic Regression baseline
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        results['LogisticRegression'] = (lr, accuracy_score(y_val, lr.predict(X_val)))

        # Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        results['RandomForest'] = (rf, accuracy_score(y_val, rf.predict(X_val)))

        if xgb:
            xgb_model = xgb.XGBClassifier()
            xgb_model.fit(X_train, y_train)
            results['XGBoost'] = (xgb_model, accuracy_score(y_val, xgb_model.predict(X_val)))

        if autosklearn is not None:
            ask = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)
            ask.fit(X_train, y_train)
            results['AutoSklearn'] = (ask, accuracy_score(y_val, ask.predict(X_val)))

    else:  # Regression
        # Linear Regression baseline
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results['LinearRegression'] = (lr, r2_score(y_val, lr.predict(X_val)))

        # Random Forest
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        results['RandomForest'] = (rf, r2_score(y_val, rf.predict(X_val)))

        if xgb:
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X_train, y_train)
            results['XGBoost'] = (xgb_model, r2_score(y_val, xgb_model.predict(X_val)))

        if autosklearn is not None:
            ask = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120)
            ask.fit(X_train, y_train)
            results['AutoSklearn'] = (ask, r2_score(y_val, ask.predict(X_val)))

    # Get best model based on validation score
    best_model_name = max(results, key=lambda k: results[k][1])
    return best_model_name, results[best_model_name][0], results[best_model_name][1]

def generate_abstract_with_deepseek(df, task, model_name, score):
    """Generate an abstract using DeepSeek's API"""
    if 'DEEPSEEK_API_KEY' not in st.secrets:
        return "DeepSeek abstract not generated. Please set DEEPSEEK_API_KEY in .streamlit/secrets.toml."
    
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Write a concise academic abstract (150 words max) for a research project with:
- Dataset columns: {list(df.columns)}
- Task: {task}
- Best model: {model_name} (validation score: {score:.3f})
- Key insights from the analysis"""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error from DeepSeek API: {response.text}"
            
    except Exception as e:
        return f"DeepSeek API unreachable. Please check your internet or DNS settings.\n{str(e)}"

def make_pdf(summary, abstract):
    """Create a PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Research Report", ln=True, align="C")
    pdf.ln(10)
    
    # Add summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.ln()
    
    # Add abstract
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Abstract", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, abstract)

    return pdf.output(dest="S").encode("latin-1")

# --- Main App ---
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])
if uploaded_file:
    try:
        file_ext = uploaded_file.name.split(".")[-1]
        df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
        
        with st.spinner("Cleaning data..."):
            df = auto_clean(df)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Data Summary", "EDA", "Modeling", "Export"])

        with tab1:
            st.subheader("🧹 Data Cleaning Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing Values", df.isnull().sum().sum())
                
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes.value_counts().to_frame("Count"))
                
            st.write("Missing Values per Column:")
            st.dataframe(df.isnull().sum().to_frame("Missing Values"))

        with tab2:
            st.subheader("📈 Exploratory Data Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_col = st.selectbox("Select numeric column for histogram:", 
                                     df.select_dtypes(include='number').columns)
                fig = plt.figure()
                sns.histplot(df[num_col], kde=True)
                st.pyplot(fig)
                
            with col2:
                if len(df.select_dtypes(include='number').columns) > 1:
                    x_axis = st.selectbox("X-axis", df.select_dtypes(include='number').columns)
                    y_axis = st.selectbox("Y-axis", df.select_dtypes(include='number').columns)
                    fig = plt.figure()
                    sns.scatterplot(data=df, x=x_axis, y=y_axis)
                    st.pyplot(fig)
            
            st.subheader("📌 Correlation Heatmap")
            if len(df.select_dtypes(include='number').columns) > 1:
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(df.select_dtypes(include='number').corr(), 
                           annot=True, 
                           cmap='coolwarm',
                           center=0)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for correlation")

        with tab3:
            st.subheader("🧠 Model Training")
            target = st.selectbox("Select Target Column:", df.columns)
            
            # Determine task type
            if df[target].dtype == "object" or df[target].nunique() < 10:
                task = 'classification'
                st.success(f"✅ Classification task detected ({df[target].nunique()} classes)")
            else:
                task = 'regression'
                st.info("📈 Regression task detected")
                
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models..."):
                    try:
                        X = df.drop(columns=[target])
                        X = pd.get_dummies(X)
                        y = df[target]
                        
                        model_name, model, score = train_models(X, y, task)
                        
                        st.success(f"✅ Best model: {model_name} | Validation score: {score:.4f}")
                        
                        # Show model performance
                        st.subheader("Model Performance")
                        if task == 'classification':
                            y_pred = model.predict(X)
                            st.write("Confusion Matrix:")
                            fig = plt.figure()
                            sns.heatmap(confusion_matrix(y, y_pred), 
                                        annot=True, 
                                        fmt='d',
                                        cmap='Blues')
                            st.pyplot(fig)
                            st.write("Classification Report:")
                            st.code(classification_report(y, y_pred))
                        else:
                            y_pred = model.predict(X)
                            fig = plt.figure()
                            plt.scatter(y, y_pred)
                            plt.xlabel("Actual")
                            plt.ylabel("Predicted")
                            st.pyplot(fig)
                            st.write(f"R² Score: {r2_score(y, y_pred):.4f}")
                            
                        # Save model
                        joblib.dump(model, "trained_model.pkl")
                        with open("trained_model.pkl", "rb") as f:
                            st.download_button(
                                "📥 Download Model", 
                                data=f.read(), 
                                file_name="model.pkl",
                                help="Download the trained model file"
                            )
                            
                        # Generate abstract
                        with st.spinner("Generating abstract..."):
                            abstract = generate_abstract_with_deepseek(df, task, model_name, score)
                            st.subheader("📝 Generated Abstract")
                            st.write(abstract)
                            
                        # Create PDF report
                        summary_text = f"""Research Report
                        
Target Variable: {target}
Task Type: {task}
Best Model: {model_name}
Validation Score: {score:.4f}

Features Used:
{', '.join(X.columns)}
"""
                        pdf_data = make_pdf(summary_text, abstract)
                        st.download_button(
                            "📄 Download Full Report", 
                            data=pdf_data, 
                            file_name="research_report.pdf",
                            help="Download a PDF summary of the analysis"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")

        with tab4:
            st.subheader("📚 Reference Management")
            
            st.write("Generate BibTeX entries for your references:")
            bib_refs = st.text_area("Enter references (one per line):", 
                                  placeholder="Paste DOI, title, or URL of each reference")
            
            if st.button("📥 Generate BibTeX"):
                if bib_refs.strip():
                    entries = []
                    for i, line in enumerate(bib_refs.strip().splitlines()):
                        if line.strip():
                            entries.append(
                                f"@misc{{ref{i+1},\n"
                                f"  title = {{ {line.strip()} }},\n"
                                f"  note = {{Added by SciGenie}}\n"
                                f"}}\n"
                            )
                    bibtex = "\n".join(entries)
                    st.download_button(
                        "📤 Download .bib", 
                        data=bibtex, 
                        file_name="references.bib",
                        help="Download BibTeX references file"
                    )
                    st.code(bibtex, language="latex")
                else:
                    st.warning("Please enter at least one reference")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Developed by **Md. Mehedi Hasan @2025**")
