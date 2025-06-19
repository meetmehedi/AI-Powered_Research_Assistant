# AI-Powered Research Assistant for Scientists
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, r2_score, 
                           confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error)
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page (must be the first Streamlit command)
st.set_page_config(
    page_title="SciGenie - AI Research Assistant",
    layout="wide",
    page_icon="🧠"
)

# Optional imports with try-except blocks
optional_imports = {
    'xgboost': None,
    'autosklearn': None,
    'pycaret': None
}

try:
    import xgboost as xgb
    optional_imports['xgboost'] = xgb
except ImportError:
    st.warning("XGBoost not installed - some features may be limited")

try:
    import autosklearn.classification
    import autosklearn.regression
    optional_imports['autosklearn'] = autosklearn
except ImportError:
    st.warning("Auto-sklearn not installed - some features may be limited")

try:
    from pycaret.classification import setup as classification_setup
    from pycaret.regression import setup as regression_setup
    optional_imports['pycaret'] = True
except ImportError:
    st.warning("PyCaret not installed - AutoML features limited")


# --- Helper Functions ---
def safe_convert_to_numeric(series):
    """Safely convert a series to numeric, handling common issues"""
    # First try direct conversion
    try:
        return pd.to_numeric(series, errors='raise')
    except:
        # Handle common string representations
        replacements = {
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0,
            'y': 1, 'n': 0,
            't': 1, 'f': 0
        }
        cleaned = series.str.lower().replace(replacements)
        try:
            return pd.to_numeric(cleaned, errors='coerce')
        except:
            return series  # Return original if conversion fails

def auto_clean(df):
    """Automatically clean the dataframe with robust type conversion"""
    # Remove completely empty columns and rows
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Fill missing values with appropriate methods
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # Drop columns with too many remaining missing values (e.g., >50%)
    threshold = 0.5
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index
    df = df.drop(columns=cols_to_drop)

    # Drop rows with any remaining NaNs
    df = df.dropna()

    # Convert columns to appropriate types
    for col in df.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Try converting to numeric first
        numeric_version = safe_convert_to_numeric(df[col].astype(str))
        if numeric_version.isna().mean() < 0.3:  # If <30% conversion failed
            df[col] = numeric_version
        elif df[col].nunique() < min(20, len(df)//10):  # Good candidate for category
            df[col] = df[col].astype('category')
            
    return df

def determine_task_type(target_series):
    """Determine whether a task is classification or regression"""
    if pd.api.types.is_categorical_dtype(target_series):
        return 'classification'
    if target_series.nunique() < 10:
        return 'classification'
    if pd.api.types.is_numeric_dtype(target_series):
        return 'regression'
    return 'classification'  # Default to classification

def train_models(X, y, task):
    """Train and evaluate multiple models for the given task"""
    results = {}
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task == 'classification' else None
    )

    if task == 'classification':
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42)
        }
        
        if optional_imports['xgboost']:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42)
            
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_val, y_pred),
                'confusion_matrix': confusion_matrix(y_val, y_pred),
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }

    else:  # Regression
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(random_state=42)
        }
        
        if optional_imports['xgboost']:
            models['XGBoost'] = xgb.XGBRegressor(random_state=42)
            
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            results[name] = {
                'model': model,
                'r2': r2_score(y_val, y_pred),
                'mse': mean_squared_error(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred)
            }

    # Get best model based on primary metric
    if task == 'classification':
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    else:
        best_model_name = max(results, key=lambda k: results[k]['r2'])
        
    return best_model_name, results[best_model_name], results

def generate_analysis_report(df, target, task, best_model_info, all_models):
    """Generate a comprehensive analysis report"""
    report = f"# SciGenie Analysis Report\n\n"
    report += f"## Dataset Overview\n"
    report += f"- Rows: {len(df):,}\n"
    report += f"- Columns: {len(df.columns):,}\n"
    report += f"- Target variable: {target}\n"
    report += f"- Task type: {task}\n\n"
    
    report += "## Best Model Performance\n"
    if task == 'classification':
        report += f"- Model: {best_model_info['model'].__class__.__name__}\n"
        report += f"- Accuracy: {best_model_info['accuracy']:.4f}\n"
        report += "\n### Classification Report\n"
        for k, v in best_model_info['classification_report'].items():
            if k not in ['accuracy', 'macro avg', 'weighted avg']:
                report += f"- {k}: precision={v['precision']:.2f}, recall={v['recall']:.2f}, f1={v['f1-score']:.2f}\n"
    else:
        report += f"- Model: {best_model_info['model'].__class__.__name__}\n"
        report += f"- R² Score: {best_model_info['r2']:.4f}\n"
        report += f"- Mean Squared Error: {best_model_info['mse']:.4f}\n"
        report += f"- Mean Absolute Error: {best_model_info['mae']:.4f}\n"
    
    report += "\n## All Models Comparison\n"
    if task == 'classification':
        report += "| Model | Accuracy |\n|-------|----------|\n"
        for name, info in all_models.items():
            report += f"| {name} | {info['accuracy']:.4f} |\n"
    else:
        report += "| Model | R² Score | MSE | MAE |\n|-------|----------|-----|-----|\n"
        for name, info in all_models.items():
            report += f"| {name} | {info['r2']:.4f} | {info['mse']:.4f} | {info['mae']:.4f} |\n"
    
    return report

# --- Main App ---
st.title("🧠 SciGenie - AI Research Assistant")
st.caption("Your intelligent companion for scientific data analysis")

uploaded_file = st.file_uploader(
    "📂 Upload your dataset (CSV/Excel)", 
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel (XLSX, XLS)"
)

if uploaded_file:
    try:
        # Read file based on extension
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Ensure all column names are valid strings
        df.columns = [str(col) if col is not None else "Unnamed_Column" for col in df.columns]
        
        with st.spinner("🧹 Cleaning and preprocessing data..."):
            df = auto_clean(df)
            st.session_state['cleaned_df'] = df

        st.success("✅ Data loaded and preprocessed successfully!")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Data Summary", "📊 EDA", "🤖 Modeling", "📤 Export"]
        )

        with tab1:
            st.subheader("Data Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Rows", df.shape[0])
                st.metric("Total Columns", df.shape[1])
                st.metric("Missing Values", df.isnull().sum().sum())
                
            with col2:
                st.write("### Data Types")
                dtype_counts = df.dtypes.value_counts().reset_index()
                dtype_counts.columns = ['Data Type', 'Count']
                st.dataframe(dtype_counts, hide_index=True)
                
            st.write("### Missing Values Analysis")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Values']
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, hide_index=True)
            else:
                st.success("No missing values found in the dataset!")
            
            st.write("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)

        with tab2:
            st.subheader("Exploratory Data Analysis")
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.write("#### Numeric Features Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_num_col = st.selectbox(
                        "Select numeric column for distribution:",
                        numeric_cols
                    )
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_num_col], kde=True, ax=ax)
                    st.pyplot(fig)
                
                with col2:
                    st.write("Descriptive Statistics")
                    st.dataframe(df[numeric_cols].describe().T)
                
                # Correlation heatmap if multiple numeric columns
                if len(numeric_cols) > 1:
                    st.write("#### Correlation Analysis")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        df[numeric_cols].corr(), 
                        annot=True, 
                        cmap='coolwarm',
                        center=0,
                        ax=ax
                    )
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found for analysis")
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['category', 'object']).columns
            if len(categorical_cols) > 0:
                st.write("#### Categorical Features Analysis")
                
                selected_cat_col = st.selectbox(
                    "Select categorical column:",
                    categorical_cols
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    df[selected_cat_col].value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                
                with col2:
                    st.write("Value Counts")
                    st.dataframe(
                        df[selected_cat_col].value_counts().reset_index(),
                        hide_index=True
                    )

        with tab3:
            st.subheader("Machine Learning Modeling")
            
            target_col = st.selectbox(
                "Select target variable:",
                df.columns,
                index=len(df.columns)-1 if len(df.columns) > 0 else 0
            )
            
            if target_col:
                task_type = determine_task_type(df[target_col])
                
                if task_type == 'classification':
                    st.success(f"🔮 Task type: Classification ({(df[target_col].nunique())} classes)")
                    st.write("Class distribution:")
                    st.dataframe(df[target_col].value_counts().reset_index(), hide_index=True)
                else:
                    st.info("📈 Task type: Regression")
                    fig, ax = plt.subplots()
                    sns.histplot(df[target_col], kde=True, ax=ax)
                    st.pyplot(fig)
                
                if st.button("🚀 Train Models", type="primary"):
                    with st.spinner("Training models..."):
                        try:
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            # Impute missing values in y
                            y = y.fillna(y.median() if pd.api.types.is_numeric_dtype(y) else y.mode()[0])
                            
                            # Handle categorical features
                            X = pd.get_dummies(X, drop_first=True)
                            # Impute missing values in X
                            from sklearn.impute import SimpleImputer
                            imputer = SimpleImputer(strategy='median')
                            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                            
                            best_name, best_model, all_models = train_models(X, y, task_type)
                            
                            st.session_state['best_model'] = best_model
                            st.session_state['all_models'] = all_models
                            st.session_state['task_type'] = task_type
                            st.session_state['target'] = target_col
                            
                            st.success(f"🎉 Best model: {best_name}")
                            
                            # Show detailed results
                            st.subheader("Model Performance")
                            
                            if task_type == 'classification':
                                st.write(f"**Accuracy**: {all_models[best_name]['accuracy']:.4f}")
                                
                                st.write("**Confusion Matrix**")
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    all_models[best_name]['confusion_matrix'],
                                    annot=True,
                                    fmt='d',
                                    cmap='Blues',
                                    ax=ax
                                )
                                st.pyplot(fig)
                                
                                st.write("**Classification Report**")
                                st.dataframe(
                                    pd.DataFrame(all_models[best_name]['classification_report']).T
                                )
                            else:
                                col1, col2, col3 = st.columns(3)
                                col1.metric("R² Score", f"{all_models[best_name]['r2']:.4f}")
                                col2.metric("MSE", f"{all_models[best_name]['mse']:.4f}")
                                col3.metric("MAE", f"{all_models[best_name]['mae']:.4f}")
                                
                                st.write("**Actual vs Predicted**")
                                fig, ax = plt.subplots()
                                sns.scatterplot(
                                    x=y,
                                    y=best_model['model'].predict(X),
                                    ax=ax
                                )
                                ax.set_xlabel("Actual")
                                ax.set_ylabel("Predicted")
                                st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error during model training: {str(e)}")

        with tab4:
            st.subheader("Export Results")
            
            if 'best_model' in st.session_state:
                # Save model
                model_bytes = joblib.dumps(st.session_state['best_model']['model'])
                
                st.download_button(
                    label="💾 Download Model",
                    data=model_bytes,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )
                
                # Generate and download report
                report = generate_analysis_report(
                    st.session_state['cleaned_df'],
                    st.session_state['target'],
                    st.session_state['task_type'],
                    st.session_state['best_model'],
                    st.session_state['all_models']
                )
                
                st.download_button(
                    label="📄 Download Full Report (TXT)",
                    data=report,
                    file_name="analysis_report.txt",
                    mime="text/plain"
                )
                
                # PDF report
                if st.button("🖨️ Generate PDF Report"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, "SciGenie Analysis Report", ln=1, align='C')
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, "Dataset Overview", ln=1)
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, report.split("## Best Model Performance")[0])
                    
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, "Model Performance", ln=1)
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, "## Best Model Performance".join(report.split("## Best Model Performance")[1:]))
                    
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    
                    st.download_button(
                        label="📑 Download PDF Report",
                        data=pdf_bytes,
                        file_name="analysis_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("Please train models first to export results")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
# Footer
st.markdown("---")
st.markdown("Developed by **Md. Mehedi Hasan** | © 2025")
