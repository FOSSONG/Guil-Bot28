# Import necessary modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, roc_auc_score, roc_curve
from scipy import stats
import numpy as np

# Set up the Streamlit framework
st.title('Guil-Bot28')  # Set the title of the Streamlit app

# Navigation options
page = st.sidebar.selectbox("Select a Page", ["Home", "Basic EDA", "Visualizations", "Advanced EDA", "Statistical Tests", "Machine Learning"])

# Define a prompt template for the cognitive computing model
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a cognitive computing model with advanced capabilities including data mining, pattern recognition, and natural language processing. 
        You can analyze complex datasets, identify patterns, extract meaningful insights, and understand and generate human-like text. 
        Please use these capabilities to assist users effectively and provide comprehensive responses.
        """),
        ("user", "Question: {question}")
    ]
)

# Initialize the Ollama model
llm = Ollama(model="llama2")

# Home Page
if page == "Home":
    input_text = st.text_input("Ask your question!")
    if input_text:
        with st.spinner('Generating response...'):
            response = prompt | llm
            st.write(response.invoke({"question": input_text}))

# Function to clean the data
def clean_data(df):
    # Convert non-numeric columns to numeric where possible, set errors='coerce' to convert non-convertible values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    return df

# File Uploader
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
if uploaded_file:
    # Read the uploaded data file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_table(uploaded_file)

    # Clean the data
    data = clean_data(data)

    st.sidebar.write("Data Loaded and Cleaned Successfully!")

    # Define numeric data globally after data is loaded
    numeric_data = data.select_dtypes(include=[float, int])

# Basic EDA Page
if page == "Basic EDA" and uploaded_file:
    st.header("Basic Exploratory Data Analysis")
    st.write("Summary Statistics")
    st.write(data.describe())

    st.write("Correlation Matrix")
    st.write(numeric_data.corr())

    st.write("Missing Values")
    st.write(data.isnull().sum())

# Visualizations Page
if page == "Visualizations" and uploaded_file:
    st.header("Visualizations")
    
    st.write("Pairplot")
    fig = sns.pairplot(data)
    st.pyplot(fig)

    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("Distribution Plots")
    for col in numeric_data.columns:
        fig, ax = plt.subplots()
        sns.histplot(data[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

# Advanced EDA Page
if page == "Advanced EDA" and uploaded_file:
    st.header("Advanced Exploratory Data Analysis")

    st.subheader("Outlier Detection")
    for col in numeric_data.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col].dropna(), ax=ax)
        st.pyplot(fig)

# Statistical Tests Page
if page == "Statistical Tests" and uploaded_file:
    st.header("Statistical Tests")
    if len(numeric_data.columns) >= 2:
        st.write("Correlation and P-Values")
        correlation_matrix = numeric_data.corr()
        p_values = pd.DataFrame(np.zeros_like(correlation_matrix), columns=numeric_data.columns, index=numeric_data.columns)
        for row in numeric_data.columns:
            for col in numeric_data.columns:
                _, p = stats.pearsonr(numeric_data[row].dropna(), numeric_data[col].dropna())
                p_values.loc[row, col] = p
        st.write("Correlation Matrix")
        st.write(correlation_matrix)
        st.write("P-Values Matrix")
        st.write(p_values)

# Machine Learning Page
if page == "Machine Learning" and uploaded_file:
    st.header("Machine Learning Predictions")
    st.write("Specify the input features and target variable")

    # User input for features and target
    features = st.multiselect("Select the input features", options=data.columns)
    target = st.selectbox("Select the target variable", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)

        if st.button("Split Data"):
            # Split the data into training, validation, and testing sets
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            st.write("Data Split Successfully!")

        if 'X_train' in locals():
            if st.button("Train Model"):
                # Determine the problem type (regression or classification)
                if y.dtype == 'float' or y.dtype == 'int':
                    model = RandomForestRegressor()
                    problem_type = "regression"
                else:
                    model = RandomForestClassifier()
                    problem_type = "classification"

                # Train the model
                model.fit(X_train, y_train)
                st.write("Model Trained Successfully!")

            if 'model' in locals():
                if st.button("Check Model Performance"):
                    y_val_pred = model.predict(X_val)
                    if problem_type == "regression":
                        rmse = mean_squared_error(y_val, y_val_pred, squared=False)
                        mae = mean_absolute_error(y_val, y_val_pred)
                        st.write(f"Validation RMSE: {rmse}")
                        st.write(f"Validation MAE: {mae}")
                    else:
                        accuracy = accuracy_score(y_val, y_val_pred)
                        f1 = f1_score(y_val, y_val_pred, average='weighted')
                        st.write(f"Validation Accuracy: {accuracy}")
                        st.write(f"Validation F1 Score: {f1}")
                        st.write("Classification Report:")
                        st.write(classification_report(y_val, y_val_pred))
                        st.write("Confusion Matrix:")
                        cm = confusion_matrix(y_val, y_val_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                        if problem_type == "classification":
                            roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                            st.write(f"Validation ROC AUC: {roc_auc}")
                            fpr, tpr, _ = roc_curve(y_val, model.predict_proba(X_val)[:, 1])
                            fig, ax = plt.subplots()
                            ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc})")
                            ax.plot([0, 1], [0, 1], linestyle='--')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend(loc="best")
                            st.pyplot(fig)

                if st.button("Make Predictions"):
                    y_test_pred = model.predict(X_test)
                    st.write("Predictions made successfully!")

                    if problem_type == "regression":
                        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
                        mae = mean_absolute_error(y_test, y_test_pred)
                        st.write(f"Test RMSE: {rmse}")
                        st.write(f"Test MAE: {mae}")
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_test_pred)
                        ax.set_xlabel('Actual Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title('Actual vs Predicted Values')
                        st.pyplot(fig)
                    else:
                        accuracy = accuracy_score(y_test, y_test_pred)
                        f1 = f1_score(y_test, y_test_pred, average='weighted')
                        st.write(f"Test Accuracy: {accuracy}")
                        st.write(f"Test F1 Score: {f1}")
                        st.write("Classification Report:")
                        st.write(classification_report(y_test, y_test_pred))
                        st.write("Confusion Matrix:")
                        cm = confusion_matrix(y_test, y_test_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                        if problem_type == "classification":
                            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                            st.write(f"Test ROC AUC: {roc_auc}")
                            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                            fig, ax = plt.subplots()
                            ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc})")
                            ax.plot([0, 1], [0, 1], linestyle='--')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend(loc="best")
                            st.pyplot(fig)

st.set_option('deprecation.showPyplotGlobalUse', False)
