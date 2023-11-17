import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, save_model, load_model, plot_model
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import os

# Check if the dataset exists and load it
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# SIDEBAR
with st.sidebar:
    st.title("Auto Mashan ML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Custom Plot", "Download"])
    st.info("This project application helps you build and explore your data.")

# UPLOAD CSV OR EXCEL FILE
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv", "data", "xls", "xlsx"])
    if file:
        # Check the file extension and load the dataset accordingly
        file_extension = file.name.split('.')[-1].lower()
        if file_extension in ['csv', 'data']:
            df = pd.read_csv(file, index_col=None)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file, index_col=None)
        # Save the dataset to a CSV file for later use
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

# EDA BY PANDAS PROFILING
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if 'df' in locals():
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
    else:
        st.warning("Please upload data by selecting 'Upload'.")

# MODELLING THE DATA
if choice == "Modelling":
    if 'df' in locals():
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            s = setup(df, target=chosen_target, silent=True, session_id=123)
            best_model = compare_models()
            save_model(best_model, 'best_model')
            st.success("Best model saved successfully!")
    else:
        st.warning("Please upload data by selecting 'Upload'.")

# CUSTOM PLOT
if choice == "Custom Plot":
    if os.path.exists('best_model.pkl'):
        best_model = load_model('best_model')
        st.title("Model Evaluation Plots")
        available_plots = ['auc', 'confusion_matrix', 'threshold', 'pr', 'error', 'class_report', 'boundary', 'learning', 'manifold', 'calibration', 'vc', 'dimension', 'feature', 'feature_all', 'parameter']
        selected_plot = st.selectbox('Select Plot Type', available_plots)
        if st.button('Generate Plot'):
            plot = plot_model(best_model, plot=selected_plot, save=True)
            st.pyplot(plot)
    else:
        st.warning("Please run modelling first by selecting 'Modelling'.")

# DOWNLOADING MODEL
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("Please run modelling first by selecting 'Modelling'.")

# Run the Streamlit application
if __name__ == '__main__':
    st.title("AutoML with Streamlit")
    st.write("Navigate through the sidebar to upload data, perform EDA, model, plot, and download the model.")
