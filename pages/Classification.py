import streamlit as st
import pycaret as py
import streamlit as st
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model,predict_model,plot_model
import pandas as pd
import pandas-profiling
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar: 

    st.title("Auto Mashan ML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Prediction","confusion_matrix","Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice=="Prediction":
    pred=predict_model(best_model)
    st.dataframe(pred)

if choice=="confusion_matrix":
    plot=plot_model(best_model, plot = 'confusion_matrix')
    st.image(plot)

if choice == "Download": 
    model_file_path = 'best_model.pkl'  # Check if this file path is correct
    with open(model_file_path, 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
