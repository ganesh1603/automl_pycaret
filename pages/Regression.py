import streamlit as st
import pycaret as py
import pandas as pd
import streamlit as st
import pandas-profiling
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model,predict_model
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


#SIDEBAR
with st.sidebar: 

    st.title("Auto Mashan ML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Prediction","Line_plot","Download"])
    st.info("This project application helps you build and explore your data.")

#upload csv file
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset",type=["csv"])
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

#EDA by pandas profiling
if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

#Modelling the data
if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    st.write("wait till to model run to predict the best model")
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

        
#Prediction
if choice=="Prediction":
    pred=predict_model(best_model)
    st.dataframe(pred)

#Line plot
if choice=="Line_plot":
    plot=plot_model(best_model, plot = 'line')
    st.image(plot)

#Downloading model
if choice == "Download": 
    model_file_path = 'best_model.pkl'  # Check if this file path is correct
    with open(model_file_path, 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
