
import streamlit as st
import pandas as pd

import os

# adding profiling functionality
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

'''ML models'''

# Regression
from pycaret.regression import setup as r_setup
from pycaret.regression import compare_models as r_compare_models
from pycaret.regression import pull as r_pull
from pycaret.regression import save_model as r_save

# Classification
from pycaret.classification import setup as c_setup
from pycaret.classification import compare_models as c_compare_models
from pycaret.classification import pull as c_pull
from pycaret.classification import save_model as c_save

# Chatbot
import openai
from streamlit_chat import message

# base64 library to be able to use urls
import base64


def set_bg(main_bg):
    main_bg_ext = "jpg"

    st.markdown(
        f"""
        <style>
        .stApp {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )


set_bg('pxfuel.jpg')


def generate_model_review(model_data):
    completions = openai.Completion.create(
        engine = 'text-davinci-003',
        prompt = f"Analyze the following ML model's performance: \n {model_data}",
        max_tokens = 1024,
        n = 1,
        stope = None,
        temparature = .5
    )


with st.sidebar:
    st.image("Dalle-image.png")
    st.title("automated ML pipeline")
    choice = st.radio("Menu", ["Upload Dataset", "Exploratory Data Analysis", "Modelling", "Download Model"])
    st.info("A customizable, automated ML pipeline.")


if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv", index_col = None)

if choice == "Upload Dataset":
    st.title("Data to be analyzed")
    file = st.file_uploader("Upload .csv file")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("data.csv", index = None)
        st.dataframe(df)


if choice == "Exploratory Data Analysis":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Modelling":
    st.title("Training and Test ML models")
    target = st.selectbox("Select your label", df.columns)
    model_type = st.radio("Select Model Type", ["Regression", "Classification"])
    if model_type == "Regression":
        r_setup(df, target = target)
        setup_df = r_pull()
        st.info("Metadata for the experiment, and settings")
        st.dataframe(setup_df)
        best_model = r_compare_models()
        compare_df = r_pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        r_save(best_model, "best_model")

    if model_type == "Classification":
        c_setup(df, target = target)
        setup_df = c_pull()
        st.info("Metadata for the experiment, and settings")
        st.dataframe(setup_df)
        best_model = c_compare_models()
        compare_df = c_pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        c_save(best_model, "best_model")

# Get Chat GPT to make commentary about model outcomes
if choice == "Download Model":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the trained pycaret Model", f, "trained_model.pkl")
