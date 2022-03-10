# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:21:56 2022

@author: Project Group 4
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from pickle import load

import streamlit as st

html_temp = """
    <div style="background-color:Teal;padding:10px">
    <h2 style="color:white;text-align:center">Incident Impact Predictor </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.sidebar.header('Input parameters')

def user_input_features():

    urgency = st.sidebar.selectbox('urgency', ('', '1 - High','2 - Medium', '3 - Low'))
    priority = st.sidebar.selectbox('priority', ('', '1 - Critical', '2 - High', '3 - Moderate', '4 - Low'))
    number = st.sidebar.text_input('number')
    opened_by = st.sidebar.text_input('opened by')
    
    data = {'urgency':urgency,
            'priority':priority,
            'number':number,
            'opened_by':opened_by}
    
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

st.subheader('Input parameters')
st.write(input_df)

# load the model
rf_model = load(open('incident_impact_rf.pkl', 'rb'))

# Define the impact scale
impact_scale = {1:'High',
                2:'Medium',
                3:'Low'}

# Default values to fill if input box is empty
default_vals = {'urgency':'2 - Medium',
                'priority':'3 - Moderate',
                'number':'INC0019396',
                'opened_by':'Opened by  17'}

if st.button("Predict"):
    for col in input_df.columns:
        #input_df[col] = input_df[col].apply(lambda x: x.strip(' -HighMediumLowCriticalINCOpenedby')).astype('int')
        input_df[col] = input_df[col].apply(lambda x: x.strip(' -HighMediumLowCriticalINCOpenedby') if x!='' else default_vals[col].strip(' -HighMediumLowCriticalINCOpenedby')).astype('int')
    
    #st.write(input_df)
    result = rf_model.predict(input_df)
    st.subheader('Prediction')
    st.success('{} impact'.format(impact_scale[result[0]]))


st.subheader("Created by:-")
st.write("***Nithish***", ",", "***Sourajit***")
st.write("***Rahul***", ",", "***Dhanya***")
st.write("***Machindra***", ",", "***Vikas***")