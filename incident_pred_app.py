# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:21:56 2022

@author: Nithish Kumar
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from pickle import load

import streamlit as st


# =============================================================================
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# =============================================================================

st.title('Incident Impact Predictor')
st.sidebar.header('Input parameters')

def user_input_features():
    urgency = st.sidebar.selectbox('urgency', ('1', '2', '3'))
    priority = st.sidebar.selectbox('priority', ('1', '2', '3', '4'))
    number = st.sidebar.number_input('number')
    opened_by = st.sidebar.number_input('opened by')
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

prediction = rf_model.predict(input_df)

st.subheader('Impact type')
st.write(prediction)
