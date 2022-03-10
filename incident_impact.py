# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:46:41 2022

@author: Nithish Kumar
"""
# import necessary libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from pickle import dump

# load data
incident_df = pd.read_csv('incident_event_log.csv')
incident_df.head()

# separate features and target (choose only reqd cols after analysis)
target = 'impact' # target column
sel_features = ['urgency', 'priority', 'number', 'opened_by']
# Note: Chosen columns/features have a direct relationship with target 

X = incident_df.loc[:, sel_features]
y = incident_df.loc[:, target]

# Data preprocessing fuctions.

# 1. Function to remove all null columns.
def null_value_handler(Xdf):
    # Replace all '?' with np.nan
    Xdf['opened_by'] = Xdf['opened_by'].replace(to_replace='?', value=np.nan)
    # impute null values with mode
    Xdf['opened_by'] = Xdf['opened_by'].fillna(value=Xdf['opened_by'].mode()[0])
    
    return Xdf  

# 2. Function to encode the urgency and priority columns
def categorical_encoder1(Xdf):
    # scale:
    urgency_scale = {'1 - High':1, '2 - Medium':2, '3 - Low':3}
    priority_scale = {'1 - Critical':1, '2 - High':2, '3 - Moderate':3, '4 - Low':4}
    
    Xdf['urgency'] = Xdf['urgency'].map(urgency_scale)
    Xdf['priority'] = Xdf['priority'].map(priority_scale)
    
    return Xdf

# 3. Function to ordinal encode high cardinality columns
def categorical_encoder2(Xdf):
       
    Xdf['number'] = Xdf['number'].apply(lambda x: x.strip('INC')).astype('int')
    Xdf['opened_by'] = Xdf['opened_by'].apply(lambda x: x.strip('Opened by  ')).astype('int')

    return Xdf

# 4.Final  data preprocessing function that includes the above preprocessing steps.
def data_preprocessor(Xdf):
    Xdf = null_value_handler(Xdf)
    Xdf = categorical_encoder1(Xdf)
    Xdf = categorical_encoder2(Xdf) 
    
    return Xdf

# 5. Function to encode the labels of the target column.
def out_label_encoder(ydf):
    impact_scale = {'1 - High':1, '2 - Medium':2, '3 - Low':3}
    ydf = ydf.map(impact_scale)
    return ydf 
  
# Preprocessing input data(initial).
# There is no information leakage in the above preprocessing steps.
# Mode, when taken either separately for train and test or for full dataset
# is the same if the test size is 20% or 0.2 (from observations).

X_prep = data_preprocessor(X.copy())
y_prep = out_label_encoder(y.copy())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, stratify=y, random_state=42)

# Constructing the model
transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(transformers=[
    ('cat_trf', transformer, sel_features)])

rf_clf = RandomForestClassifier(random_state=42)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_clf)
    ])

# Function to initialize, train and give predictions 
def clf_model(Xtrain, Xtest,  ytrain, ytest, classifier):
    """ Fits the model and generates predictions, returns
    the predictions for train and test set 
    along with fitted model. 
    input
    -----
    Xtrain, Xtest,  ytrain, ytest, classifier
    
    output
    ------
    ypred_train, ypred_test, clf
    """
    clf = classifier
    clf.fit(Xtrain, ytrain)
    
    ypred_train = clf.predict(Xtrain)
    ypred_test = clf.predict(Xtest)
    
    # Train data performance
    #display_results(ytrain, ypred_train, clf)
    #display_results(ytest, ypred_test, clf)  
    return ypred_train, ypred_test, clf

# Function to evaluate model
def display_results(y_test, y_pred, clf):
    """Displays model evaluation/performance report that includes
    accuracy_score, confusion_matrix, precision_score, and 
    recall_score.
    input
    -----
    y_test, y_pred
    
    output
    ------
    Model evaluation/performance report"""
    print(classification_report(y_test, y_pred))
    #print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=clf.classes_, ax=ax)
    
# Model training
y_pred_train, y_pred_test, clf =clf_model(Xtrain=X_train,
          Xtest=X_test,
          ytrain=y_train,
          ytest=y_test,
          classifier=clf)

# Model evaluation: Train data
display_results(y_train, y_pred_train, clf)
plt.title('model performance: Train data')
plt.show()

# Model evaluation: Test data
display_results(y_test, y_pred_test, clf)
plt.title('model performance: Test data')
plt.show()

# generate pickle file of the model
dump(clf, open('incident_impact_rf.pkl', 'wb'))