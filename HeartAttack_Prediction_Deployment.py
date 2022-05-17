# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:11:17 2022

Below are the deployment coding for the probability of getting heart attack.

Model training - Machine learning
Method: Logistic Regression
Deployment apps: Streamlit

@author: snaff
"""

import pickle
import os
import numpy as np
import streamlit as st

# Constant
MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_path', 'mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), "saved_path", "model.pkl")

# Data Loading
with open(MMS_SAVE_PATH, 'rb') as f:
    mms_scaler = pickle.load(f)
    
# Model
with open(MODEL_PATH, 'rb') as g:
    classifier = pickle.load(g)
    
heartattack_chance = {0:"LOW", 1:"HIGH"}

# Test data 
#patient_info = np.array([55,1,0,132,353,0,1,132,1,1.2,1,1,3]) #low
#patient_info = np.array([47,1,2,130,253,0,1,179,0,0,2,0,2]) #high
#patient_info = mms_scaler.transform(np.expand_dims(patient_info, axis=0))

#predict using model
#new_pred = classifier.predict(patient_info)
#if np.argmax(new_pred) == 1:
#   new_pred = [0,1]
#   print(heartattack_chance[np.argmax(new_pred)])
#else:
#   new_pred = [1,0]
#   print(heartattack_chance[np.argmax(new_pred)])
    
#%% Build app using streamlit

with st.form('Heart Attack Prediction Form'):
    st.title("Heart Attack Prediction")
    st.header("Patient's Info")
    age = int(st.number_input("Age:")) # add int because not float
    sex = int(st.number_input("Sex") )
    cp = int(st.number_input("Chest Pain type:"))
    st.caption('''
        Value 1: typical angina \n
        Value 2: atypical angina \n
        Value 3: non-anginal pain \n
        Value 4: asymptomatic \n
             ''')
    trtbps = st.number_input("Resting blood pressure (in mm Hg):") # not need int because of float value
    chol = st.number_input("Cholestoral (in mg/dl):")
    fbs = int(st.number_input("Fasting blood sugar:"))
    st.caption('''
        Value 1 => 120 mg/dl \n
        Value 0 < 120 mg/dl \n
             ''')
    restecg = int(st.number_input("Resting electrocardiographic:"))
    st.caption('''
        Value 0: normal\n
        Value 1: having ST-T wave abnormality \n
        Value 2: showing probable or definite left ventricular hypertrophy \n
             ''')
    thalachh = st.number_input("Maximum heart rate:")
    exng = int(st.number_input("Exercise induced angina:"))
    st.caption('''
        Value 1 = Yes \n
        Value 0 = No \n
             ''')
    oldpeak = st.number_input("Previous Peak:")
    slp = int(st.number_input("Slope:"))
    caa = int(st.number_input("Number of major vessels (0-3):"))
    thall = int(st.number_input("Thal rate:"))
    
    submitted = st.form_submit_button('✅ Submit')
    
    if submitted == True:
        patient_info = np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall])
        patient_info = mms_scaler.transform(np.expand_dims(patient_info, axis=0))
        new_pred = classifier.predict(patient_info)
        if np.argmax(new_pred) == 1:
            st.warning
            (f'''Warning! This patient have {heartattack_chance[np.argmax(new_pred)]} 
             chances of heart attack ☹️ ''')
        else:
            st.snow()
            st.success
            (f'''This patient have {heartattack_chance[np.argmax(new_pred)]} 
             chances of heart attack 🙂''')
             
# To open streamlit apps
# 1. Open command prompt
# 2. Activate environment
#    example: conda activate tf_env
# 3. Change directory to your folder path
#    example: cd C:\Users\snaff\OneDrive\Desktop\project 1\HeartAttack_Analysis 
# 4. Run streamlit of your folder
#    example: streamlit run HeartAttack_Prediction_Deployment.py