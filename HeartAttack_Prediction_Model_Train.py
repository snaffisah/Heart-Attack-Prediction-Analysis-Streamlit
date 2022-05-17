# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:54:49 2022

Below are the prediction model training created to check the probability of getting
heart attack by analysing the patient’s age, gender, exercise induced angina, 
number of major vessels, chest pain indication, resting blood pressure, 
cholesterol level, fasting blood sugar, resting electrocardiographic results, 
and maximum heart rate achieved.

Model training - Machine learning
Method: Logistic Regression
Deployment apps: Streamlit

@author: snaff
"""

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# Constant
DATASET_HEART = os.path.join(os.getcwd(),'Data', 'heart.csv')
MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_path', 'mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), "saved_path", "model.pkl")

#%% EDA

# Step 1) Data loading
df = pd.read_csv(DATASET_HEART)

# Step 2) Data inspection
df.info() # from info, we saw that there is no missing data, type all in num
df.describe() # # mean and 50% value seems to be ok. No anomalies detected
df.head()

# Step 3) Data cleaning
df.isnull().sum() # to double confirm there is no missing data. Result = ok
df[df.duplicated()] # to check if there is any duplicate data
df.drop_duplicates(keep='first',inplace=True) # remove duplicate data
df.info() # double cek total of data

# Step 4) Feature selection
# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.Reds, linewidths=.5, ax=ax)
plt.show()
plt.figure()

# From EDA, only 1 duplicated data found during data cleaning and left only 
# 302 data for the analysis after drop it.
# From the correlation heatmap, cp, thalachh and slp has high correlation. 
# Exng and oldpeak has low correlation
# In this model, all features will be train to get the descrete prediction.
# Thus, logistic regression method will be use.

x = df.drop(labels=['output'], axis=1) # features 
y = df['output'] # target

# Step 5) Data Pre-processing

# Min Max Scaler
mms_scaler = MinMaxScaler()
x = mms_scaler.fit_transform(x)

# Saved the scaler used
pickle.dump(mms_scaler, open(MMS_SAVE_PATH, 'wb'))

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, 
                                                    random_state=0)

# Standard scaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Logistic regression method
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# prediction
y_predict = classifier.predict(x_test)

# Result of prediction model
cm = confusion_matrix(y_test, y_predict)
class_report = classification_report(y_test, y_predict)
score = accuracy_score(y_test, y_predict)

# Print result
print(cm)
print(class_report)
print("The model accuracy is:" ,score*100,)


#%% Model Saving
pickle.dump(classifier, open(MODEL_PATH, "wb"))
