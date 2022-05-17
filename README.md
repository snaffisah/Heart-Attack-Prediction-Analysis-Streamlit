# Heart Attack Prediction Analysis

## Description
Creating machine learning model analysis using logistic regression and run the streamlit apps to predict the probability of having heart attack in future.
 
* Model training - Machine learning
* Method: Logistic Regression
* Deployment apps: Streamlit

In this analysis, dataset used from https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset.

### About The Dataset:
age: Age of the patient

sex: Sex of the patient

cp: Chest pain type, 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic

trtbps: Resting blood pressure (in mm Hg)

chol: Cholestoral in mg/dl fetched via BMI sensor

fbs: (fasting blood sugar > 120 mg/dl), 1 = True, 0 = False

restecg: Resting electrocardiographic results, 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy

thalachh: Maximum heart rate achieved

oldpeak: Previous peak

slp: Slope

caa: Number of major vessels

thall: Thalium Stress Test result ~ (0,3)

exng: Exercise induced angina ~ 1 = Yes, 0 = No

output: Target variable 0:"LOW", 1:"HIGH"

### Correlation of each dataset:
<p align="center">
  <img width="460" src="https://github.com/snaffisah/Heart-Attack-Prediction-Analysis/blob/main/Image/Heatmap%20heart%20attack.png">
</p>

### Prediction accuracy:
<p align="center">
  <img width="360" src="https://github.com/snaffisah/Heart-Attack-Prediction-Analysis/blob/main/Image/Resulttraining.JPG">
</p>
By using logistic regression method, we get the percentace of accuracy 87%

## How to run the Streamlit apps
You may clone the repository and open the Streamlit apps to test the prediction.

Steps to run the Streamlit apps:
1. Open command prompt
2. Activate environment
   Example: conda activate tf_env
3. Change directory to your folder path
   Example: cd C:\Users\snaff\OneDrive\Desktop\project 1\HeartAttack_Analysis 
4. Run streamlit of your folder
   Example: streamlit run HeartAttack_Prediction_Deployment.py
5. Streamlit apps will appear automatically on your browser

<p align="center">
  <img width="460" src="https://github.com/snaffisah/Heart-Attack-Prediction-Analysis/blob/main/Image/Commandprompt.JPG">
</p>

<p align="center">
  <img width="460" src="https://github.com/snaffisah/Heart-Attack-Prediction-Analysis/blob/main/Image/Streamlit_form.JPG">
</p>

You may insert the patient information details to check the prediction and click "submit" button for the result.

<p align="center">
  <img width="460" src="https://github.com/snaffisah/Heart-Attack-Prediction-Analysis/blob/main/Image/Streamlit_success.JPG">
</p>

Enjoy!

