import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import gradio as gr

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Data preprocessing
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=1)

# Train the SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Define a function to make predictions
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return 'You are Diabetic' if prediction[0] == 1 else 'Yay,you are Non-diabetic!'

# Create Gradio interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="DiabetesPedigreeFunction"),
        gr.Number(label="Age")
    ],
    outputs='text',
    title='Diabetes Prediction',
    description='Lets fill your details , to predict if you are diabetic or not.'
)

interface.share=True
# Launch the interface
interface.launch()
