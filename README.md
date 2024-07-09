# Diabetes Prediction Using Support Vector Machine and Gradio

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Explanation](#code-explanation)
6. [Gradio Interface](#gradio-interface)
7. [Conclusion](#conclusion)

## Project Overview
This project demonstrates how to predict diabetes using a Support Vector Machine (SVM) and Gradio for creating a user-friendly interface. The aim is to provide an accessible tool for predicting diabetes based on various health metrics.

## Dataset
We use the PIMA Indian Diabetes Dataset, which includes health metrics such as pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. The dataset contains 768 entries labeled as diabetic or non-diabetic.

## Installation
To run this project, you need to install the required dependencies. Create a `requirements.txt` file with the following content:

```plaintext
pandas
numpy
scikit-learn
gradio
```
You can install these dependencies using the following command:
```bash
pip install -r requirements.txt
```
## Usage
Load the Dataset:
1. Place the diabetes.csv file in the same directory as your script.
2. Run the Script:
Execute the app.py script to start the Gradio interface.
```bash
python app.py
```
## Code Explanation
The project code is divided into several parts:

### 1. Importing Libraries and Loading the Dataset:
```bash
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import gradio as gr

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()
```

### 2.Data Preprocessing:
```bash
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=1)
```
### 3.Model Training:
```bash
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
```
### 4.Defining the Prediction Function:
```bash
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return 'You are Diabetic' if prediction[0] == 1 else 'Yay, you are Non-diabetic!'
```
### 5.Creating and Launching the Gradio Interface:
```bash
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
    description='Enter your details to predict if you are diabetic or not.'
)

interface.launch(share=True)
```
## Gradio Interface
Gradio is used to create a web interface that allows users to input their health metrics and receive a prediction on whether they are diabetic or not. The interface simplifies the process of using the model and makes it accessible to users without a technical background.

## Conclusion
This project demonstrates the application of machine learning in healthcare. Using an SVM model and Gradio, we can accurately predict diabetes and provide an easy-to-use interface for users. This approach can be extended to other health conditions, making healthcare more accessible and proactive.
