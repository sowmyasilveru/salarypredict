import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Function to train and save the model, if not already saved
def train_and_save_model(data_path, model_path='salary_prediction_model.pkl'):
    salary_data = pd.read_csv(data_path)
    # Remove rows where 'Salary' is NaN
    salary_data.dropna(subset=['Salary'], inplace=True)


    X = salary_data.drop('Salary', axis=1)
    y = salary_data['Salary']

    
    # Updated part of train_and_save_model function

    categorical_features = ['Gender', 'Education Level', 'Job Title']
    one_hot = OneHotEncoder(handle_unknown='ignore')  # Updated to handle unknown categories
    preprocessor = ColumnTransformer(transformers=[('onehot', one_hot, categorical_features)],
                                 remainder='passthrough')


    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', DecisionTreeRegressor(random_state=42))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

# Check if model is already trained and saved
model_path = 'salary_prediction_model.pkl'
if not os.path.exists(model_path):
    # You would need to provide the path to your dataset here
    train_and_save_model('D:/salary_proj/Salary Data.csv', model_path)

# Load the trained model
model = joblib.load(model_path)

st.title('Salary Prediction App')

# User inputs
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['Male', 'Female'])
education_level = st.selectbox('Education Level', ['Bachelor\'s', 'Master\'s', 'PhD'])
job_title = st.text_input('Job Title')
experience = st.slider('Years of Experience', 0, 50, 1)

if st.button('Predict Salary'):
    input_data = pd.DataFrame([[age, gender, education_level, job_title, experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
    
    predicted_salary = model.predict(input_data)[0]
    st.write(f'Predicted Salary: ${predicted_salary:.2f}')
