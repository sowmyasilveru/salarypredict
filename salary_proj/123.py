import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load and preprocess the data
@st.cache(allow_output_mutation=True)
def load_data(data_path):
    data = pd.read_csv(data_path)
    # Convert 'Job Title' to string to ensure consistency
    data['Job Title'] = data['Job Title'].astype(str)
    return data

def train_and_save_model(data, model_path='salary_prediction_model.pkl'):
    X = data.drop('Salary', axis=1)
    y = data['Salary']
    
    categorical_features = ['Gender', 'Education Level', 'Job Title']
    one_hot = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(transformers=[('onehot', one_hot, categorical_features)],
                                     remainder='passthrough')
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', DecisionTreeRegressor(random_state=42))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)

# Path to your dataset
data_path = 'D:/salary_proj/Salary Data.csv'
# Load data and train model if not already saved
if not os.path.exists('salary_prediction_model.pkl'):
    data = load_data(data_path)
    train_and_save_model(data)

# Streamlit app
st.title('Salary Prediction App')

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

# Assuming you have loaded your data as 'data'
data = load_data(data_path)
# Filter out non-string and 'nan' values from job titles
unique_job_titles = [title for title in data['Job Title'].unique() if type(title) == str and title.lower() != 'nan']
unique_job_titles.sort()  # This now works without error

# User inputs
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', data['Gender'].unique())
education_level = st.selectbox('Education Level', data['Education Level'].unique())
job_title = st.selectbox('Job Title', unique_job_titles)
experience = st.slider('Years of Experience', 0, 50, 1)

if st.button('Predict Salary'):
    input_df = pd.DataFrame([[age, gender, education_level, job_title, experience]],
                            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
    
    predicted_salary = model.predict(input_df)[0]
    st.write(f'Predicted Salary: ${predicted_salary:.2f}')
