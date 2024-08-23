import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Define a function to load the data
def load_data():
    # Load the dataset
    data = pd.read_csv("salary data.csv")
    
    # Drop any rows with missing target values
    data.dropna(subset=[ 'Salary'], inplace=True)
    
    return data

# Define a function to preprocess the data and train the models for each job title
def preprocess_and_train_models(data):
    models = {}
    job_titles_with_sufficient_data = []
    
    # Iterate over unique job titles
    for job_title in data['Job Title'].unique():
        # Filter data for the current job title
        job_data = data[data['Job Title'] == job_title]
        
        # Check if there are sufficient samples for training
        if len(job_data) < 2:
            continue
        
        job_titles_with_sufficient_data.append(job_title)
        
        # Split features and target variable
        X = job_data.drop('Salary', axis=1)
        y = job_data['Salary']
        
        # Define preprocessing steps
        categorical_features = ['Gender', 'Education Level']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        numerical_features = ['Years of Experience']
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Define the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(max_depth=10, min_samples_split=8, min_samples_leaf=1))
        ])

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the trained model for the current job title
        models[job_title] = model

    return models, job_titles_with_sufficient_data

# Define a function to adjust salary based on education level and years of experience
def adjust_salary(prediction, education_level, years_of_experience):
    # Adjust salary based on education level
    if education_level == "Master's":
        prediction *= 1.05  # Increase salary by 5% for Master's degree
    
    # Adjust salary based on years of experience
    prediction += 1000 * years_of_experience  # Increase salary by $1000 for each year of experience
    
    return prediction

# Main function to run the Streamlit app
def main():
    # Title of the app
    st.title('Salary Prediction App')
    
    # Load data
    data = load_data()
    
    # Preprocess the data and train models for each job title
    models, job_titles_with_sufficient_data = preprocess_and_train_models(data)

    # User input section
    st.sidebar.header('Enter Your Information')
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    education_level = st.sidebar.selectbox('Education Level', ['Bachelor\'s', 'Master\'s'])
    selected_job_title = st.sidebar.selectbox('Job Title', job_titles_with_sufficient_data)
    years_of_experience = st.sidebar.number_input('Years of Experience', min_value=0, max_value=50, value=5)
    
    # Button to trigger prediction
    if st.sidebar.button('Predict Salary'):
        # Get the model for the selected job title
        model = models.get(selected_job_title)
        
        if model is not None:
            # Make prediction
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Education Level': [education_level],
                'Years of Experience': [years_of_experience]
            })

            prediction = model.predict(input_data)[0]
            
            # Adjust salary based on education level and years of experience
            adjusted_prediction = adjust_salary(prediction, education_level, years_of_experience)

            # Display prediction
            st.subheader('Salary Prediction')
            st.write('Based on the information provided, the predicted salary for {} is: '.format(selected_job_title), int(round(adjusted_prediction, 2)))
        else:
            st.write('No model available for the selected job title.')

if __name__ == "__main__":
    main()
