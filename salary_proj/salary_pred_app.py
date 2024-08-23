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
    data = pd.read_csv("D:/salary_proj/Salary Data.csv")
    
    # Drop any rows with missing target values
    data.dropna(subset=['Salary'], inplace=True)
    
    # Split features and target variable
    X = data.drop('Salary', axis=1)
    y = data['Salary']
    
    return X, y

# Define a function to preprocess the data and train the model
def preprocess_and_train_model(X, y):
    # Define preprocessing steps
    categorical_features = ['Gender', 'Education Level', 'Job Title']
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

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score

# Main function to run the Streamlit app
def main():
    # Title of the app
    st.title('Salary Prediction App')
    
    # Load data
    X, y = load_data()
    
    # Train the model
    model, train_score, test_score = preprocess_and_train_model(X, y)

    

    # User input section
    st.sidebar.header('Enter Your Information')
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    education_level = st.sidebar.selectbox('Education Level', ['Bachelor\'s', 'Master\'s', 'PhD'])
    job_title = st.sidebar.selectbox('Job Title', X['Job Title'].unique())
    years_of_experience = st.sidebar.number_input('Years of Experience', min_value=0, max_value=50, value=5)
    
    # Button to trigger prediction
    if st.sidebar.button('Predict Salary'):
        # Make prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education_level],
            'Job Title': [job_title],
            'Years of Experience': [years_of_experience]
        })
        
        prediction = model.predict(input_data)
        
        # Display prediction
        st.subheader('Salary Prediction')
        st.write('Based on the information provided, the predicted salary is: $', round(prediction[0], 2))

if __name__ == "__main__":
    main()
