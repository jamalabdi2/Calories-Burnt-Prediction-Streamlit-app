import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import xgboost
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')
import os


# streamlit ui 
# title
st.title('Welcome to my My web App:flag-ke:')
st.title('Calories Burnt Prediction :mechanical_arm:')
st.write("This Web App, aims to predict the amount of calories burnt during physical activity based on various factors such as the your gender, height, weight, and the type of activity performed. The project uses machine learning algorithms to achieve this goal.")
st.sidebar.header("Please Provide The Following Data")
#Gender	Age	Height	Weight	Duration	Heart_Rate	Body_Temp	Calories
def user_input():
    """
    Collect user inputs through Streamlit sliders and radio button and return the information as a Pandas dataframe.

    Returns:
    Pandas DataFrame: A dataframe containing the collected information of age, BMI, heart rate, duration, body temperature, and gender. Gender is transformed to a numerical value (0 for male and 1 for female/non-binary) for easier processing.

    Inputs:
    - Age (int): User's age input through a Streamlit slider with a range from 5 to 100 and a default value of 20.
    - BMI (int): User's BMI input through a Streamlit slider with a range from 10 to 40 and a default value of 20.
    - Heart Rate (int): User's heart rate input through a Streamlit slider with a range from 60 to 130 and a default value of 70.
    - Duration (int): User's duration input through a Streamlit slider with a range from 0 to 60 and a default value of 20.
    - Body Temperature (float): User's body temperature input through a Streamlit slider with a range from 36 to 40 and a default value of 37.
    - Gender (str): User's gender input through a Streamlit radio button with options of Male and Female.
    """
    global age,bmi,heart_rate,duration,body_temperature,gender,name
    name = st.sidebar.text_input('Name:')
    age = st.sidebar.slider('Age',5,100,20)
    bmi = st.sidebar.slider('BMI',10,40,20)
    heart_rate = st.sidebar.slider('Heart Rate',60,130,70)
    duration = st.sidebar.slider('Duration in Minute',0,60,20)
    body_temperature = st.sidebar.slider('Body Temperature',36,40,37)
    gender = st.sidebar.radio('Gender: ',('Male','Female'))


    
   #put information collected from user to a dictionary(for display purpose only)
    user_data_dictionary = {
        'Name':name,
        'Gender':gender,
        'Age':age,
        'BMI':bmi,
        'Heart Rate':heart_rate,
        'Body Temperature':body_temperature,
        'Duration':duration,
    }

    #{'Gender':{'male':0,'female':1}}
    # label encoding for gender 
  
    if gender =='Male':
        gender = 1
    else:
        gender=0

    #Gender	Age	Height	Weight	Duration	Heart_Rate	Body_Temp
    model_features_dictionary= {
        'Gender':gender,
        'Age':age,
        'BMI':bmi,
        'Duration':duration,
        'Heart Rate':heart_rate,
        'Body Temperature':body_temperature,
    }

    #turn the dictionaries to a pandas dataframe
    information_collected = pd.DataFrame([user_data_dictionary])
    model_input_features = pd.DataFrame([model_features_dictionary])
    return information_collected,model_input_features

user_display,model_features = user_input()
st.header("Your information")
st.dataframe(user_display)


# data loading for model training
def load_data(path: str) -> pd.DataFrame:
    """Load data from the specified file path and return a Pandas DataFrame.

    Parameters:
    path (str): The file path of the data to be loaded.

    Returns:
    pd.DataFrame: The loaded data.
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f'The path {path} does not exist')

base_dir =os.getcwd()
calories_path = os.path.join(base_dir,'calories.csv')
exercise_path = os.path.join(base_dir,'exercise.csv')

try:
    calories_dataset = load_data(calories_path)
except FileNotFoundError as e:
    print(e)
try:
    exercise_dataset = load_data(exercise_path)
except FileNotFoundError as e:
    print(e)


def drop_column(df:pd.DataFrame, column_name:str) ->pd.DataFrame:

    """
    This function returns a Dataframe with specified column removed.

    Parameters:
    df (pd.Dataframe): The Dataframe to be modified
    column_name: The name of a column to be removed.

    Returns:
    pd.DataFrame: the modified DataFrame
    """
    return df.drop(column_name,axis=1)

calories_dataset = drop_column(calories_dataset,'User_ID')
exercise_dataset = drop_column(exercise_dataset,'User_ID')
# append calories to exercise dataset to make it one
##seperate a Calories column from calories_data and then append to exercise_data
exercise_dataset['Calories'] = calories_dataset


#add BMI Column to the dataset

def calculate_BMI(df:pd.DataFrame) ->pd.DataFrame:
    """
    Calculate and round the Body Mass Index (BMI) for each row in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame that contains the "Height" and "Weight" columns.
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional "BMI" column.
    """
    df['BMI'] = df['Weight'] / ((df['Height']/100)**2)
    df['BMI'] = round(df['BMI'],2)
    return df


#exercise_dataset = [calculate_BMI(data) for data in exercise_dataset]
exercise_dataset = calculate_BMI(exercise_dataset)
#label encoding

def encode_label(df: pd.DataFrame):
    encoder = LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])
    return df

exercise_dataset = encode_label(exercise_dataset)


def split_exercise_dataset(exercise_dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 1):
    """
    Split the given exercise dataset into training and testing data and labels.
    
    Parameters:
    exercise_dataset (pd.DataFrame): The input exercise dataset that contains the "Calories" column.
    test_size (float): The size of the testing data, with a default value of 0.2.
    random_state (int): The seed used by the random number generator, with a default value of 1.
    
    Returns:
    tuple: A tuple containing the training data, testing data, training labels, and testing labels.
    """
    features = exercise_dataset.drop(['Calories','Height','Weight'],axis=1).values
    target = exercise_dataset["Calories"].values
    train_data, test_data, train_labels, test_labels = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return train_data, test_data, train_labels, test_labels

training_data, testing_data, training_labels, testing_labels = split_exercise_dataset(exercise_dataset)

def evaluate_random_forest_regressor(training_data, training_labels, testing_data, testing_labels):
    RandomForest_model = RandomForestRegressor()
    RandomForest_model.fit(training_data,training_labels)
    model_prediction = RandomForest_model.predict(testing_data)
    model_MAE = round(mean_absolute_error(testing_labels,model_prediction),2)
    model_r2score = round(r2_score(testing_labels,model_prediction),1)
    model_MSE = round(mean_squared_error(testing_labels,model_prediction),2)
    model_results = {
        'Model Name':'Random Forest Regressor',
        'R2 Score':model_r2score,
        'Mean Absolute error': model_MAE,
        'Mean Squared Error':model_MSE
    }
    return RandomForest_model, model_results

model,results = evaluate_random_forest_regressor(training_data, training_labels, testing_data, testing_labels)
result_df = pd.DataFrame([results])
st.header('Linear Model Training Information')
st.write(result_df)
#st.write('-------------------- Prediction ----------------')

#add height and weight from bmi
#st.write('Prediction is: ',prediction)

st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  bar.progress(i + 1)
  time.sleep(0.01)
prediction = model.predict(model_features)
prediction_title = f"<p style='font-family:sans-serif; color:Black; font-size: 42px;'  >Model Prediction is: {prediction[0]} Kilocalories</p>"
st.markdown(prediction_title,unsafe_allow_html=True)
#st.header('Model Prediction is: ',prediction[0])

print('This is a prediction: ',prediction[0])



