# Calorie Burnt Prediction
This is a machine learning web app which predicts the amount of calories burnt during physical activity based on various factors such as the user's gender, height, weight, and the type of activity performed. The project uses the Random Forest Regression algorithm from the scikit-learn library to achieve this goal.

# Requirements
numpy
pandas
streamlit
matplotlib
seaborn
scikit-learn
# Usage
The app allows users to input their information such as gender, age, height, weight, duration, heart rate, and body temperature through Streamlit sliders and radio buttons. The app then displays the user's information and predicts the amount of calories burnt using the trained machine learning model.

# Data
The app uses a CSV file with the following columns:

Gender
Age
Height
Weight
Duration
Heart Rate
Body Temperature
Calories

# Model
The project uses the Random Forest Regression algorithm from the scikit-learn library for training and prediction.

# Metrics
The model's performance is evaluated using the following metrics:

Mean Squared Error (MSE)
R^2 score
Mean Absolute Error (MAE)
# Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
