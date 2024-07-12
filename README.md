## HEART DISEASE PREDICTION

## Overview
This project aims to predict the presence of heart disease in a patient using logistic regression. The model is trained and evaluated on the `heart.csv` dataset, which contains various health metrics.

## Software requirement
Pycharm

## Dataset
The dataset used for this project is `heart.csv`. It includes the following features:

1. **age**: Age of the patient
2. **sex**: Gender of the patient
3. **cp**: Chest pain type
4. **trestbps**: Resting blood pressure
5. **chol**: Serum cholesterol in mg/dl
6. **fbs**: Fasting blood sugar > 120 mg/dl
7. **restecg**: Resting electrocardiographic results
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: The slope of the peak exercise ST segment
12. **ca**: Number of major vessels (0-3) colored by fluoroscopy
13. **thal**: Thalassemia
14. **target**: Diagnosis of heart disease (1: presence, 0: absence)

Predicting heart disease using logistic regression involves several key steps
1. Data collection
2. Data preprocessing
3. Model training
4. Model  evaluation
5. Prediction

## Data Collection
Gather the dataset that contains patient health metrics and the target variable indicating the presence or absence of heart disease. For this project, we use the heart.csv file.

## Data Preprocessing

Load the Data: Load the dataset into a pandas DataFrame.
Exploratory Data Analysis (EDA): Analyze the data to understand its structure, identify patterns, and detect any anomalies. Common steps include:
Summary statistics.
Visualizations like histograms, box plots, and correlation matrices.
Handling Missing Values: Check for and handle any missing values in the dataset. Common methods include imputation or removal of missing data points.
Feature Selection/Engineering: Select the relevant features for the model. Sometimes new features are created from existing ones to improve model performance.
Data Normalization/Standardization: Scale the features to a standard range (e.g., 0-1) or normalize them to have a mean of 0 and a standard deviation of 1, if necessary.
Splitting the Data: Split the data into training and testing sets. Typically, an 80-20 or 70-30 split is used


## Model Training

Logistic Regression Model: Instantiate a logistic regression model from a library like Scikit-learn.
Training the Model: Fit the logistic regression model to the training data using the fit method.
Hyperparameter Tuning: Optionally, tune the model's hyperparameters to improve performance. This can be done using techniques like grid search or randomized search.

## Model Evaluation

Predictions on Test Set: Use the trained model to make predictions on the test set using the predict method.
Performance Metrics: Evaluate the model's performance using metrics such as:
Accuracy: The proportion of correct predictions.
Precision: The proportion of true positive predictions among all positive predictions.
Recall (Sensitivity): The proportion of true positive predictions among all actual positive cases.
F1 Score: The harmonic mean of precision and recall.

## Prediction

Making Predictions: Use the trained model to predict the presence of heart disease in new, unseen data.
Interpreting Results: Interpret the prediction results to make informed decisions. For instance, if a patient is predicted to have heart disease, further medical examination and intervention may be necessary.

![Screenshot (103)](https://github.com/user-attachments/assets/f50afae7-906d-4390-9543-36b79fa774e6)

## Results

The logistic regression model achieves an accuracy of 78.2% on the test set
