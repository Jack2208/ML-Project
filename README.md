# ML-Project - Password Strength Classifier

## Overview
This project aims to develop a machine learning model to classify passwords based on their strength. 

## Dataset
The dataset used for this analysis contains 670k passwords and it's found at the link: https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset?resource=download and it's derived from the 000webhost leak and has been labeled using various commercial password strength meters, combined in the tool PARS developed by Georgia Tech.

## Model
The final model used is a RandomForestClassifier. Additional models experimented with include Logistic Regression and other ensemble methods. Hyperparameter tuning was performed using GridSearchCV.

## Evaluation
The model's performance is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC metrics. A confusion matrix is also provided to visualize the model's performance.

## Group
This project has been done by Giancarlo Pederiva, Alessio Vinci and Nicola Scema.

