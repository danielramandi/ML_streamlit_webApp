# Machine Learning Web Application

This repository contains a Streamlit-based web application for training and deploying machine learning models. With a user-friendly interface, this web application allows users to upload their dataset, select features, choose a machine learning model, train it, and download the trained model as a pickle file for future use.

The web application offers a variety of machine learning models suitable for both regression and classification tasks. The available models include:
## Classification Models:

    Decision Tree: A decision tree is a flowchart-like structure where each internal node represents a feature(or attribute), each branch represents a decision rule, and each leaf node represents an outcome. The topmost node in a decision tree is known as the root node.
    Logistic Regression: Despite its name, logistic regression is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier.
    Support Vector Machines: Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However, it is mostly used in classification problems.

## Regression Models:

    Ridge Regression: Ridge regression addresses some of the problems of ordinary linear regression by imposing a penalty on the size of the coefficients.
    K-Nearest Neighbors: The target is predicted by local interpolation of the targets associated with the nearest neighbors in the training set.
    Decision Tree Regressor: Decision Trees (DTs) are a non-parametric supervised learning method used for both classification and regression tasks.

## Installation

Clone the repository using the following command:

bash

git clone https://github.com/YOUR_USERNAME/ML-Web-App.git

Navigate to the cloned repository:

bash

cd ML-Web-App

Install the required packages:

pip install -r requirements.txt

Usage

To run the application on your local machine, use the following command:

arduino

streamlit run app.py

This will start the application in your default web browser.

In the application, you can navigate between the Train and Deploy tabs.
Train Tab:

In this tab, you can upload your CSV data file, drop unnecessary columns, select the target column, and choose the model for training based on whether the task is a classification or regression problem. After training the model, you can download the model (along with its associated transformers) as a pickle file.
Deploy Tab:

In this tab, you can upload the pickle file downloaded from the Train tab, and upload a new CSV file with the same features but without the target column. The application will then predict the target values using the uploaded model and allow you to download the CSV file with the predicted target column.
Contributing

Contributions to this project are welcome. You can contribute in several ways including but not limited to:

    Reporting a bug
    Discussing the current state of the code
    Submitting a fix
    Proposing new features
    Adding more machine learning models
