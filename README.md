# Phishing-URL-Detection

This project is a Phishing URL Detection System built using Python, Streamlit, and machine learning techniques. It analyzes the characteristics of a URL and predicts whether it is legitimate or phishing. The system is based on a pre-trained Gradient Boosting Classifier model and uses advanced feature extraction to distinguish between phishing and legitimate websites.

## Table of Contents
Overview
Features
Prerequisites
Dataset

## Overview
Phishing attacks are a type of cybercrime where attackers disguise as legitimate websites to steal sensitive data such as login credentials and financial information. The aim of this system is to detect such malicious URLs using a machine learning model that analyzes the structure and features of a given URL.

## Features
Phishing URL Classification: Input a URL and the model will classify it as phishing or legitimate.
Exploratory Data Analysis (EDA): Includes feature correlation and distribution analysis.
Model Results: Comparison of different machine learning models with key performance metrics.
Feature Importance: Analysis of which features contribute most to predicting phishing URLs.


## Prerequisites
Python 3.7+
Streamlit
Scikit-learn
Pandas
Numpy
Matplotlib
Seaborn

## Dataset
The phishing.csv file contains the data used for training and testing. The dataset includes several features such as URL length, use of HTTPS, presence of IP addresses, and more, which help distinguish between phishing and legitimate URLs.
