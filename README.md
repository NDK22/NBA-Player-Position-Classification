# NBA-Player-Position-Classification

## Introduction

This repository contains code and documentation for a project focused on classifying NBA players into their respective positions using machine learning techniques. The primary goal is to achieve accurate predictions based on player statistics from the 2020-2021 season.

## Tasks

### 1. Classification Using SVM

Implemented a Support Vector Classifier (SVC) on the dataset. Explored hyperparameter tuning and achieved optimal results with a linear kernel and C=3. Evaluated the model using a 75%/25% train/test split and printed training and test set accuracy.

### 2. Confusion Matrix

Printed out a 6x6 confusion matrix to assess the model's performance on a multi-class classification problem with 5 basketball positions (PG, SG, SF, PF, C).

### 3. 10-Fold Stratified Cross-Validation

Utilized the same SVM model with parameters from task 1. Applied 10-fold stratified cross-validation and printed out accuracy for each fold. Calculated the average accuracy across all folds.

### 4. Documentation

Provided documentation explaining the methodology that led to better accuracy. Discussed data preprocessing, feature engineering, model selection, hyperparameter tuning, and other insights.

## Code

The Python code is organized and well-commented. Each task is clearly marked and includes necessary explanations. You can find the main code file in `nba_classification.py`.

## Results

The results of the classification model, including accuracy scores, confusion matrix, and cross-validation results, are documented. Check the results section in the documentation for detailed information.
