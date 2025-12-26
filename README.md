# Obesity prediction Using Machine Learning
## Project Overview

The rapid rise in obesity has become a major global public health concern. This project focuses on building a machine learning–based classification system that predicts an individual’s obesity level based on demographic, lifestyle, and health-related attributes. The model classifies individuals into 7 obesity categories, ranging from Insufficient Weight to Obesity Type III.

The project covers the entire machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, dimensionality reduction, and hyperparameter tuning.

## Dataset Description

Total records: 2,111 (after cleaning: 2,104)

Features: 16 input features + 1 target variable

Target variable: NObeyesdad (7 obesity classes)

Key Features Include:
Age, Height, Weight, Physical activity (FAF), Water intake (CH2O), Eating habits (FCVC, NCP, CAEC), Lifestyle factors (SMOKE, CALC, SCC, TUE), Transportation mode (MTRANS)

The dataset contains both real and synthetically generated data.

## Exploratory Data Analysis (EDA)

Visualized distributions using histograms, boxplots, and count plots

Analyzed relationships using pairplots and correlation heatmaps

Detected outliers using Z-score, IQR, Isolation Forest, and DBSCAN

DBSCAN was selected for final outlier removal, reducing noise while preserving data integrity

## Data Preprocessing

No missing values found in the dataset

Categorical features encoded using One-Hot Encoding

Numerical features standardized using StandardScaler

Outliers removed using DBSCAN clustering

Label encoding applied to the target variable

## Dimensionality Reduction

LDA (Linear Discriminant Analysis): Used for class separability and visualization

PCA (Principal Component Analysis): Used for variance-based feature projection

LDA provided better class separation compared to PCA

## Machine Learning Models Used

The following classification models were implemented and compared:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Classifier (SVC)

Decision Tree

Random Forest

Gradient Boosting

Naive Bayes

XGBoost

## Model Evaluation

Models were evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrices

Best Performing Models (Before Tuning):

XGBoost, Gradient Boosting, Random Forest: ~95% F1-score

Naive Bayes showed poor performance due to independence assumptions

## Hyperparameter Tuning

Applied GridSearchCV with 5-fold cross-validation

Significant performance improvement observed after tuning

Best Final Model:

SVC (Linear Kernel)

F1-score: 0.98

XGBoost, Random Forest, and Gradient Boosting achieved ~0.95 F1-score

## Key Results

Weight and physical activity were the strongest predictors of obesity

Ensemble models performed consistently well

LDA helped preserve performance while reducing dimensionality

Hyperparameter tuning played a critical role in achieving near-perfect classification

## Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost

## Conclusion

This project demonstrates a complete and well-structured machine learning workflow for multi-class obesity prediction. By combining effective EDA, robust preprocessing, dimensionality reduction, and advanced model tuning, the system achieves high accuracy and strong generalization, making it suitable for real-world health analytics applications.
