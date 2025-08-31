Taiwanese Bankruptcy Prediction

Project Overview
This project aims to predict the likelihood of bankruptcy for Taiwanese companies using a dataset of financial ratios and indicators. The primary goal is to build an accurate classification model and to gain insights into which financial metrics are the strongest predictors of bankruptcy.

Data
The dataset contains financial information for a large number of Taiwanese companies. The target variable is Bankrupt?, a binary indicator where 1 signifies bankruptcy and 0 signifies a financially healthy company.

There are 95 independent variables, each representing a different financial aspect of a company, such as profitability, debt ratios, liquidity, and asset turnover.

Methodology
This project follows a systematic machine learning pipeline to handle the challenges of this dataset, particularly its high dimensionality and severe class imbalance.

1. Data Cleaning and Preprocessing
   Correlation Analysis: We first examine the correlation matrix of the 95 independent variables to identify highly correlated features. This is a crucial step to understand redundancy in the dataset and to prepare for dimensionality reduction.
   Manual Feature Removal: Highly correlated variables are manually removed. This helps simplify the dataset and avoids issues like multicollinearity in later stages.
   Dimensionality Reduction with PCA: We use Principal Component Analysis (PCA) to reduce the number of features. PCA transforms the original variables into a smaller set of uncorrelated principal components while retaining most of the variance in the data. The PCs are then used as features for our classification models.

2. Handling Imbalanced Data
   The dataset is highly imbalanced, with a small number of bankrupt companies compared to non-bankrupt ones. To address this, we use the SMOTE (Synthetic Minority Over-sampling Technique) method on the training data. SMOTE creates synthetic examples of the minority class, ensuring our models don't become biased towards predicting the majority class.

3. Model Building and Evaluation
   We conduct a blind classification model comparison on the PCA-transformed and SMOTE-resampled data. We train and evaluate a number of different models to find the best-performing one.

   The models we compare include:
    Logistic Regression
    Random Forest Classifier
    Support Vector Machine (SVM)
    Naive Bayes
    XGBoost
    Neural Network (MLP)

4. Interpretation of Results
After identifying the best-performing model, we use the model's coefficients and the PCA loadings to interpret the results. The key is to link the principal components' predictive power back to the original financial variables that define them. This allows us to explain exactly which financial metrics are most strongly associated with bankruptcy risk.

