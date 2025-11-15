# Regression Analysis Lab

## Overview
This lab explores multiple regression techniques and regularization methods using the Diabetes Dataset from sklearn.

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Lab Steps

### Step 1: Data Preparation
- Load the Diabetes dataset from sklearn
- Explore features, target values, and data distribution
- Check for missing values and perform data cleaning

### Step 2: Simple Linear Regression
- Implement linear regression using BMI as single feature
- Split data into training (80%) and testing (20%) sets
- Train model and evaluate with MAE, MSE, RMSE, and R²
- Visualize predictions vs actual values with regression line

### Step 3: Multiple Regression
- Use all 10 features as independent variables
- Train model and calculate evaluation metrics
- Display feature coefficients to show importance
- Create predicted vs actual plot and residual plot

### Step 4: Polynomial Regression
- Extend linear regression with polynomial features (degrees 1-5)
- Compare performance across different polynomial degrees
- Demonstrate overfitting with higher degrees
- Visualize how polynomial degree affects model fit

### Step 5: Ridge and Lasso Regression
- Apply Ridge (L2) and Lasso (L1) regularization
- Test multiple alpha values (0.001 to 100)
- Explain how alpha parameter influences model behavior
- Compare Ridge (shrinks coefficients) vs Lasso (eliminates features)
- Visualize coefficient shrinkage and feature selection

### Step 6: Model Comparison and Analysis
- Compare all models using comprehensive metrics table
- Visualize performance across all models
- Identify best performing model
- Discuss overfitting, feature importance, and dataset insights
- Provide recommendations for model selection

## Key Findings
- Multiple features significantly improve predictions (R² ~0.45-0.52)
- Regularization prevents overfitting without major accuracy loss
- Lasso provides automatic feature selection
- Linear models perform well - complex polynomials offer limited benefit

 