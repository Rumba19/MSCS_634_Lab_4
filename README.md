# Regression Analysis Lab: Linear, Multiple, Polynomial, Ridge, and Lasso Regression

## 1. Overview

This lab implements and compares several regression techniques on the `Diabetes` dataset from `scikit-learn`.  
The notebook walks through a complete workflow:

1. Data preparation and exploration  
2. Simple Linear Regression (single feature)  
3. Multiple Linear Regression (all features)  
4. Polynomial Regression (degrees 1–5)  
5. Regularization: Ridge (L2) and Lasso (L1)  
6. Comprehensive model comparison and analysis  

The goal is to predict disease progression and evaluate how different regression strategies perform, including the effects of model complexity and regularization.

---

## 2. Dataset

- **Source:** `sklearn.datasets.load_diabetes`
- **Samples:** 442 patients  
- **Features (10 standardized predictors):**
  - `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`
- **Target:** Quantitative measure of disease progression one year after baseline

The notebook confirms:
- No missing values
- All features are numeric and already standardized

---

## 3. Environment and Dependencies

### 3.1. Python Version

- Python 3.x

### 3.2. Required Libraries

Install the required packages (if not already available):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3.3. Libraries Used

- `numpy`, `pandas` – data handling and numerical operations  
- `matplotlib`, `seaborn` – visualization  
- `scikit-learn`:
  - `load_diabetes` (dataset)
  - `train_test_split` (data splitting)
  - `LinearRegression`, `Ridge`, `Lasso` (models)
  - `PolynomialFeatures`, `StandardScaler` (preprocessing)
  - `mean_absolute_error`, `mean_squared_error`, `r2_score` (metrics)

---

## 4. How to Run

1. Open a terminal and navigate to the project directory.
2. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open the notebook containing this code (e.g. `regression_lab_diabetes.ipynb`).
4. Run all cells in order (`Kernel` → `Restart & Run All`) to reproduce:
   - Data exploration outputs
   - Model training and metrics
   - All plots and comparison tables

---

## 5. Notebook Structure and Steps

This README corresponds to the following major steps in the notebook.

### STEP 1: Data Preparation

- Loads the Diabetes dataset.
- Wraps features and target into a `pandas` DataFrame.
- Prints:
  - Dataset shape (samples, features)
  - Sample rows
  - Descriptive statistics
  - Missing-values check
  - Target distribution (mean, std, min, max)
  - Data types
- Verifies no cleaning is needed because:
  - No missing values
  - All numeric data

### STEP 2: Simple Linear Regression

- Uses **BMI** (Body Mass Index) as a single predictor:
  - `X_simple = X[:, 2]`
- Splits into train/test sets (80% / 20%, `random_state=42`).
- Trains `LinearRegression` on BMI alone.
- Evaluates using:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² (coefficient of determination)
- Prints:
  - Metrics
  - Model coefficient (slope)
  - Intercept
- Visualizes:
  - Scatter of actual vs BMI
  - Predictions and regression line

This serves as a baseline using a single feature.

### STEP 3: Multiple Regression

- Uses **all 10 features** as predictors.
- Splits into train/test sets (80% / 20%, `random_state=42`).
- Trains a `LinearRegression` model on all features.
- Evaluates with MAE, MSE, RMSE, and R².
- Prints:
  - Intercept
  - Coefficients for each feature in a sorted table (by magnitude)
- Visualizations:
  - Predicted vs actual values (with perfect-prediction line)
  - Residuals vs predicted plot

This step shows how much performance improves when all features are used.

### STEP 4: Polynomial Regression

- Still uses **BMI** as the base feature.
- Tests polynomial degrees: `1, 2, 3, 4, 5`.
- For each degree:
  - Generates polynomial features via `PolynomialFeatures(degree=...)`.
  - Fits `LinearRegression` on the transformed training data.
  - Evaluates on the test set (MAE, MSE, RMSE, R²).
- Stores results in `results_poly_df` and prints a summary table.
- Provides textual analysis:
  - Best degree based on R²
  - Discussions of underfitting/overfitting based on changes in R² across degrees
- Visualizations:
  - For each degree: actual points and polynomial fit curve
  - Line plots:
    - RMSE vs degree
    - R² vs degree

This illustrates how increasing polynomial degree changes model complexity and performance.

### STEP 5: Regularization (Ridge and Lasso)

- Uses **all features** again.
- Standardizes features using `StandardScaler` (critical for regularization).
- Defines a grid of alpha values: `[0.001, 0.01, 0.1, 1, 10, 100]`.
- For each alpha:

  **Ridge Regression (L2):**
  - Trains `Ridge(alpha=alpha)` on standardized training data.
  - Evaluates (MAE, MSE, RMSE, R²) on test data.
  - Stores results in `results_ridge_df`.

  **Lasso Regression (L1):**
  - Trains `Lasso(alpha=alpha, max_iter=10000)` on standardized training data.
  - Evaluates (MAE, MSE, RMSE, R²) on test data.
  - Counts and stores the number of non-zero coefficients (active features).
  - Stores results in `results_lasso_df`.

- Prints Ridge and Lasso summary tables.
- Explains how alpha affects:
  - Under/over-regularization
  - Bias–variance trade-off
  - Coefficient shrinkage and feature selection (Lasso)

- Identifies:
  - Best Ridge alpha (highest R²)
  - Best Lasso alpha (highest R²)

- Visualizations:
  - RMSE vs alpha (log scale) for Ridge and Lasso
  - R² vs alpha (log scale) for Ridge and Lasso
  - Coefficient bar plots:
    - Ridge coefficients at α = 1
    - Lasso coefficients at α = 1 (showing zeros for dropped features)
  - Lasso: number of non-zero coefficients vs alpha
  - Predicted vs actual plots comparing Ridge and Lasso at α = 1

This step demonstrates how regularization can improve generalization and perform feature selection (Lasso).

### STEP 6: Model Comparison and Analysis

- Aggregates results into a combined comparison table for:

  - Simple Linear Regression (BMI only)  
  - Multiple Linear Regression (all features)  
  - Polynomial Regression (degree 2)  
  - Polynomial Regression (degree 3)  
  - Ridge Regression (α = 1)  
  - Lasso Regression (α = 1)  

- For each model, it reports:
  - MAE
  - MSE
  - RMSE
  - R²

- Identifies the **best performing model** based on R²:
  - Prints model name and its key metrics

- Visualizations:
  - Bar charts comparing MAE, RMSE, R² across all models
  - Horizontal bar ranking by R²

- Provides extensive textual discussion:
  - Performance summary per model type
  - Underfitting/overfitting observations
  - Impact of regularization (Ridge vs Lasso)
  - Feature selection behavior of Lasso
  - Overall predictability of the Diabetes dataset given available features
  - Recommendations for:
    - Model choice for deployment
    - Future improvements (feature engineering, more data, ensembles, CV for alpha)
    - Lessons learned about regression and regularization

---

## 6. Key Results and Insights (High-Level)

- Using only BMI provides limited predictive power compared to multivariate models.
- Multiple Linear Regression with all features significantly improves performance over simple linear regression.
- Polynomial Regression shows that adding non-linearity can help, but higher degrees may yield diminishing returns or overfitting.
- Ridge and Lasso:
  - Achieve performance comparable to unregularized multiple regression.
  - Add robustness and interpretability, especially Lasso due to feature selection.
- The best-performing model in the comparison (by R²) is identified at the end of the notebook, along with its MAE, MSE, and RMSE.


## 8. How to Adapt or Extend

- Swap the Diabetes dataset for other regression datasets (e.g., Boston housing, synthetic data).
- Add:
  - Cross-validation for each model
  - ElasticNet (combined L1 + L2)
  - Non-linear models (Random Forests, Gradient Boosting)
- Integrate automated model selection (GridSearchCV / RandomizedSearchCV) for alpha and polynomial degree.

---

## 9. References

- Scikit-learn documentation:
  - `load_diabetes`, regression models, preprocessing, and metrics  
- Standard regression and regularization theory (course textbook/notes)
