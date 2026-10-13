## Phase A: California Housing

### Objective

Establish a Linear Regression baseline using the California Housing dataset, evaluate predictive performance, and investigate improvements through feature engineering and regularization.

### Dataset

- **Source:** scikit-learn California Housing
- **Observations:** 20,640 census block groups
- **Original features:** 8 numerical features
- **Target:** Median house value (USD, stored in $100,000 units)
- **Data split:** 80% training, 20% testing (`random_state=42`)

### Methodology

1. Explored feature distributions, missing values, correlations, and geographic price patterns.
2. Standardized numerical features using `StandardScaler`.
3. Trained a baseline `LinearRegression` model.
4. Evaluated model performance using R², MAE, RMSE, and residual analysis.
5. Engineered `BedroomRatio` from average bedrooms and average rooms.
6. Compared Linear Regression, Ridge, and Lasso using 5-fold cross-validation.
7. Selected Ridge Regression based on cross-validation MAE and evaluated it on the held-out test set.

### Cross-Validation Results

| Model | Mean CV MAE |
|---|---:|
| Linear Regression | $52,906.07 |
| Linear Regression + Feature Engineering | $52,326.74 |
| Ridge + Feature Engineering | $52,306.44 |
| Lasso + Feature Engineering | $52,318.10 |

### Held-Out Test Results

| Metric | Baseline Linear Regression | Final Ridge Regression |
|---|---:|---:|
| R² | 0.5758 | 0.5942 |
| MAE | $53,320.01 | $52,556.97 |
| RMSE | $74,558.14 | $72,925.19 |

Ridge Regression with the engineered bedroom ratio achieved the lowest cross-validation MAE among the tested configurations. On the held-out test set, it reduced MAE by approximately 1.43% compared with the original baseline.

### Key Findings

- Median income showed the strongest positive linear correlation with median house value.
- Geographic features exhibited relationships not fully represented by their individual Pearson correlations.
- The target variable contains capped values, limiting prediction and evaluation accuracy for high-value areas.
- Feature engineering improved cross-validation performance more than regularization in this experiment.
- Residual analysis revealed systematic patterns and large prediction errors.
- Ridge and Lasso produced similar cross-validation performance, with Lasso retaining all nine features at its selected regularization strength.

### Limitations

The dataset represents census block groups rather than individual properties. Its capped target values and geographic structure limit the interpretation and generalizability of predictions. Additional evaluation would be required before practical deployment.