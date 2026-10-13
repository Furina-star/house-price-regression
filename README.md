# House Price Prediction with Linear Regression

A hands-on machine learning project focused on understanding regression fundamentals through house price prediction using Python and scikit-learn.

The project progresses from a relatively clean dataset to a more realistic dataset requiring additional preprocessing, feature engineering, and regularization.

## Project Overview

The objective is to develop, evaluate, and improve regression models while understanding the complete machine learning workflow.

The project is divided into two phases:

- **Phase A: California Housing** — Build a baseline regression pipeline, perform exploratory data analysis, evaluate predictions, and experiment with feature engineering and regularization.
- **Phase B: Ames Housing** — Apply regression techniques to a more challenging dataset containing missing values, categorical features, and skewed distributions.

## Technologies Used

- **Python** — Primary programming language
- **pandas / NumPy** — Data manipulation and numerical operations
- **Matplotlib / Seaborn** — Data visualization
- **scikit-learn** — Regression models, preprocessing, and evaluation
- **Jupyter Notebook** — Interactive experiments and analysis
- **PyCharm** — Development environment

## Project Structure

```text
house-price-regression/
├── data/
├── notebooks/
│   └── 01_california_housing.ipynb
│   └── 02_ames_housing.ipynb 
├── reports/
│   └── figures/
├── src/
├── .gitignore
├── requirements.txt
└── README.md
```

- `data/` — Local datasets used in experiments.
- `notebooks/` — Exploratory analysis, model training, and evaluation.
- `reports/figures/` — Selected visualizations for documentation.
- `src/` — Optional reusable Python functions.
- `requirements.txt` — Python dependencies needed to reproduce the project.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Furina-star/house-price-regression.git
cd house-price-regression
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Use a Python version compatible with the pinned dependencies.

### 4. Run the Notebook

1. Open the project in PyCharm or another Jupyter-compatible environment.
2. Select the project's virtual environment as the Python interpreter and notebook kernel.
3. Open `notebooks/01_california_housing.ipynb`.
4. Execute the notebook cells in order.

The California Housing dataset is retrieved using scikit-learn and may require an internet connection on the first run.

---

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

The dataset represents census block groups rather than individual properties. Its capped target values and geographic structure limit the interpretation and generalizability of predictions.

The reported improvement is specific to the evaluated dataset and split. Additional evaluation would be required before practical deployment.

---

## Phase B: Ames Housing

**Status: Not started**

This phase will extend the regression workflow to the Ames Housing dataset from Kaggle.

The planned work includes missing-value handling, categorical encoding, target transformations, preprocessing pipelines, regularization, and cross-validation.

Results will be documented after completing the experiments.

---

## Project Goals

By completing both phases, this project aims to demonstrate:

- Exploratory data analysis and preprocessing
- Linear Regression fundamentals
- Feature engineering and model interpretation
- Regularization using Ridge, Lasso, and ElasticNet
- Cross-validation and hyperparameter tuning
- Model evaluation using R², MAE, RMSE, and residual analysis
- Reproducible machine learning workflows