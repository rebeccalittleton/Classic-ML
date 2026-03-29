# Linear Regression & Gradient Descent from Scratch

Implementation of linear regression and gradient descent in NumPy, benchmarked 
against scikit-learn on a real-world used car pricing dataset.

## What this covers
- Multivariate linear regression implemented from scratch using NumPy
- Gradient descent with convergence detection and divergence handling
- Learning rate analysis — convergence speed across a range of alpha values
- Ridge and Lasso regularisation with cross-validated hyperparameter tuning
- Residual analysis including Q-Q plot and histogram to validate model assumptions

## Dataset
[Car Data](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) — 
301 used car listings with features including present price, kilometres driven, 
fuel type, and year. Target variable: resale selling price.

## Results

| Model | R² | RMSE |
|---|---|---|
| Multivariate regression (scratch) | 0.5148864936016647 | 18035.61883186126|
| Multivariate regression (sklearn) | 0.42363817283363225| 18728.274015231207 |
| Ridge CV | 0.43272580566733965 | 18580.040830531572 |
| Lasso CV | 0.4236381979319982| 18728.273607458643 |

## Key finding
A linear model captures meaningful structure in the data but is limited by the 
dataset's size and complexity. Regularisation via Ridge and Lasso helped identify 
the most valuable predictors, with next steps pointing toward polynomial features.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Linear_Regression_and_Gradient_Descent.ipynb
```

## Stack
Python · NumPy · Pandas · scikit-learn · Matplotlib
