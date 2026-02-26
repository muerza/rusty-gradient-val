# Rusty Bargain — Used Car Price Prediction

The used car sales service **Rusty Bargain** needs an application that allows customers to quickly estimate the market value of their car. This project trains and compares multiple Machine Learning models to predict prices based on technical specifications, trim versions, and price history.

## Objective

- Predict the market price of a used car.
- Compare models using three key criteria:
  - **Prediction quality** (RMSE)
  - **Training time**
  - **Prediction time**
- Implement a **Linear Regression** model with **manual Gradient Descent**.

## Dataset

The `car_data.csv` file contains historical used car data with features such as:

- Vehicle type, brand, and model
- Registration year and mileage
- Fuel type, transmission, and horsepower
- Sale price

## Compared Models

| Model | Description |
|--------|-------------|
| **Linear Regression** | Baseline model (sanity check) |
| **Decision Tree** | DecisionTreeRegressor |
| **Random Forest** | RandomForestRegressor |
| **LightGBM** | Gradient Boosting — handles categorical features natively |
| **XGBoost** | Gradient Boosting — requires One-Hot Encoding |
| **Manual Linear Regression** | Custom implementation with Gradient Descent |

## Technologies

- **Python 3**
- **pandas**, **numpy** — data manipulation
- **matplotlib** — visualization
- **scikit-learn** — baseline models, preprocessing, metrics
- **LightGBM** — gradient boosting
- **XGBoost** — gradient boosting
- **joblib** — model persistence

## Project Structure

```text
rusty-bargain-gradient-autos/
├── rusty-gradient-val.ipynb   # Main notebook
├── car_data.csv               # Dataset
├── regressor.json             # Model configuration
└── README.md
```

## How to Run

1. Clone the repository and install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm xgboost joblib
```

2. Open and run the notebook:

```bash
jupyter notebook "rusty-gradient-val.ipynb"
```

## Results

| Model                | R2     | MSE       | RMSE    | Time_s   | Pred_ms |
|----------------------|--------|-----------|---------|----------|---------|
| LGB                  | 0.9027 | 2,161,786 | 1,470.30| 18.566   | 1354.08 |
| LGBMR                | 0.9027 | 2,162,785 | 1,470.64| 13.322   | 870.78  |
| RandomForest         | 0.8960 | 2,312,579 | 1,520.72| 34.553   | 336.47  |
| XGBoost              | 0.8874 | 2,502,547 | 1,581.94| 1.246    | 94.11   |
| DecisionTree         | 0.8295 | 3,788,603 | 1,946.43| 36.059   | 45.07   |
| LinearRegression     | 0.7169 | 6,291,183 | 2,508.22| 2.266    | 97.76   |
| LinearRegression_GD  | 0.6815 | 7,079,618 | 2,660.76| 6.084    | 31.70   |

Each model was evaluated using **RMSE** on the test set, while also measuring training and prediction time. Gradient boosting models (LightGBM, XGBoost) provide the best prediction quality, while linear regression serves as a baseline reference. Detailed results and comparison charts are available inside the notebook.
