# Introduction
Three scripts: catmodel, predict_price, and predict_cli. predict_price and predict_cli utilize pre-trained parameters, while predict_cli is a command-line interface that accepts vehicle information and returns prediction results.
catmodel is the code for training the model. Adjust the configuration region based on your data and modify the dataset path accordingly.

## 0. Workflow
1. Clean the data first, ensuring all fields contain only numbers without characters, then perform feature engineering.
2. Train the model using the cleaned data and refine it based on historical fluctuations in the used car market.
3. Prediction script `predict_price.py`: Load the model + meta, apply identical feature processing to each vehicle → generate price and range.
`predict_cli.py`: Command-line interface that interacts with you, assembles a single input line, passes it to `predict_price`, and formats the results for better readability.

## 1. Environment & Dependencies
Python Version: Recommended 3.8+ (your current environment is fine).
Main Packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `catboost`
- `joblib`

## 2. Parameter Download
In Github, find a file folder named "used_car_price_v4", download the entire folder