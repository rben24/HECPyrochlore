# src/build_models/evaluate_model.py
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import math

def evaluate_regressor(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
