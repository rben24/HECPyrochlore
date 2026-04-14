# src/build_models/predict_model.py
import pandas as pd

def predict_and_format(estimator, X_new, return_residuals=False):
    preds = estimator.predict(X_new)
    out = pd.DataFrame({"prediction": preds}, index=X_new.index)
    if return_residuals:
        # if true values provided as column 'true' in X_new, compute residuals
        if "true" in X_new.columns:
            out["residual"] = X_new["true"] - preds
    return out
