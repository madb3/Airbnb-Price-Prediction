import pandas as pd
import numpy as np
import joblib
from config import engine

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#LOAD MODEL and TEST DATA
pipeline, X_test, y_test = joblib.load("models/artifacts.pkl")
df = pd.read_sql("SELECT * FROM listings", engine)

#PREDICT
preds = pipeline.predict(X_test)

y_true = np.expm1(y_test)
y_pred = np.expm1(preds)

#METRICS
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

#SAVE RESULTS
results = pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

results.to_csv("models/model_metrics.csv", index=False)