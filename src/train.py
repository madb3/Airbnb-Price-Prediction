import pandas as pd
import numpy as np
import joblib

from config import engine, numeric_features, categorical_features

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

#LOAD FROM POSTGRESQL
df = pd.read_sql("SELECT * FROM listings", engine)

#df.columns = df.columns.str.strip()

# CLEAN TARGET
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])

#Log transition
df["price"] = np.log1p(df["price"])

# FEATURES/TARGET
X = df[numeric_features + categorical_features]
y = df["price"]

#TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#PREPROCESS
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# MODEL
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# TRAIN MODEL
pipeline.fit(X_train, y_train)

# SAVE MODEL
joblib.dump((pipeline, X_test, y_test), "models/artifacts.pkl")

print("Model trained and saved.")