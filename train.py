import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

#---LOAD DATA
df = pd.read_csv('data/listings.csv')

df.columns = df.columns.str.strip()

#Clean price by converting from string type to numeric
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
df["price"] = np.log1p(df["price"]) 

#---FEATURES
numeric_features = [
    "accommodates",
    "bathrooms",
    "beds",
    "bedrooms",
    "review_scores_rating",
    "number_of_reviews",
    "availability_365",
    "latitude",
    "longitude"
]

categorical_features = [
    "room_type", "neighbourhood_cleansed"
]

#df = df[features + [target]]

X = df[numeric_features + categorical_features]
y = df["price"]


#---PREPROCESSING
#Impute missing numeric values
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

#Impute missing categorical values
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

#Preprocess numerical and categorical features separately
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

#Fill missing values
#df = df.fillna(df.median(numeric_only=True))


#---MODEL
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs = -1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

#---TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#---TRAIN
pipeline.fit(X_train, y_train)

#---EVALUATE
preds = pipeline.predict(X_test)

mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("R2:", r2)

#---SAVE MODEL
joblib.dump(pipeline, "models/airbnb_model.pkl")