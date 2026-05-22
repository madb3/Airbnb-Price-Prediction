# Overview
A machine learning model to predict Airbnb listing prices using structured listing data. The goal is to identify key factors that influence pricing and build a reproducible data workflow.

## Tech Stack
* Python
* pandas, numpy
* sci-kit learn
* PostreSQL
* SQLAlchemy
* Joblib

## Dataset
The dataset contains Airbnb property listing information in Hawaii including: 

**Numeric features:**
- accomodates
- bathrooms
- beds
- bedrooms
- review_scores_rating
- number_of_reviews
- availability_365
- latitude
- longitude

**Categorical features:**
- room_type
- neighbourhood_cleansed

Source: [InsideAirbnb dataset](https://insideairbnb.com/get-the-data/)

## How it works
### 1. Data Loading
Data is pulled directly from PostgreSQL
```df = pd.read_sql("SELECT * FROM listings", engine)```

### 2. Data Cleaning
* Removes $ and commas from price
* Converts price to numeric
* Drops missing values
* Applies log transformation

### 3. Training Pipeline
* Train/test split (80/20)
* Missing value imputation
* One-hot encoding for categorical variables
* Random Forest regression model

### 4. Model Training
```pipeline.fit(X_train, y_train)```

### 5. Model Saving
```joblib.dump((pipeline, X_test, y_test), "models/artifacts.pkl")```


## Future Improvements
* Build interactive dashboard (Tableau)


