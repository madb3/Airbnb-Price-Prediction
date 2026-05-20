# Overview
A machine learning model to predict Airbnb listing prices using structured listing data. The goal is to identify key factors that influence pricing and build a reproducible data workflow.

## Tech Stack
* Python
* pandas, numpy
* sci-kit learn
* PostreSQL

## Dataset
The dataset contains Airbnb property listing information in Hawaii including: 
| Column | Type | Description |
|---|---|---|
| id | INTEGER | Unique listing ID |
| price | FLOAT | Nightly listing price |
| room_type | VARCHAR | Type of property |
| accommodates | INTEGER | Maximum guest count |
| neighborhood_cleansed | VARCHAR | Neighborhood location |

Source: [InsideAirbnb dataset](https://insideairbnb.com/get-the-data/)

## listings

Stores Airbnb property listing information.
| Column | Type | Description |
|---|---|---|
| id | INTEGER | Unique listing ID |
| price | FLOAT | Nightly listing price |
| room_type | VARCHAR | Type of property |
| accommodates | INTEGER | Maximum guest count |
| neighborhood_cleansed | VARCHAR | Neighborhood location |

## Approach
**1. Data Cleansing**
   * Converted price from string to numeric
   * Handles missing values using median/mode imputation
   * Applied log transformation to reduce price skew

**2. Feature Engineering**
   * Selected numerical and categorical features
   * Encoded categorical variables (room type, neighborhood)
   * Included geographic features

**3. Modeling Pipeline**
   * Built a pipeline using:
     - ColumnTransformer
     - SimpleImputer
     - OneHotEncoder
    * Trained a Random Forest regression model

**4. Evaluation**
   * Evaluated using R² score and Mean Absolute Error (MAE)

