# Overview
A machine learning model to predict Airbnb listing prices using structured listing data. The goal is to identify key factors that influence pricing and build a reproducible data workflow.

## Tech Stack
* Python
* pandas, numpy
* sci-kit learn
* PostreSQL

## Dataset
The dataset contains Airbnb listing information in Hawaii including:
* Property characteristics (beds, bathrooms, accommodates, etc.)
* Location data (latitude, longitude, neighborhood)
* Review and availability metrics
* Pricing info
Source: InsideAirbnb dataset

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

