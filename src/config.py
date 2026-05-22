import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

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
    "room_type",
    "neighbourhood_cleansed"
]

