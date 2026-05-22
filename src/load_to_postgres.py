import pandas as pd  
from sqlalchemy import create_engine
from config import engine

#--Load csv
df = pd.read_csv('data/listings.csv')

df.columns = df.columns.str.strip()

#Write to Postgres
df.to_sql(
    "listings",
    engine,
    if_exists="replace",
    index=False,
    chunksize=1000,
    method="multi"
)

print("Data loaded to PostgreSQL")