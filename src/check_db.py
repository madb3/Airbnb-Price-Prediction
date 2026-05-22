import pandas as pd
from config import engine


df = pd.read_sql("SELECT * FROM listings LIMIT 10", engine)

print(df.head())