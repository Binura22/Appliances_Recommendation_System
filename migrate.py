import pandas as pd
from sqlalchemy import create_engine

# Create a SQLAlchemy engine
engine = create_engine('postgresql://postgres:2001@localhost/Recommendation System')

# Load your CSV data
#df = pd.read_csv('ecommerce_data_new.csv')

# Insert data into the database
#df.to_sql('ecommerce_data', engine, if_exists='replace', index=False)

df = pd.read_csv('user_item_matrix.csv')
df.to_sql('user_item_matrix', engine, if_exists='replace', index=False)