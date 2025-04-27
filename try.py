import pandas as pd
df = pd.read_parquet('outputs/tier1/240A.parquet')
print(df.head())
print(df.columns)
