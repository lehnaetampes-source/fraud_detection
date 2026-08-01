import pandas as pd
df = pd.read_csv("https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv")
print(df.columns.tolist())
print(df.head(2))