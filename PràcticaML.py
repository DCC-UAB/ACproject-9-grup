import pandas as pd 
from sklearn.model_selection import train_test_split


df= pd.read_csv("student-mat.csv")
df.drop("Dalc",axis=1,inplace=True)


# Dividir el DataFrame
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


