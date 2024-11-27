import pandas as pd 
df= pd.read_csv("student-mat.csv")
df= df.drop("Dalc",axis=1,inplace=True)
