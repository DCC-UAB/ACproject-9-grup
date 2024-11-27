import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Leer el archivo CSV
df = pd.read_csv("student-mat.csv")

# Eliminar la columna "Dalc"
df.drop("Dalc", axis=1, inplace=True)

# Dividir el DataFrame en train y test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separar las características (X) y la etiqueta (Y) del conjunto de entrenamiento
X_train = train_df.drop(columns=['Walc'])
Y_train = train_df['Walc']

# Identificar las columnas categóricas en X
cat_cols = X_train.select_dtypes(include=['object']).columns

# Aplicar OrdinalEncoder a las columnas categóricas
encoder = OrdinalEncoder()
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])

# Imprimir resultados
print("Primeras filas de X_train después del encoding:")
print(X_train.head())

print("\nPrimeras filas de Y_train:")
print(Y_train.head())
