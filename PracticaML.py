import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Llegir el fitxer CSV
df = pd.read_csv("student-mat.csv")

# Eliminar la columna "Dalc"
df.drop("Dalc", axis=1, inplace=True)

# Dividir el DataFrame en train i test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separar les característiques (X) i l'etiqueta (Y)
X_train = train_df.drop(columns=['Walc'])
Y_train = train_df['Walc']
X_test = test_df.drop(columns=['Walc'])
Y_test = test_df['Walc']

# Identificar i codificar les columnes categòriques en X
cat_cols = X_train.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder()

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Aplicar una transformació polinòmica de grau 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Entrenar un model de regressió lineal sobre les dades polinòmiques
model_poly = LinearRegression()
model_poly.fit(X_train_poly, Y_train)

# Predicció en el conjunt de prova
Y_pred_poly = model_poly.predict(X_test_poly)

# Calcular l'error quadràtic mig (MSE) com a funció de cost
mse_poly = mean_squared_error(Y_test, Y_pred_poly)

# Imprimir resultats
print("Coeficients del model polinòmic:", model_poly.coef_)
print("Intercepte:", model_poly.intercept_)
print("Error quadràtic mig (MSE):", mse_poly)

# Gràfica de valors reals vs prediccions (Regressió polinòmica)
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred_poly, alpha=0.7, color='blue', label='Prediccions Polinòmiques')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Línia Ideal')

# Configuració de la gràfica
plt.title('Valors reals vs Prediccions - Regressió Polinòmica', fontsize=14)
plt.xlabel('Valors reals (Walc)', fontsize=12)
plt.ylabel('Valors predits (Walc)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
