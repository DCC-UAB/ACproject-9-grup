import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv("student-mat.csv")

# Eliminar la columna "Dalc"
df.drop("Dalc", axis=1, inplace=True)

# Dividir el DataFrame en train y test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separar características (X) y etiqueta (Y)
X_train = train_df.drop(columns=['Walc'])
Y_train = train_df['Walc']
X_test = test_df.drop(columns=['Walc'])
Y_test = test_df['Walc']

# Identificar y codificar columnas categóricas en X
cat_cols = X_train.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder()

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Crear el modelo ordinal
ordinal_model = OrderedModel(Y_train, X_train, distr='logit')  # 'logit' o 'probit'
ordinal_result = ordinal_model.fit(method='bfgs')

# Resumen del modelo
print(ordinal_result.summary())

# Predicción en el conjunto de prueba
predicted = ordinal_result.predict(X_test)

# Redondear las predicciones para obtener categorías
predicted_categories = predicted.idxmax(axis=1)

# Comparar predicciones con valores reales
accuracy = (predicted_categories == Y_test).mean()
print(f'Accuracy: {accuracy:.2f}')

# Gráfica de valores reales vs predicciones
plt.figure(figsize=(8, 6))

# Scatter plot para los valores reales vs predichos
plt.scatter(Y_test, predicted_categories, alpha=0.7, color='blue', label='Predicciones Ordinales')

# Línea ideal para referencia
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Línea Ideal')

# Configuración del gráfico
plt.title('Valores reales vs Predicciones - Regresión Ordinal', fontsize=14)
plt.xlabel('Valores reales (Walc)', fontsize=12)
plt.ylabel('Valores predichos (Walc)', fontsize=12)
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()
