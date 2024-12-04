import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# Cargar los datos
data = pd.read_csv('student-mat.csv')

# Limpieza de columnas con caracteres no deseados (si aplica)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.replace("'", "").str.strip()

# Separar variables categóricas y numéricas
categoricals = []
numericals = []
for _ in data.columns:
    if data[_].dtype == 'object':
        categoricals.append(_)
    else:
        numericals.append(_)

# Codificar variables categóricas binarias
label_encoder = LabelEncoder()
for _ in categoricals:
    if len(data[_].value_counts()) == 2:  # Solo para variables binarias
        data[_] = label_encoder.fit_transform(data[_])

# Codificar variables categóricas no binarias usando One-Hot Encoding
df_encoded = pd.get_dummies(data, columns=['Mjob', 'Fjob', 'reason', 'guardian'], prefix=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=False)

# Definir variable objetivo y características predictoras
y = df_encoded['G3']
X = df_encoded.drop('G3', axis=1)

# Escalado de datos
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Crear y ajustar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
print(f"Model R2: {model.score(X_test, y_test)}")

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Mostrar métricas
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualización de predicciones vs valores reales
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Línea ideal
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
