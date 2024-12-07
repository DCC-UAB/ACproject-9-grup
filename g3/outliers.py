import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
file_path = 'student-mat.csv'
df = pd.read_csv(file_path)

# Identificar columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Detectar outliers usando IQR para cada columna numérica
outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)

# Crear un DataFrame con el resumen de outliers
outlier_summary_df = pd.DataFrame({
    "Column": outlier_summary.keys(),
    "Outliers": outlier_summary.values(),
    "Total": len(df),
    "Percentage": [100 * count / len(df) for count in outlier_summary.values()]
}).sort_values(by="Percentage", ascending=False)

# Visualizar los boxplots para las columnas con outliers
num_columns = len(outlier_summary_df['Column'])
rows = (num_columns // 3) + (num_columns % 3 > 0)  # Calcular filas necesarias

plt.figure(figsize=(15, 5 * rows))
for i, col in enumerate(outlier_summary_df['Column']):
    plt.subplot(rows, 3, i + 1)
    sns.boxplot(data=df, y=col, color='skyblue')
    plt.title(f'Boxplot - {col}')
    plt.ylabel(col)
    plt.xlabel('')

plt.tight_layout()
plt.show()


from sklearn.preprocessing import PowerTransformer
import numpy as np

# Identificar columnas relevantes para transformación
outlier_cols = ['absences', 'failures', 'studytime', 'famrel', 'Dalc', 'Walc']

# Transformación logarítmica para columnas específicas
log_transform_vars = ['absences', 'Dalc', 'Walc']
for col in log_transform_vars:
    df[col] = np.log1p(df[col])  # log1p para manejar ceros

# Clip de valores extremos para otras columnas
clip_transform_vars = [col for col in outlier_cols if col not in log_transform_vars]
for col in clip_transform_vars:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Normalización Yeo-Johnson para todo el conjunto de datos numérico
numeric_cols = df.select_dtypes(include=[np.number]).columns
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
df[numeric_cols] = power_transformer.fit_transform(df[numeric_cols])

# Separar variables predictoras (X) y objetivo (y)
X = df.drop('G3', axis=1)
y = df['G3']

# Dividir en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=1)

# Escalar los datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelos y reentrenamiento
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modelo de Random Forest como ejemplo
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas de rendimiento
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")
