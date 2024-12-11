# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GB, AdaBoostRegressor as AB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer

# Semilla para reproducibilidad
SEED = 1

# Función para cargar y preprocesar los datos
def cargar_y_preprocesar_datos(filepath):
    # Cargar los datos desde un archivo CSV
    df = pd.read_csv(filepath)

    # Limpiar valores no deseados (quitar comillas, espacios)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace("'", "").str.strip()

    # Renombrar columnas para mayor claridad
    df.rename(columns={
        'absences': 'absències',
        'failures': 'fracassos',
        'goout': 'sortides',
        'freetime': 'temps',
        'age': 'edat',
        'health': 'salut',
        'G3': 'nota'
    }, inplace=True)

    # Mapear valores binarios (sí/no, etc.)
# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GB, AdaBoostRegressor as AB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer

# Semilla para reproducibilidad
SEED = 1

# 1. Cargar los datos
df = pd.read_csv('student-mat.csv')

# 2. Limpiar valores no deseados
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace("'", "").str.strip()

# 3. Renombrar columnas para mayor claridad
df.rename(columns={
    'absences': 'absències',
    'failures': 'fracassos',
    'goout': 'sortides',
    'freetime': 'temps',
    'age': 'edat',
    'health': 'salut',
    'G3': 'nota'
}, inplace=True)

# 4. Eliminar la columna 'G2'
if 'G2' and 'G1' in df.columns:
    df.drop(columns=['G2','G1'], inplace=True)

# 5. Mapear valores binarios (sí/no, etc.)
bin_map = {
    'yes': 1, 'no': 0,
    'GP': 1, 'MS': 0,
    'F': 1, 'M': 0,
    'U': 1, 'R': 0,
    'LE3': 0, 'GT3': 1,
    'T': 1, 'A': 0
}

cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for col in cat_cols:
    if df[col].dtype == 'object':
        if set(df[col].unique()).issubset(bin_map.keys()):
            df[col] = df[col].map(bin_map)
        else:
            df[col] = LabelEncoder().fit_transform(df[col])

# 6. Normalizar y manejar outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
df[numeric_cols] = power_transformer.fit_transform(df[numeric_cols])

# 7. Separar predictores (X) y objetivo (y)
X = df.drop('nota', axis=1)
y = df['nota']

# 8. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=SEED)

# 9. Escalar las variables predictoras
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 10. Definir modelos
models = {
    'KNN': KNN(),
    'DT': DT(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'Random Forest': RF(random_state=SEED),
    'Gradient Boosting': GB(random_state=SEED),
    'AdaBoost': AB(random_state=SEED),
    'Linear Regression': LinearRegression()
}

# 11. Configurar la validación cruzada (5 folds)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 12. Evaluar modelos con validación cruzada
metrics = {}
predictions = {}

for model_name, model in models.items():
    # Obtener predicciones usando validación cruzada
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    predictions[model_name] = y_pred_cv

    # Calcular métricas
    metrics[model_name] = {
        'MAE': mean_absolute_error(y, y_pred_cv),
        'MSE': mean_squared_error(y, y_pred_cv),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_cv)),
        'R²': r2_score(y, y_pred_cv)
    }

# 13. Convertir las métricas en un DataFrame para visualización
metrics_df = pd.DataFrame(metrics).T.sort_values('MAE')
print(metrics_df)

# 14. Visualizar MAE para todos los modelos
plt.figure(figsize=(10, 6))
metrics_df['MAE'].plot(kind='barh', color='skyblue', edgecolor='black')
plt.title("MAE por Modelo (Validación Cruzada)")
plt.xlabel("MAE")
plt.ylabel("Modelo")
plt.tight_layout()
plt.show()

# 15. Análisis de importancia de características usando Random Forest
rf_model = RF(random_state=SEED)
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_

# Crear un DataFrame de importancia de características
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Visualizar importancia de características
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.title("Importancia de las Características (Random Forest)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 16. Visualizar predicciones finales del mejor modelo (Random Forest)
y_pred_final = rf_model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_final, alpha=0.7, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Valores Reales vs Predicciones (Random Forest)")
plt.tight_layout()
plt.show()
