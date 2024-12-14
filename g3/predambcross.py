import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GB, AdaBoostRegressor as AB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import PowerTransformer

# Semilla
SEED = 1


df = pd.read_csv('student-mat.csv')

# Netejar valors no desitjats
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace("'", "").str.strip()

# Renombrar columnes per simplicitat
df.rename(columns={
    'absences': 'absències',
    'failures': 'fracassos',
    'goout': 'sortides',
    'freetime': 'temps',
    'age': 'edat',
    'health': 'salut',
    'G3': 'nota'
}, inplace=True)

# Netejar valors no desitjats
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace("'", "").str.strip()

# Identificar columnas relevantes para transformación
outlier_cols = ['absències', 'fracassos', 'studytime', 'famrel', 'Dalc', 'Walc']

# Transformación logarítmica para columnas específicas
log_transform_vars = ['absències', 'Dalc', 'Walc']
for col in log_transform_vars:
    df[col] = np.log1p(df[col])  # log1p para manejar ceros

# Clip de valores extremos para otras columnas
clip_transform_vars = [col for col in outlier_cols if col not in log_transform_vars]
for col in clip_transform_vars:
    Q1 = df[col].quantile(0.22)
    Q3 = df[col].quantile(0.78)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Separar variables (X: predictives, y: objectiu)
X = df.drop('nota', axis=1)
y = df['nota']

# Dividir entrenament i prova
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=1)

# Escalar dades
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

                                                              
# Definir models
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

# Hiperparàmetres per a cada model
# Definir los hiperparámetros para cada modelo
params = {
    'KNN': {'n_neighbors': [i for i in range(3, 50)]},
    'DT': {'max_depth': [i for i in range(1, 25)]},
    'ElasticNet': {'alpha': [i for i in range(0, 50)], 'l1_ratio': [0.1, 0.5, 0.9]},
    'SVR': {'kernel': ['linear', 'poly', 'rbf'], 'C': [i for i in range(1, 101)], 'epsilon': [0.01, 0.1]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30], 'max_features': [0.3, 0.5, 0.7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'Linear Regression': {'fit_intercept': [True, False], 'positive': [True, False]}
}

# Buscar millors paràmetres
best_params = {}
for model in models.keys():
    search = RandomizedSearchCV(models[model], params[model], n_iter=10, cv=5, random_state=SEED)
    search.fit(X_train, y_train)
    models[model] = search.best_estimator_
    best_params[model] = search.best_params_

# Avaluar mètriques
metrics = {}
predictions = {}
for model in models.keys():
    y_pred = models[model].predict(X_test)
    predictions[model] = y_pred
    metrics[model] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }

# Mostrar mètriques
metrics_df = pd.DataFrame(metrics).T.sort_values('MAE')
print(metrics_df)

# Gràfic MAE
plt.figure(figsize=(10, 6))
metrics_df['MAE'].plot(kind='barh', color='skyblue', edgecolor='black')
plt.title("MAE per Model")
plt.xlabel("MAE")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# Prediccions amb Random Forest
y_pred_rf = models['Random Forest'].predict(X_test)

# Importància de les característiques en Random Forest
importances = models['Random Forest'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Filtrar características para eliminar 'G1' y 'G2'
filtered_feature_importance_df = feature_importance_df[~feature_importance_df['Feature'].isin(['G1', 'G2'])]

# Mostrar importància de les característiques sense 'G1' i 'G2'
plt.figure(figsize=(10, 6))
plt.bar(filtered_feature_importance_df['Feature'], filtered_feature_importance_df['Importance'], color='skyblue')
plt.title("Importància de les Característiques (Random Forest)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Visualización de predicciones vs valores reales
plt.scatter(y_test, y_pred, alpha=0.80)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Línea ideal
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

# Validación cruzada
cv = KFold(n_splits=10, shuffle=True, random_state=SEED)  # 10-fold cross-validation

# Evaluar modelos con validación cruzada
metrics_cv = {}
predictions_cv = {}

for model_name, model in models.items():
    # Obtener predicciones usando validación cruzada
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    predictions_cv[model_name] = y_pred_cv

    # Calcular métricas
    metrics_cv[model_name] = {
        'MAE': mean_absolute_error(y, y_pred_cv),
        'MSE': mean_squared_error(y, y_pred_cv),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_cv)),
        'R²': r2_score(y, y_pred_cv)
    }

# Convertir las métricas de validación cruzada en un DataFrame para visualización
metrics_cv_df = pd.DataFrame(metrics_cv).T.sort_values('MAE')
print("Métricas con Validación Cruzada:")
print(metrics_cv_df)

# Gráfico MAE con validación cruzada
plt.figure(figsize=(10, 6))
metrics_cv_df['MAE'].plot(kind='barh', color='lightgreen', edgecolor='black')
plt.title("MAE por Modelo (Validación Cruzada)")
plt.xlabel("MAE")
plt.ylabel("Modelo")
plt.tight_layout()
plt.show()

# Gráfico de valores reales vs predicciones con validación cruzada
model_name = 'Random Forest'  # Cambiar por el modelo que desees analizar
y_pred_cv_rf = predictions_cv[model_name]
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_cv_rf, alpha=0.7, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title(f"Valores Reales vs Predicciones ({model_name} - Validación Cruzada)")
plt.tight_layout()
plt.show()
