import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GB, AdaBoostRegressor as AB
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Semilla para reproducibilidad
SEED = 1

# Cargar los datos
data = pd.read_csv('student-mat.csv')

# Renombrar las variables a catalán
data.rename(columns={
    'absences': 'absències',
    'failures': 'fracassos',
    'goout': 'sortides',
    'freetime': 'temps_lliure',
    'age': 'edat',
    'health': 'salut',
    'G3': 'nota_final'
}, inplace=True)

# Selección de características más importantes
selected_features = [
    "absències",
    "fracassos",
    "sortides",
    "temps_lliure",
    "edat",
    "salut",
    "nota_final"
]

# Reducir el dataset a las características seleccionadas
data = data[selected_features]

# Definir las variables predictoras y la variable objetivo
x = data.drop('nota_final', axis=1)
y = data['nota_final']

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=SEED)

# Escalar los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Definir los modelos
models = {
    'KNN': KNN(),
    'DT': DT(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'Random Forest': RF(random_state=SEED),
    'Gradient Boosting': GB(random_state=SEED),
    'AdaBoost': AB(random_state=SEED)
}

# Definir los hiperparámetros para cada modelo
parameters = {
    'KNN': {'n_neighbors': [i for i in range(3, 50)]},
    'DT': {'max_depth': [i for i in range(1, 25)]},
    'Lasso': {'alpha': [i for i in range(0, 50)], 'tol': [0.1, 0.01, 0.001]},
    'Ridge': {'alpha': [i for i in range(0, 50)], 'tol': [0.1, 0.01, 0.001]},
    'ElasticNet': {'alpha': [i for i in range(0, 50)], 'l1_ratio': [0.1, 0.5, 0.9]},
    'SVR': {'kernel': ['linear', 'poly', 'rbf'], 'C': [i for i in range(1, 101)], 'epsilon': [0.01, 0.1]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30], 'max_features': [0.3, 0.5, 0.7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Buscar los mejores parámetros
best_params = {}
cval = {}
for model in models.keys():
    param_space = np.prod([len(v) for v in parameters[model].values()])
    n_iter = min(10, param_space)  # Ajustar iteraciones dinámicamente
    cval[model] = RandomizedSearchCV(models[model], parameters[model], cv=5, n_iter=n_iter, scoring='neg_mean_absolute_error', random_state=SEED)
    cval[model].fit(x_train, y_train)
    models[model] = cval[model].best_estimator_
    best_params[model] = cval[model].best_params_

# Evaluar modelos
errors = {}
predictions = {}
for model in models.keys():
    y_pred = models[model].predict(x_test)
    predictions[model] = y_pred
    errors[model] = MAE(y_test, y_pred)

# Ordenar modelos por menor MAE
sorted_errors = sorted(errors.items(), key=lambda x: x[1])

# Gráfico de MAE para todos los modelos
plt.figure(figsize=(10, 6))
mae_series = pd.Series(errors).sort_values()
bars = mae_series.plot(kind='barh', color=sns.color_palette("husl", len(mae_series)), edgecolor='black')
plt.title("Mean Absolute Error (MAE) por Modelo")
plt.xlabel("MAE")
plt.ylabel("Modelos")
for bar in bars.containers:
    bars.bar_label(bar, fmt='%.2f')
plt.show()

# Visualización con subplots
fig, axes = plt.subplots(2, 1, figsize=(20, 18))

# Gráfico para el modelo SVR
axes[0].scatter(range(len(y_test)), y_test, label='Datos Reales', marker='x', color='red')
axes[0].scatter(range(len(y_test)), predictions['AdaBoost'], label='Predicciones (ADA)', marker='o', color='blue')
axes[0].set_title('Predicciones vs Valores Reales (ADA')
axes[0].legend()
axes[0].set_xlabel('Índice')
axes[0].set_ylabel('Valores')

plt.tight_layout()
plt.show()

