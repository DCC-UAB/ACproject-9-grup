import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GB, AdaBoostRegressor as AB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer

# Semilla
SEED = 1

# Función para validar los datos
def validar_datos(df):
    print("\n=== Valores Nulos por Columna ===")
    print(df.isnull().sum())
    
    print("\n=== Categorías Únicas de Columnas Categóricas ===")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"{col}: {df[col].unique()}")
    
    print("\n=== Estadísticas de Datos Numéricos ===")
    print(df.describe())

# Función para cargar y preprocesar datos
def cargar_y_preprocesar_datos(filepath):
    df = pd.read_csv(filepath)
    validar_datos(df)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace("'", "").str.strip()

    df.rename(columns={
        'absences': 'absències', 'failures': 'fracassos', 'goout': 'sortides',
        'freetime': 'temps', 'age': 'edat', 'health': 'salut', 'G3': 'nota'
    }, inplace=True)

    bin_map = {'yes': 1, 'no': 0, 'GP': 1, 'MS': 0, 'F': 1, 'M': 0, 'U': 1, 'R': 0, 'LE3': 0, 'GT3': 1, 'T': 1, 'A': 0}
    cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

    for col in cat_cols:
        if df[col].dtype == 'object':
            if set(df[col].unique()).issubset(bin_map.keys()):
                df[col] = df[col].map(bin_map)
            else:
                df[col] = LabelEncoder().fit_transform(df[col])

    outlier_cols = ['absències', 'fracassos', 'studytime', 'famrel', 'Dalc', 'Walc']
    log_transform_vars = ['absències', 'Dalc', 'Walc']
    for col in log_transform_vars:
        df[col] = np.log1p(df[col])
    clip_transform_vars = [col for col in outlier_cols if col not in log_transform_vars]
    for col in clip_transform_vars:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    df[numeric_cols] = power_transformer.fit_transform(df[numeric_cols])

    X = df.drop('nota', axis=1)
    y = df['nota']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return df, X, X_train, X_test, y_train, y_test, y

df, X, X_train, X_test, y_train, y_test, y = cargar_y_preprocesar_datos('student-mat.csv')

# Definir modelos
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

params = {
    'KNN': {'n_neighbors': range(3, 50)},
    'DT': {'max_depth': range(1, 25)},
    'ElasticNet': {'alpha': range(1, 50), 'l1_ratio': [0.1, 0.5, 0.9]},
    'SVR': {'kernel': ['linear', 'poly', 'rbf'], 'C': range(1, 101), 'epsilon': [0.01, 0.1]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30], 'max_features': [0.3, 0.5, 0.7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'Linear Regression': {'fit_intercept': [True, False], 'positive': [True, False]}
}

# Buscar mejores parámetros
best_params = {}
for model_name in models:
    search = RandomizedSearchCV(models[model_name], params[model_name], n_iter=5, cv=3, random_state=SEED)
    search.fit(X_train, y_train)
    models[model_name] = search.best_estimator_
    best_params[model_name] = search.best_params_

# Métricas
metrics = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    metrics[model_name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }

# Resultados
metrics_df = pd.DataFrame(metrics).T.sort_values(by='MAE')
print(metrics_df)

# Gráficas
plt.figure(figsize=(10, 6))
metrics_df['MAE'].plot(kind='barh', color='skyblue', edgecolor='black')
plt.title("MAE por Modelo")
plt.xlabel("MAE")
plt.ylabel("Modelo")
plt.tight_layout()
plt.show()

# Validación cruzada general
def validacion_cruzada_general(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)
    return mean_r2, std_r2

# Validación cruzada para todos los modelos
for model_name, model in models.items():
    mean_r2, std_r2 = validacion_cruzada_general(model, X, y)
    print(f"{model_name} - R² Promedio: {mean_r2:.4f}, Desviación Estándar: {std_r2:.4f}")

# Asegúrate de que X_test y y_test estén definidos y que Random Forest esté entrenado
# Si es necesario, ajusta nuevamente el modelo aquí:
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(random_state=SEED)
model_rf.fit(X_train, y_train)

# Predicciones del modelo Random Forest
y_pred_rf = model_rf.predict(X_test)

# Crear gráfico de valores predichos vs valores reales
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.7, edgecolors='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.title("Valores Predichos vs Valores Reales (Random Forest)")
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.xlim(min(y_test), max(y_test))
plt.ylim(min(y_test), max(y_test))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
