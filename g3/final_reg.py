import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Importar datos
df = pd.read_csv('student-mat.csv')

# Diccionario para mapear valores categóricos binarios
bin_map = {
    'yes': 1, 'no': 0,
    'GP': 1, 'MS': 0,
    'F': 1, 'M': 0,
    'U': 1, 'R': 0,
    'LE3': 0, 'GT3': 1,
    'T': 1, 'A': 0
}

# Lista de columnas categóricas binarias
cat_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
            'higher', 'internet', 'romantic']

# Aplicar mapeo binario solo a columnas con valores válidos
for col in cat_cols:
    if col in df.columns:
        unique_vals = df[col].unique()
        if all(val in bin_map for val in unique_vals):
            df[col] = df[col].map(bin_map)
        else:
            print(f"Columna {col} contiene valores no mapeados: {unique_vals}")

# Codificar columnas categóricas con más de dos valores
multi_cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
label_encoders = {}
for col in multi_cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Limpieza de columnas categóricas restantes
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace("'", "").str.strip()

# Asegurarse de que todas las columnas son numéricas o categorizadas correctamente
numeric_df = df.select_dtypes(include=['number'])

# División en conjunto de entrenamiento y prueba
test_size = 0.2
train_df, test_df = train_test_split(numeric_df, test_size=test_size, random_state=42)

# División del dataset para el análisis
results = {}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for features, name in [(['G1', 'G2'], 'con_G1_y_G2'), (['G1'], 'solo_G1'), ([], 'sin_G1_y_G2')]:
    # Verificar que las columnas existen antes de eliminarlas
    features_to_drop = [f for f in features if f in train_df.columns]

    # Conjunto de entrenamiento
    X_train = train_df.drop(['G3'] + features_to_drop, axis=1)
    y_train = train_df['G3']

    # Conjunto de prueba
    X_test = test_df.drop(['G3'] + features_to_drop, axis=1)
    y_test = test_df['G3']

    # Modelo con validación cruzada
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    y_train_pred = cross_val_predict(rf_model, X_train, y_train, cv=kfold)

    # Entrenar en el conjunto de entrenamiento completo y evaluar en el conjunto de prueba
    rf_model.fit(X_train, y_train)
    y_test_pred = rf_model.predict(X_test)

    # Evaluación
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    results[name] = {'MAE_Train': mae_train, 'MAE_Test': mae_test, 'Predicciones': y_test_pred, 'Reales': y_test}

    # Gráfico de predicciones vs reales
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'Predicciones vs Reales ({name})')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Exclusión de G3 = 0
numeric_df_no_zero = numeric_df[numeric_df['G3'] != 0]
train_df_no_zero, test_df_no_zero = train_test_split(numeric_df_no_zero, test_size=test_size, random_state=42)

# Verificar que las columnas existen antes de eliminarlas
features_to_drop_no_zero = [f for f in ['G1', 'G2'] if f in train_df_no_zero.columns]

X_train_no_zero = train_df_no_zero.drop(['G3'] + features_to_drop_no_zero, axis=1)
y_train_no_zero = train_df_no_zero['G3']
X_test_no_zero = test_df_no_zero.drop(['G3'] + features_to_drop_no_zero, axis=1)
y_test_no_zero = test_df_no_zero['G3']

# Modelo con validación cruzada (sin G3 = 0)
rf_model_no_zero = RandomForestRegressor(n_estimators=100, random_state=42)
y_train_no_zero_pred = cross_val_predict(rf_model_no_zero, X_train_no_zero, y_train_no_zero, cv=kfold)
rf_model_no_zero.fit(X_train_no_zero, y_train_no_zero)
y_test_no_zero_pred = rf_model_no_zero.predict(X_test_no_zero)

# Evaluación
mae_train_no_zero = mean_absolute_error(y_train_no_zero, y_train_no_zero_pred)
mae_test_no_zero = mean_absolute_error(y_test_no_zero, y_test_no_zero_pred)

# Gráfico de predicciones vs reales (sin G3 = 0)
plt.figure(figsize=(6, 6))
plt.scatter(y_test_no_zero, y_test_no_zero_pred, alpha=0.6, color='g')
plt.plot([y_test_no_zero.min(), y_test_no_zero.max()], [y_test_no_zero.min(), y_test_no_zero.max()], 'k--', lw=2)
plt.title('Predicciones vs Reales (sin G3 = 0)')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid(True)
plt.tight_layout()
plt.show()

# Resultados finales
results['sin_G3_0'] = {'MAE_Train': mae_train_no_zero, 'MAE_Test': mae_test_no_zero, 'Predicciones': y_test_no_zero_pred, 'Reales': y_test_no_zero}

# Mostrar MAE para cada escenario
for key, value in results.items():
    print(f"{key}: MAE_Train = {value['MAE_Train']:.2f}, MAE_Test = {value['MAE_Test']:.2f}")
