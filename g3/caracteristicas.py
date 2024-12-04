import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Cargar el dataset completo
data = pd.read_csv('student-mat.csv')

# Eliminar las columnas G1 y G2
data = data.drop(columns=['G1', 'G2'])

# Mapear las variables categóricas (si aplica)
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data[column] = data[column].astype('category').cat.codes

# Separar las características (X) y la variable objetivo (y)
x = data.drop('G3', axis=1)  # G3 sigue siendo la variable objetivo
y = data['G3']

# Entrenar el modelo de Random Forest para calcular importancias
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x, y)

# Importancia de las características
feature_importances = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)
print("\nImportancia de características:")
print(feature_importances)

# Seleccionar características con importancia mayor a la media
average_importance = feature_importances.mean()
important_features_above_mean = feature_importances[feature_importances > average_importance]
print(f"\nCaracterísticas seleccionadas por encima de la media ({len(important_features_above_mean)}):")
print(important_features_above_mean)
