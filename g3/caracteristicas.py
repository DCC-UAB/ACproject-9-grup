import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Cargar el dataset
data = pd.read_csv('student-mat.csv')

# Selección de las columnas relevantes
cols = ['age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'studytime',
        'schoolsup', 'famsup', 'paid', 'activities', 'internet', 'romantic',
        'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'traveltime', 'G3']
data = data[cols]

# Mapear las variables categóricas
mapping = {'address': {'U': 0, 'R': 1},
           'famsize': {'LE3': 0, 'GT3': 1},
           'Pstatus': {'T': 0, 'A': 1},
           'schoolsup': {'no': 0, 'yes': 1},
           'famsup': {'no': 0, 'yes': 1},
           'paid': {'no': 0, 'yes': 1},
           'activities': {'no': 0, 'yes': 1},
           'internet': {'no': 0, 'yes': 1},
           'romantic': {'no': 0, 'yes': 1}}
for column in mapping:
    data[column] = data[column].map(mapping[column])

# Separar las características (X) y la variable objetivo (y)
x = data.drop('G3', axis=1)
y = data['G3']

# Entrenar el modelo de Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x, y)

# Importancia de las características
feature_importances = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)
print("Importancia de características:")
print(feature_importances)

# Seleccionar características más importantes (umbral: media de importancia)
selector = SelectFromModel(rf, threshold="mean", prefit=True)
important_features = x.columns[selector.get_support()]
print(f"\nCaracterísticas seleccionadas ({len(important_features)}): {list(important_features)}")

average_importance = feature_importances.mean()
print(f"Promedio de importancia: {average_importance:.4f}")
important_features = feature_importances[feature_importances > average_importance]
print(f"Características seleccionadas ({len(important_features)}):\n{important_features}")
