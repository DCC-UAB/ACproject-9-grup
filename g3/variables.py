import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Cargar el dataset completo
data = pd.read_csv('student-mat.csv')

plt.figure(figsize=(10, 5), dpi=400)
sns.countplot(x = data['G3'], palette='Blues')
plt.title('Distribució notas Math(G3)')
plt.show()

# Eliminar las columnas G1 y G2
data = data.drop(columns=['G1', 'G2'])

D= {'sex':{'M':0, 'F':1},
    'address': {'Urban':0, 'Rural':1},
    'famsize': {'Less then 3':0, 'Greater then 3':1},
    'Pstatus': {'Together':0, 'Apart':1},
           'schoolsup':{'no':0,'yes':1},
           'famsup':{'no':0,'yes':1},
           'paid':{'no':0,'yes':1},
           'activities':{'no':0,'yes':1},
           'internet':{'no':0,'yes':1},
           'romantic':{'no':0,'yes':1},
            'higher':{'no':0,'yes':1},
             'nursery':{'no':0,'yes':1}}

for column in list(D.keys()):
    data[column] = data[column].map(D[column])

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
