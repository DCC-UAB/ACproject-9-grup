import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar el dataset
data = pd.read_csv('student-mat.csv')

# Visualitzar la distribució de les notes (G3)
plt.figure(figsize=(10, 5), dpi=400)
sns.countplot(x=data['G3'], palette='Blues')
plt.title('Distribució de les Notes (G3)', fontsize=14)
plt.xlabel('Nota', fontsize=12)
plt.ylabel('Nombre d\'Estudiants', fontsize=12)
plt.show()

# Eliminar les columnes G1 i G2
data = data.drop(columns=['G1', 'G2'])

# Mapejar variables categòriques
D = {'sex': {'M': 0, 'F': 1},
     'address': {'U': 0, 'R': 1},
     'famsize': {'LE3': 0, 'GT3': 1},
     'Pstatus': {'T': 0, 'A': 1},
     'schoolsup': {'no': 0, 'yes': 1},
     'famsup': {'no': 0, 'yes': 1},
     'paid': {'no': 0, 'yes': 1},
     'activities': {'no': 0, 'yes': 1},
     'internet': {'no': 0, 'yes': 1},
     'romantic': {'no': 0, 'yes': 1},
     'higher': {'no': 0, 'yes': 1},
     'nursery': {'no': 0, 'yes': 1}}

for column in D.keys():
    data[column] = data[column].map(D[column])

# Separar variables explicatives (X) i l'objectiu (y)
x = data.drop('G3', axis=1)
y = data['G3']

# Entrenar el model Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x, y)

# Importància de les característiques
feature_importances = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)
print("\nImportància de les característiques:")
print(feature_importances)

# Seleccionar característiques amb importància superior a la mitjana
average_importance = feature_importances.mean()
important_features_above_mean = feature_importances[feature_importances > average_importance]
print(f"\nCaracterístiques seleccionades per sobre de la mitjana ({len(important_features_above_mean)}):")
print(important_features_above_mean)

# Gràfic d'importància de les característiques seleccionades
plt.figure(figsize=(10, 6), dpi=150)
important_features_above_mean.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Importància de les Característiques Seleccionades (Per Sobre de la Mitjana)', fontsize=14)
plt.xlabel('Importància', fontsize=12)
plt.ylabel('Característiques', fontsize=12)
plt.tight_layout()
plt.show()


