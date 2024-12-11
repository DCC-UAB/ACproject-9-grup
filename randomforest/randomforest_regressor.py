import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer

# Llegir el fitxer CSV
df = pd.read_csv("student-mat.csv")

# Filtratge de columnes
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 
        'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 
        'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
df = df[cols]

# Mapeig de les columnes categòriques
mapping = {
    'address': {'U': 0, 'R': 1},
    'famsize': {'LE3': 0, 'GT3': 1},
    'Pstatus': {'T': 0, 'A': 1},
    'schoolsup': {'no': 0, 'yes': 1},
    'famsup': {'no': 0, 'yes': 1},
    'paid': {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'internet': {'no': 0, 'yes': 1},
    'romantic': {'no': 0, 'yes': 1},
    'school': {'GP': 0, 'MS': 1},
    'sex': {'M': 0, 'F': 1},
    'higher': {'no': 0, 'yes': 1},
    'nursery': {'no': 0, 'yes': 1}
}

# Aplicar el mapeig
for column in mapping:
    df[column] = df[column].map(mapping[column])
    
df['G1'] = df['G1'].str.replace("'", "").astype(float).astype(int)

# Separar les característiques (X) i l'etiqueta (Y)
X = df.drop(columns=['Walc'])  # 'Walc' és la variable contínua
Y = df['Walc']  # Etiqueta contínua

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Crear el model Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Entrenar el model amb el conjunt de train
regressor.fit(X_train, Y_train)

# Fer les prediccions sobre el conjunt de test
y_pred_regressor = regressor.predict(X_test)

# Convertir les prediccions contínues en classes
# Utilitzem KBinsDiscretizer per dividir les prediccions en classes
kbins = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
y_pred_binned = kbins.fit_transform(y_pred_regressor.reshape(-1, 1)).astype(int).flatten()

# Convertir la variable real de test també en classes
y_test_binned = kbins.transform(Y_test.values.reshape(-1, 1)).astype(int).flatten()

# Crear el model RandomForestClassifier per a la classificació de les prediccions
classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# Entrenar el model amb el conjunt de train
classifier.fit(X_train, y_test_binned)

# Fer les prediccions sobre el conjunt de test
y_pred_classifier = classifier.predict(X_test)

# Avaluar el model de classificació
accuracy = accuracy_score(y_test_binned, y_pred_classifier)
print(f"Accuracy en el conjunt de test: {accuracy:.4f}")

# Report de classificació
print("\nClassification Report:")
print(classification_report(y_test_binned, y_pred_classifier))

