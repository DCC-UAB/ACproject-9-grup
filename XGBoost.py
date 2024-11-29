"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Llegir el fitxer CSV
df = pd.read_csv("student-mat.csv")

# Eliminar la columna "Dalc"
df.drop("Dalc", axis=1, inplace=True)

# Dividir el DataFrame en train i test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separar les característiques (X) i l'etiqueta (Y)
X_train = train_df.drop(columns=['Walc'])
Y_train = train_df['Walc'] - 1  # Ajustar les classes de 1-5 a 0-4
X_test = test_df.drop(columns=['Walc'])
Y_test = test_df['Walc'] - 1  # Ajustar les classes de 1-5 a 0-4

# Identificar les columnes categòriques
cat_cols = X_train.select_dtypes(include=['object']).columns

# Aplicar One-Hot Encoding a les columnes categòriques
X_train = pd.get_dummies(X_train, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)

# Alinear les columnes de train i test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Entrenar el model de XGBoost
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, Y_train)

# Fer prediccions sobre el conjunt de test
y_pred = model.predict(X_test)

# Avaluar el model
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Report de classificació
print("\nClassification Report:")
print(classification_report(Y_test, y_pred, target_names=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4', 'Classe 5']))

# Matriu de confusió
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Classe Predicha')
plt.ylabel('Classe Real')
plt.title('Matrícula de Confusió')
plt.show()

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt

# Llegir el fitxer CSV
df = pd.read_csv("student-mat.csv")

# Filtratge de columnes
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 
        'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

df = df[cols]

# Mapeig de les columnes categòriques
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
    'school': {'GP': 0, 'MS': 1},  # Mapeig per la columna 'school'
    'sex': {'M': 0, 'F': 1},  # Mapeig per la columna 'sex'
    'higher': {'no': 0, 'yes': 1},  # Afegit el mapeig per la columna 'higher'
    'nursery': {'no': 0, 'yes': 1}  # Afegit el mapeig per la columna 'nursery'
}

# Afegeix més mapeig per a totes les columnes categòriques que tinguin valors 'yes' / 'no' o altres categories
# Afegeix aquí les altres columnes que tinguin 'yes'/'no' o altres valors categòrics a mapejar.
for column in list(mapping.keys()):
    df[column] = df[column].map(mapping[column])

# Comprovar si hi ha alguna columna que no s'ha mapejat correctament
print(df.dtypes)


# Separar les característiques (X) i l'etiqueta (Y)
X = df.drop(columns=['Walc'])
Y = df['Walc']

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Calcular els pesos de les classes de manera automàtica
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = dict(zip(np.unique(Y_train), class_weights))

# Crear el model Random Forest amb class weights
model = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)

# Validació creuada estratificada (Stratified K-Fold) per tenir en compte el desbalanceig
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Realitzar la validació creuada
cross_val_results = cross_val_score(model, X_train, Y_train, cv=cv, scoring='accuracy')

# Mostrar l'accuracy mitjà en Cross-Validation
print(f"Accuracy mitjà del model en Cross-Validation: {np.mean(cross_val_results):.4f}")

# Entrenar el model amb el conjunt de dades complet d'entrenament
model.fit(X_train, Y_train)

# Fer les prediccions sobre el conjunt de test
y_pred = model.predict(X_test)

# Mostrar l'accuracy en el conjunt de test
test_accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy en el conjunt de test: {test_accuracy:.4f}")

# Report de classificació per veure les mètriques
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Matrícula de Confusió
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Classe Predicha')
plt.ylabel('Classe Real')
plt.title('Matrícula de Confusió')
plt.show()
