""""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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

# Aplicar mapeig
for column in list(mapping.keys()):
    df[column] = df[column].map(mapping[column])

# Separar les característiques (X) i l'etiqueta (Y)
X = df.drop(columns=['Walc'])
Y = df['Walc']

# Reindexar les classes perquè comencin des de 0
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)  # Això farà que les classes siguin 0, 1, 2, 3, 4 en lloc de 1, 2, 3, 4, 5

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Definir el rang dels paràmetres per provar
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.1, 0.2],
    'alpha': [0, 0.1, 1, 10],  # L1 regularization (més alt = més regularització)
    'lambda': [0, 0.1, 1, 10]  # L2 regularization (més alt = més regularització)
}

# Crear el model XGBoost
model = XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)

# Utilitzar GridSearchCV per trobar els millors paràmetres
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, Y_train)

# Mostrar els millors paràmetres
print("Millors paràmetres trobats: ", grid_search.best_params_)

# Entrenar el model amb els millors paràmetres
best_model = grid_search.best_estimator_
best_model.fit(X_train, Y_train)

# Fer prediccions sobre el conjunt de test
y_pred = best_model.predict(X_test)

# Report de classificació per veure les mètriques
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Matrícula de Confusió
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
plt.xlabel('Classe Predicha')
plt.ylabel('Classe Real')
plt.title('Matrícula de Confusió')
plt.show()

# Accuracy en el conjunt de test
test_accuracy = np.mean(y_pred == Y_test)
print(f"Accuracy en el conjunt de test: {test_accuracy:.4f}")

"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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

# Separar les característiques (X) i l'etiqueta (Y)
X = df.drop(columns=['Walc'])
Y = df['Walc']

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Calcular els pesos de les classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = dict(zip(np.unique(Y_train), class_weights))

# Crear el model Random Forest amb class weights
model = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)

# Validació creuada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Realitzar la validació creuada
fold_accuracies = cross_val_score(model, X_train, Y_train, cv=cv, scoring='accuracy')

# Mostrar l'accuracy per a cada "fold"
for i, accuracy in enumerate(fold_accuracies, 1):
    print(f"Accuracy del fold {i}: {accuracy:.4f}")

# Mostrar l'accuracy mitjà en Cross-Validation
print(f"\nAccuracy mitjà en Cross-Validation: {np.mean(fold_accuracies):.4f}")

# Entrenar el model amb el conjunt d'entrenament complet
model.fit(X_train, Y_train)

# Fer les prediccions sobre el conjunt de test
y_pred = model.predict(X_test)

# Mostrar l'accuracy en el conjunt de test
test_accuracy = accuracy_score(Y_test, y_pred)
print(f"\nAccuracy en el conjunt de test: {test_accuracy:.4f}")

# Report de classificació
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Matriu de confusió
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Classe Predicha')
plt.ylabel('Classe Real')
plt.title('Matriu de Confusió')
plt.show()


