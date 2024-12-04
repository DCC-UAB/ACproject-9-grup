import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Llegir el fitxer CSV
df = pd.read_csv("student-mat.csv")

# Filtratge de les columnes necessàries
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 
        'famrel', 'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
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

# Modificar les classes per començar en 0 (en comptes de 1-5, seran 0-4)
Y = Y - 1

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Aplicar SMOTE per a classes desbalancejades
smote = SMOTE(sampling_strategy='not minority', random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Definir l'escala del model
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Definir el model XGBoost
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Definir els paràmetres per a la cerca en graella
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# Realitzar la cerca en graella
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_res, Y_train_res)

# Mostrar els millors paràmetres trobats
print("Millors paràmetres trobats:", grid_search.best_params_)

# Entrenar el model amb els millors paràmetres
best_model = grid_search.best_estimator_

# Validació creuada
fold_accuracies = cross_val_score(best_model, X_train_res, Y_train_res, cv=5, scoring='accuracy')

# Mostrar l'accuracy per a cada fold
for i, accuracy in enumerate(fold_accuracies, 1):
    print(f"Accuracy del fold {i}: {accuracy:.4f}")
print(f"\nAccuracy mitjà en Cross-Validation: {np.mean(fold_accuracies):.4f}")

# Fer les prediccions sobre el conjunt de test
y_pred = best_model.predict(X_test)

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



""""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
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

# Definir el model
model = RandomForestClassifier(random_state=42)

# Definir la graella d'hiperparàmetres per a GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt'],
    'class_weight': ['balanced', None]
}

# Crear el GridSearchCV per optimitzar els hiperparàmetres
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Realitzar el GridSearchCV per trobar els millors paràmetres
grid_search.fit(X_train, Y_train)

# Mostrar els millors paràmetres trobats
print(f"Millors paràmetres trobats: {grid_search.best_params_}")

# Obtenir el millor model amb els paràmetres trobats
best_model = grid_search.best_estimator_

# Realitzar cross-validation amb el millor model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = cross_val_score(best_model, X_train, Y_train, cv=cv, scoring='accuracy')

# Mostrar els resultats de la cross-validation
for i, accuracy in enumerate(fold_accuracies, 1):
    print(f"Accuracy del fold {i}: {accuracy:.4f}")

# Mostrar l'accuracy mitjà en Cross-Validation
print(f"\nAccuracy mitjà en Cross-Validation: {np.mean(fold_accuracies):.4f}")

# Entrenar el model amb el conjunt d'entrenament complet
best_model.fit(X_train, Y_train)

# Fer les prediccions sobre el conjunt de test
y_pred = best_model.predict(X_test)

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

"""

""""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

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

# Separar les característiques (X) i l'etiqueta (Y)
X = df.drop(columns=['Walc'])
Y = df['Walc']

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Aplicar SMOTE per a l'oversampling de les classes minoritàries
smote = SMOTE(sampling_strategy='all', random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Crear el model Random Forest amb els millors hiperparàmetres trobats
best_params = {
    'bootstrap': False,
    'class_weight': 'balanced',
    'max_depth': 20,
    'max_features': 'log2',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 100
}

model = RandomForestClassifier(random_state=42, **best_params)

# Realitzar cross-validation utilitzant cross_val_score
cv_scores = cross_val_score(model, X_train_res, Y_train_res, cv=5, scoring='accuracy')

# Mostrar l'accuracy per a cada fold
for i, score in enumerate(cv_scores, 1):
    print(f"Accuracy del fold {i}: {score:.4f}")

# Mostrar l'accuracy mitjà de la validació creuada
print(f"\nAccuracy mitjà en Cross-Validation: {np.mean(cv_scores):.4f}")

# Entrenar el model amb el conjunt de train complet
model.fit(X_train_res, Y_train_res)

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

"""
