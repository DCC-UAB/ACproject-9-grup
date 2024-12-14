import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Carregar dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Comprovar distribució inicial de classes
print("Distribució inicial de les classes:", Counter(Y_train))

# Balancejar les dades utilitzant SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)  # Ajustar k_neighbors segons la distribució
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Comprovar distribució després de balancejar
print("Distribució després de SMOTE:", Counter(Y_train_res))

# Crear el model Random Forest amb els millors hiperparàmetres trobats
best_params = {
    'bootstrap': True,
    'class_weight': 'balanced',
    'max_depth': 30,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 10,
    'n_estimators': 200
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Classe Predicha')
plt.ylabel('Classe Real')
plt.title('Matriu de Confusió')
plt.show()

