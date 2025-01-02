import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from roc_curve import plot_roc_curve

# Carregar dades preprocessades
X = pd.read_csv("Xbinari_preprocessed.csv")
y = pd.read_csv("ybinari_preprocessed.csv")

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Comprovar distribució inicial de les classes
print("Distribució inicial de les classes:", Counter(Y_train))

# Balancejar les dades utilitzant SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)  # Ajustar k_neighbors segons la distribució
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Comprovar distribució després de balancejar
print("Distribució després de SMOTE:", Counter(Y_train_res))

# Crear el model SVM amb els millors hiperparàmetres trobats
# Usarem un SVM amb un kernel radial basis function (RBF) i ajustarem els hiperparàmetres
best_params = {
    'C': 1.0,            # Penalització
    'kernel': 'rbf',     # Kernel radial basis function
    'gamma': 'scale',    # Gamma per al kernel
    'class_weight': 'balanced'  # Pes de les classes per gestionar el desbalanceig
}

model = SVC(random_state=42, **best_params)

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


# Calcular l'accuracy en el conjunt de test
test_accuracy = accuracy_score(Y_test, y_pred)
print(f"\nAccuracy en el conjunt de test: {test_accuracy:.4f}")

# Report de classificació
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Matriu de confusió
cm_test = confusion_matrix(Y_test, y_pred)
# Percentatges
cm_percentage_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100

# Crear el text combinat (freqüència + percentatge)
labels = np.array([[f"{int(val)}\n({pct:.1f}%)" 
                    for val, pct in zip(row, pct_row)] 
                   for row, pct_row in zip(cm_test, cm_percentage_test)])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=labels, fmt='', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.title('Matriu de Confusió - SVM')
plt.show()

roc_curve = plot_roc_curve(model, X_test, Y_test, "SVM")