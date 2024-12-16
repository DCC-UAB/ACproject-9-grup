import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from roc_curve import plot_roc_curve

# Cargar datos preprocessados
X = pd.read_csv("Xbinari_preprocessed.csv")
y = pd.read_csv("ybinari_preprocessed.csv")

# Dividir en conjunto de train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balancear las clases utilizando SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Crear el modelo de regresión logística
model = LogisticRegression(random_state=42)

# Realizar cross-validation
cv_scores = cross_val_score(model, X_train_res, Y_train_res, cv=5, scoring='accuracy')
print(f"\nAccuracy promedio en Cross-Validation: {np.mean(cv_scores):.4f}")

# Entrenar el modelo
model.fit(X_train_res, Y_train_res)

# Realizar predicciones en el conjunto de test
y_pred = model.predict(X_test)

# Evaluar el modelo
test_accuracy = accuracy_score(Y_test, y_pred)
print(f"\nAccuracy en el conjunto de test: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Matriz de confusión
cm_test = confusion_matrix(Y_test, y_pred)
cm_percentage_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100

labels = np.array([[f"{int(val)}\n({pct:.1f}%)" 
                    for val, pct in zip(row, pct_row)] 
                   for row, pct_row in zip(cm_test, cm_percentage_test)])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=labels, fmt='', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.title('Matriz de Confusión - Logistic Regression')
plt.show()

roc_curve = plot_roc_curve(model, X_test, Y_test, "Regressió Logística")