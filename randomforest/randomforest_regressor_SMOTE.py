import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Carregar dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")
y = y['Walc']

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalitzar les dades
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Comprovar distribució inicial de classes
print("Distribució inicial de les classes:", Counter(Y_train))

# Configurar SMOTE per balancejar les dades
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Comprovar distribució després de SMOTE
print("Distribució després de SMOTE:", Counter(Y_train_res))

best_params = {
    'bootstrap': True,
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 200
}

# Crear el model Random Forest Regressor
model = RandomForestRegressor(random_state=42, **best_params)

# Realitzar cross-validation per avaluar el model
cv_scores = cross_val_score(model, X_train_res, Y_train_res, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MAE: {-np.mean(cv_scores):.4f}")

# Entrenar el model amb el conjunt de train balancejat
model.fit(X_train_res, Y_train_res)

# Fer prediccions sobre el conjunt de test
y_pred = model.predict(X_test)

limits = [1.5, 2.5, 3.5, 4.5]  # Límits per a les categories
y_pred = np.digitize(y_pred, bins=limits, right=True)  # Classificar valors
y_pred = np.clip(y_pred, 1, 5)  # Assegurar que les categories estiguin entre 1 i 5

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
plt.title('Matriu de Confusió Test')
plt.show()



# Avaluar el model
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f"MAE en el conjunt de test: {mae:.4f}")
print(f"R2 en el conjunt de test: {r2:.4f}")

