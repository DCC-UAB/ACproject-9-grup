import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Carregar dades preprocessades
X = pd.read_csv("X123_preprocessed.csv")
y = pd.read_csv("y123_preprocessed.csv")

def assign_class(y_pred):
    if y_pred <= 1.5:
        return 1
    elif y_pred <= 2.5:
        return 2
    elif y_pred <= 3.5:
        return 3
    elif y_pred <= 4.5:
        return 4
    else:
        return 5

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalitzar les dades
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Comprovar distribució inicial de classes
print("Distribució inicial de les classes:", Counter(Y_train))

# Definir el model inicial de Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Definir el conjunt de paràmetres per a GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Realitzar la cerca de hiperparàmetres
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)

# Entrenar el model amb GridSearchCV
grid_search.fit(X_train, Y_train)

# Resultats de la cerca en graella
print(f"\nMillors paràmetres trobats: {grid_search.best_params_}")

# Prediccions amb el millor model
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)


# Calcular MSE per entrenament i test
mse_train = mean_squared_error(Y_train, y_train_pred)
mse_test = mean_squared_error(Y_test, y_test_pred)
mae_train = mean_absolute_error(Y_train, y_train_pred)
mae_test = mean_absolute_error(Y_test, y_test_pred)

# Mètriques
print(f"\nMSE entrenament: {mse_train:.4f}, MSE test: {mse_test:.4f}")
print(f"\nMAE entrenament: {mae_train:.4f}, MAE test: {mae_test:.4f}")

#Classificació
# Convertir les prediccions contínues a classes utilitzant assign_class
y_train_pred_class = [assign_class(pred) for pred in y_train_pred]
y_test_pred_class = [assign_class(pred) for pred in y_test_pred]

# Accuracy
accuracy_train = accuracy_score(Y_train, y_train_pred_class)
accuracy_test = accuracy_score(Y_test, y_test_pred_class)
print(f"\nAccuracy entrenament: {accuracy_train:.4f}")
print(f"Accuracy test: {accuracy_test:.4f}")

# Matriu de confusió
cm_test = confusion_matrix(Y_test, y_test_pred_class)
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


