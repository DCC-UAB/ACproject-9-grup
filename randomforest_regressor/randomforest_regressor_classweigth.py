import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Carregar dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")

# Assegurar que y és una sèrie de pandas
y = pd.Series(y.values.flatten(), name="target")  # Assegurar que és una sola columna i és una sèrie

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

# Pesos de les classes (proporcionats)
class_weights = {2: 0.943, 1: 0.549, 4: 1.505, 3: 0.929, 5: 2.633}
print("Pesos de les classes utilitzats:", class_weights)

# Convertir Y_train en una sèrie (si no ho és ja) i calcular els pesos de les mostres
Y_train = pd.Series(Y_train)
sample_weights = Y_train.map(class_weights).values

# Crear el model Random Forest Regressor amb els hiperparàmetres proporcionats
best_params = {
    'bootstrap': True,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 50
}
best_model = RandomForestRegressor(random_state=42, **best_params)

# Entrenar el model amb el conjunt de train i els pesos de les mostres
best_model.fit(X_train, Y_train, sample_weight=sample_weights)

# Prediccions amb el model entrenat
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

# Classificació
# Convertir les prediccions contínues a classes utilitzant assign_class
y_train_pred_class = [assign_class(pred) for pred in y_train_pred]
y_test_pred_class = [assign_class(pred) for pred in y_test_pred]

# Accuracy
accuracy_train = accuracy_score(Y_train, y_train_pred_class)
accuracy_test = accuracy_score(Y_test, y_test_pred_class)
print(f"\nAccuracy entrenament: {accuracy_train:.4f}")
print(f"Accuracy test: {accuracy_test:.4f}")

# Report de classificació
print("\nClassification Report:")
print(classification_report(Y_test, y_test_pred_class))

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

