import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Carregar dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")

# Funció per classificar els diferents valors predits pel regressor
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

# Dividir en conjunt d'entrenament i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanceig amb SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Distribució de classes després de SMOTE:")
print(pd.Series(y_train_balanced.to_numpy().ravel()).value_counts())

# Crear el pipeline per a la cerca de hiperparàmetres
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Per millorar l'estabilitat de l'algoritme
    ('poly', PolynomialFeatures()),  # Característiques polinòmiques
    ('model', Ridge())  # Regressió Ridge
])

# Definir els valors per fer la cerca en graella (Grid Search)
param_grid = {
    'poly__degree': [2, 3, 4, 5],  # Graus del polinomi
    'model__alpha': [0.1, 1.0, 10.0, 100.0]  # Valors d'alpha
}

# Crear el GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

# Entrenar el model amb GridSearchCV
grid_search.fit(X_train_balanced, y_train_balanced)

# Resultats de la cerca en graella
print(f"\nMillors paràmetres trobats: {grid_search.best_params_}")

# Prediccions amb el millor model
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train_balanced)
y_test_pred = best_model.predict(X_test)

# Convertir les prediccions contínues a classes utilitzant assign_class
y_train_pred_class = [assign_class(pred) for pred in y_train_pred]
y_test_pred_class = [assign_class(pred) for pred in y_test_pred]

# Calcular MSE per entrenament i test
mse_train = mean_squared_error(y_train_balanced, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train_balanced, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Matriu de confusió i mètriques
print(f"\nMSE entrenament: {mse_train:.4f}, MSE test: {mse_test:.4f}")
print(f"\nMAE entrenament: {mae_train:.4f}, MSE test: {mae_test:.4f}")

# Mostrar la Matriu de Confusió
print("Matriu de Confusió (Train):")
print(confusion_matrix(y_train_balanced, y_train_pred_class))
print("Matriu de Confusió (Test):")
print(confusion_matrix(y_test, y_test_pred_class))
    
# Mostrar mètriques de precision, recall i f1-score
print("Mètriques de classificació (Train):")
print(classification_report(y_train_balanced, y_train_pred_class))
print("Mètriques de classificació (Test):")
print(classification_report(y_test, y_test_pred_class))




