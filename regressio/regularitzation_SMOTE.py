import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE 
from collections import Counter

# Carregar dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")
y = y['Walc']

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

# Aplicar SMOTE per balancejar les classes en el conjunt d'entrenament
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Comprovar distribució de classes abans i després de SMOTE
print("Distribució original de les classes:", Counter(y_train))
print("Distribució després de SMOTE:", Counter(y_train_res))

# Models a ajustar
models = {'Ridge': Ridge(), 'Lasso': Lasso()}

# Diccionari per la cerca en graella
param_grid = {
    'model__alpha': [0.01, 0.1, 1, 10, 100],  # 'model' és el nom del model dins del pipeline
    'model__fit_intercept': [True, False]     # Paràmetres per al model
}

# Creació de la cerca en graella per cada model
for model_name, model in models.items():
    # Crear el pipeline per a cada model
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),  # Característiques polinòmiques (grau 2)
        ('scaler', StandardScaler()),  # Normalitzar les dades
        ('model', model)  # Model (Ridge o Lasso)
    ])

    # Crear GridSearchCV per a cada model
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Entrenar el model amb GridSearchCV utilitzant les dades balancejades
    grid_search.fit(X_train_res, y_train_res)

    # Resultats de la cerca en graella
    print(f"\nMillors paràmetres per al model {model_name}: {grid_search.best_params_}")

    # Prediccions amb el millor model
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Mètriques
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"\nMSE {model_name} - Entrenament: {mse_train:.4f}, Test: {mse_test:.4f}")
    print(f"MAE {model_name} - Entrenament: {mae_train:.4f}, Test: {mae_test:.4f}")

    # Accuracy
    y_train_pred_class = [assign_class(pred) for pred in y_train_pred]
    y_test_pred_class = [assign_class(pred) for pred in y_test_pred]

    accuracy_train = accuracy_score(y_train, y_train_pred_class)
    accuracy_test = accuracy_score(y_test, y_test_pred_class)
    print(f"\nAccuracy entrenament: {accuracy_train:.4f}")
    print(f"Accuracy test: {accuracy_test:.4f}")

    # Matriu de confusió
    cm_test = confusion_matrix(y_test, y_test_pred_class)
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
