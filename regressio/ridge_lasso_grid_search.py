import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Carregar dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")

# Dividim les dades en entrenament i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models a ajustar
models = {'Ridge': Ridge(), 'Lasso': Lasso()}
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],  # Valors per ajustar la regularització
    'fit_intercept': [True, False]     # Incloure o no el bias
}

# Resultats
results = {}

# Bucle per ajustar cada model
for model_name, model in models.items():
    print(f"\n--- Ajustant model: {model_name} ---")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=50, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    
    # Guardem els resultats
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convertim a positiu
    
    results[model_name] = {
        'Best Params': best_params,
        'Best Score (MSE)': best_score
    }
    
    print(f"Millors paràmetres per a {model_name}: {best_params}")
    print(f"Millor puntuació (MSE): {best_score}")
    
    # Reajustar el model amb els millors paràmetres
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)  # Entrenament amb tot el conjunt de train
    
    # Prediccions
    y_pred = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    print(f"Test MSE per a {model_name}: {mse_test}")

    # Guardem les prediccions i el test MSE
    results[model_name]['Test MSE'] = mse_test
    results[model_name]['Predictions'] = y_pred

# Visualitzar els resultats
print("\n--- Resum dels resultats ---")
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(f"  Millors paràmetres: {result['Best Params']}")
    print(f"  Millor puntuació (Train MSE): {result['Best Score (MSE)']}")
    print(f"  Puntuació Test MSE: {result['Test MSE']}")

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

# Postprocessament per Ridge i Lasso (CLASSIFICACIO VALORS PREDITS)
for model_name, result in results.items():
    print(f"\n--- Postprocessament per {model_name} ---")
    y_pred = result['Predictions']  # Prediccions ja generades al pas anterior

    # Convertim les prediccions a classes
    y_pred_classes = np.array([assign_class(val) for val in y_pred])

    # Calculant la matriu de confusió
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print("\nMètriques valors classificats:")
    # Mostrar la matriu de confusió
    print("Matriu de confusio:")
    print(conf_matrix)

    # Càlcul de mètriques
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')  # 'weighted' pondera les mètriques per la seva freqüència
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")

    # Opcional: Visualitzar la matriu de confusió com una imatge
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriu de Confusio per {model_name}")
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, [1, 2, 3, 4, 5])
    plt.yticks(tick_marks, [1, 2, 3, 4, 5])

    # Etiquetes de les matrius
    thresh = conf_matrix.max() / 2.
    for i in range(5):
        for j in range(5):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('Classe real')
    plt.xlabel('Classe predita')
    plt.tight_layout()
    plt.show()


