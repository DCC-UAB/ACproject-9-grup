import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import KFold
import numpy as np


def cross_validate_model(X, Y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(X):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        Y_train_cv, Y_val_cv = Y.iloc[train_index], Y.iloc[val_index]

        # Entrenar modelo
        ordinal_model = OrderedModel(Y_train_cv, X_train_cv, distr='logit')
        ordinal_result = ordinal_model.fit(method='bfgs', disp=False)

        # Predicción
        predicted = ordinal_result.predict(X_val_cv)
        predicted_categories = predicted.idxmax(axis=1)

        # Calcular precisión
        accuracy = (predicted_categories == Y_val_cv).mean()
        accuracies.append(accuracy)

    print(f"Precisión promedio en validación cruzada: {np.mean(accuracies):.2f}")
    return np.mean(accuracies)


def train_ordinal_model():
    # Cargar datos balanceados
    X_train = pd.read_csv("X_train_balanced.csv")
    Y_train = pd.read_csv("Y_train_balanced.csv").squeeze()
    X_test = pd.read_csv("X_test.csv")
    Y_test = pd.read_csv("Y_test.csv").squeeze()

    # Seleccionar características significativas
    significant_features = ['sex', 'Fjob', 'studytime', 'famrel', 'goout', 'absences']
    X_train = X_train[significant_features]
    X_test = X_test[significant_features]

    # Entrenar modelo ordinal
    ordinal_model = OrderedModel(Y_train, X_train, distr='logit')
    ordinal_result = ordinal_model.fit(method='bfgs')

    # Guardar resumen del modelo
    with open("model_summary.txt", "w") as f:
        f.write(ordinal_result.summary().as_text())

    # Validación cruzada
    cross_validate_model(X_train, Y_train)

    # Predicciones
    predicted = ordinal_result.predict(X_test)
    predicted_categories = predicted.idxmax(axis=1)

    # Calcular precisión final
    accuracy = (predicted_categories == Y_test).mean()
    print(f"Precisión final: {accuracy:.2f}")

    # Guardar predicciones
    pd.DataFrame({"True": Y_test, "Predicted": predicted_categories}).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    train_ordinal_model()
