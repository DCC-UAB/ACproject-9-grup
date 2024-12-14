import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

# Carregar les dades preprocessades
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

# Crear el pipeline per a la cerca de hiperparàmetres
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),  # Característiques polinòmiques
    ('scaler', StandardScaler()),  # Normalitzar les dades
    ('model', Ridge(alpha=100, fit_intercept=True))  # Regressió Ridge amb alpha = 100
])
kf = KFold(n_splits=8)

# Obtenir les prediccions utilitzant validació creuada
y_pred = cross_val_predict(pipeline, X, y, cv=kf)

#Calcular el MSE mitja
mse_mean = mean_squared_error(y, y_pred)
print(f"MSE mitjà: {mse_mean:.4f}")
mae_mean = mean_absolute_error(y, y_pred)
print(f"MAE mitjà: {mae_mean:.4f}")








