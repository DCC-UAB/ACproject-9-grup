import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

# Carregar les dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")

# Crear el pipeline per a la cerca de hiperparàmetres
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),  # Característiques polinòmiques
    ('scaler', StandardScaler()),  # Normalitzar les dades
    ('model', Ridge(alpha=100, fit_intercept=True))  # Regressió Ridge amb alpha = 100
])
kf = KFold(n_splits=8)

