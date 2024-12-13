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

cvs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
mses_kf = []
mses_skf = []

for cv in cvs:
    kf = KFold(n_splits=cv)
    skf = StratifiedKFold(n_splits=cv)
    mse_kf = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
    mses_kf.append((cv, float(mse_kf)))
    mse_skf = -cross_val_score(pipeline, X, y, cv=skf, scoring='neg_mean_squared_error').mean()
    mses_skf.append((cv, float(mse_skf)))
    
kf_cv, kf_mse = zip(*mses_kf)
skf_cv, skf_mse = zip(*mses_skf)

best_kfold = (min(mses_kf, key=lambda x: x[1]))
print(f"\nBest Kfold {best_kfold[0]}")
print(f"Best MSE: {best_kfold[1]:.4f}")

plt.plot(kf_cv, kf_mse, marker='o', color='b')
plt.xlabel('KFolds')
plt.ylabel('Error Quadràtic Mig (MSE)')
plt.title('MSE KFold')
plt.grid(True)
plt.show()

best_skfold = (min(mses_skf, key=lambda x: x[1]))
print(f"\nBest StratifiedKfold {best_skfold[0]}")
print(f"Best MSE: {best_skfold[1]:.4f}")

plt.plot(skf_cv, skf_mse, marker='o', color='r')
plt.xlabel('KFolds')
plt.ylabel('Error Quadràtic Mig (MSE)')
plt.title('MSE StratifiedKFold')
plt.grid(True)
plt.show()
