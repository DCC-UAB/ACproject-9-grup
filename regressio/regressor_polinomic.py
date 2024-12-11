import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

# Llegir el fitxer CSV
data = pd.read_csv("student-mat.csv")

# Filtratge de les columnes
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 
        'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

data = data[cols]

# Comprovar valors nuls
print("Valors nulls?", data.isna().any().any())

# Mapeig de variables categòriques
mapping = {'school': {'GP': 0, 'MS': 1},
           'sex': {'F': 0, 'M': 1},
           'address': {'U': 0, 'R': 1},
           'famsize': {'LE3': 0, 'GT3': 1},
           'Pstatus': {'T': 0, 'A': 1},
           'schoolsup': {'no': 0, 'yes': 1},
           'famsup': {'no': 0, 'yes': 1},
           'paid': {'no': 0, 'yes': 1},
           'activities': {'no': 0, 'yes': 1},
           'nursery': {'no': 0, 'yes': 1},
           'higher': {'no': 0, 'yes': 1},
           'internet': {'no': 0, 'yes': 1},
           'romantic': {'no': 0, 'yes': 1}}

for column in mapping.keys():
    data[column] = data[column].map(mapping[column])
    
data['G1'] = data['G1'].str.replace("'", "").astype(float).astype(int)

# Dividir dades en X i y
X = data.drop(columns=["Walc"])
y = data["Walc"]

# Dividir en conjunt d'entrenament i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanceig amb SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Distribució de classes després de SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

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

# Configuració de Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 particions

# Funció per avaluar el model dins del cross-validation
def evaluate_model_with_cv(model, X, y, degree):
    # Generació de característiques polinòmiques
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Cross-validation
    scores = cross_val_score(model, X_poly, y, cv=kfold, scoring='neg_mean_squared_error')
    return -scores  # Canviem a valors positius per al MSE

# Avaluem el model amb cross-validation
degree = 3  # Grau del polinomi
mse_scores = evaluate_model_with_cv(LinearRegression(), X, y, degree)

# Resultats
print(f"Resultats Cross-Validation (MSE): {mse_scores}")
print(f"MSE Promig: {np.mean(mse_scores):.4f}")
print(f"MSE Desviació Estàndard: {np.std(mse_scores):.4f}")

# Entrenar el model amb tot el conjunt d'entrenament
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train_balanced)
X_test_poly = poly.transform(X_test)

# Entrenar el model de regressió lineal
model = LinearRegression()
model.fit(X_train_poly, y_train_balanced)

# Prediccions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Avaluar el model
mse_train = mean_squared_error(y_train_balanced, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"MSE entrenament: {mse_train:.4f}")
print(f"MSE test: {mse_test:.4f}")

# Visualització dels resultats
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6, label='Prediccions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Línia ideal')
plt.title("Prediccions vs Valors Reals (Regressió Polinòmica)")
plt.xlabel("Valors Reals")
plt.ylabel("Prediccions")
plt.legend()
plt.grid(True)
plt.show()
