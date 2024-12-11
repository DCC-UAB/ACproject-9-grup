import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Matriz de confusió i mètriques
print(f"\nMSE entrenament: {mse_train:.4f}, MSE test: {mse_test:.4f}")

# Mostrar la Matriz de Confusió
print("Matriz de Confusió (Train):")
print(confusion_matrix(y_train_balanced, y_train_pred_class))
print("Matriz de Confusió (Test):")
print(confusion_matrix(y_test, y_test_pred_class))
    
# Mostrar mètriques de precision, recall i f1-score
print("Mètriques de classificació (Train):")
print(classification_report(y_train_balanced, y_train_pred_class))
print("Mètriques de classificació (Test):")
print(classification_report(y_test, y_test_pred_class))

# Visualitzar els resultats
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6, label='Prediccions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Línia ideal')
plt.title("Prediccions vs Valors Reals (Regressió Ridge amb GridSearch)")
plt.xlabel("Valors Reals")
plt.ylabel("Prediccions")
plt.legend()
plt.grid(True)
plt.show()


