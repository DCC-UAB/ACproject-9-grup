import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Llegir el fitxer CSV
df = pd.read_csv("student-mat.csv")

# Filtratge de columnes
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 
        'freetime', 'goout', 'Dalc', 'health', 'absences', 'G1', 'G2', 'G3', 'Walc']
df = df[cols]

# Mapeig de les columnes categòriques
mapping = {
    'address': {'U': 0, 'R': 1},
    'famsize': {'LE3': 0, 'GT3': 1},
    'Pstatus': {'T': 0, 'A': 1},
    'schoolsup': {'no': 0, 'yes': 1},
    'famsup': {'no': 0, 'yes': 1},
    'paid': {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'internet': {'no': 0, 'yes': 1},
    'romantic': {'no': 0, 'yes': 1},
    'school': {'GP': 0, 'MS': 1},
    'sex': {'M': 0, 'F': 1},
    'higher': {'no': 0, 'yes': 1},
    'nursery': {'no': 0, 'yes': 1}
}

# Aplicar el mapeig
for column in mapping:
    df[column] = df[column].map(mapping[column])
    
df['G1'] = df['G1'].str.replace("'", "").astype(float).astype(int)    


# Separar les característiques (X) i l'etiqueta (Y)
X = df.drop(columns=['Walc'])
Y = df['Walc']

# Dividir en conjunt de train i test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalitzar les dades
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Comprovar distribució inicial de classes
print("Distribució inicial de les classes:", Counter(Y_train))

# Configurar SMOTE per balancejar les dades
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Comprovar distribució després de SMOTE
print("Distribució després de SMOTE:", Counter(Y_train_res))

# Crear el model Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)

# Realitzar cross-validation per avaluar el model
cv_scores = cross_val_score(model, X_train_res, Y_train_res, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MAE: {-np.mean(cv_scores):.4f}")

# Entrenar el model amb el conjunt de train balancejat
model.fit(X_train_res, Y_train_res)

# Fer prediccions sobre el conjunt de test
y_pred = model.predict(X_test)

# Avaluar el model
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f"MAE en el conjunt de test: {mae:.4f}")
print(f"R2 en el conjunt de test: {r2:.4f}")

# Gràfic de comparació entre valors reals i prediccions
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, y_pred, alpha=0.7, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.xlabel('Valors Reals')
plt.ylabel('Prediccions')
plt.title('Comparació entre valors reals i prediccions')
plt.show()

# Matriu de confusió per categories (convertir prediccions en enters)
y_pred_classes = np.round(y_pred).astype(int)
conf_matrix = pd.crosstab(Y_test, y_pred_classes, rownames=['Real'], colnames=['Predicció'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriu de Confusió')
plt.show()


