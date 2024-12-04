import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GB, AdaBoostRegressor as AB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Semilla
SEED = 1

# Carregar dades
df = pd.read_csv('student-mat.csv')

# Netejar valors no desitjats
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace("'", "").str.strip()

# Renombrar columnes per simplicitat
df.rename(columns={
    'absences': 'absències',
    'failures': 'fracassos',
    'goout': 'sortides',
    'freetime': 'temps',
    'age': 'edat',
    'health': 'salut',
    'G3': 'nota'
}, inplace=True)

# Mapatge per valors binaris
bin_map = {
    'yes': 1, 'no': 0,
    'GP': 1, 'MS': 0,
    'F': 1, 'M': 0,
    'U': 1, 'R': 0,
    'LE3': 0, 'GT3': 1,
    'T': 1, 'A': 0
}

cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

# Codificar columnes categòriques
for col in cat_cols:
    if df[col].dtype == 'object':
        if set(df[col].unique()).issubset(bin_map.keys()):
            df[col] = df[col].map(bin_map)
        else:
            df[col] = LabelEncoder().fit_transform(df[col])

# Separar variables (X: predictives, y: objectiu)
X = df.drop('nota', axis=1)
y = df['nota']

# Dividir entrenament i prova
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=SEED)

# Escalar dades
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir models
models = {
    'KNN': KNN(),
    'DT': DT(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'Random Forest': RF(random_state=SEED),
    'Gradient Boosting': GB(random_state=SEED),
    'AdaBoost': AB(random_state=SEED),
    'Linear Regression': LinearRegression()
}

# Hiperparàmetres per a cada model
# Definir los hiperparámetros para cada modelo
params = {
    'KNN': {'n_neighbors': [i for i in range(3, 50)]},
    'DT': {'max_depth': [i for i in range(1, 25)]},
    'Lasso': {'alpha': [i for i in range(0, 50)], 'tol': [0.1, 0.01, 0.001]},
    'Ridge': {'alpha': [i for i in range(0, 50)], 'tol': [0.1, 0.01, 0.001]},
    'ElasticNet': {'alpha': [i for i in range(0, 50)], 'l1_ratio': [0.1, 0.5, 0.9]},
    'SVR': {'kernel': ['linear', 'poly', 'rbf'], 'C': [i for i in range(1, 101)], 'epsilon': [0.01, 0.1]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30], 'max_features': [0.3, 0.5, 0.7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'Linear Regression': {'fit_intercept': [True, False], 'positive': [True, False]}
}

# Buscar millors paràmetres
best_params = {}
for model in models.keys():
    search = RandomizedSearchCV(models[model], params[model], n_iter=5, cv=3, random_state=SEED)
    search.fit(X_train, y_train)
    models[model] = search.best_estimator_
    best_params[model] = search.best_params_

# Avaluar mètriques
metrics = {}
predictions = {}
for model in models.keys():
    y_pred = models[model].predict(X_test)
    predictions[model] = y_pred
    metrics[model] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }

# Mostrar mètriques
metrics_df = pd.DataFrame(metrics).T.sort_values('MAE')
print(metrics_df)

# Gràfic MAE
plt.figure(figsize=(10, 6))
metrics_df['MAE'].plot(kind='barh', color='skyblue', edgecolor='black')
plt.title("MAE per Model")
plt.xlabel("MAE")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# Prediccions amb Random Forest
y_pred_rf = models['Random Forest'].predict(X_test)

# Gràfic de valors reals i prediccions
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.7, label='Valors Reals', marker='o')
plt.scatter(range(len(y_pred_rf)), y_pred_rf, color='orange', alpha=0.7, label='Prediccions', marker='x')
plt.title('Valors Reals vs Prediccions (Random Forest)', fontsize=16)
plt.xlabel('Índex', fontsize=12)
plt.ylabel('Nota', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Funció per classificar les notes
def classificar_nota(nota):
    if nota >= 17.5:
        return 'Excel·lent'
    elif nota >= 15.5:
        return 'Molt Bé'
    elif nota >= 13.5:
        return 'Bé'
    elif nota >= 9.5:
        return 'Suficient'
    elif nota >= 3.5:
        return 'Feble'
    else:
        return 'Pobre'

# Classificar valors reals i prediccions
y_real_class = y_test.apply(classificar_nota)  # Categories reals
y_pred_class = pd.Series(models['Random Forest'].predict(X_test)).apply(classificar_nota)  # Prediccions (Random Forest)

# Generar la matriu de confusió
cm = confusion_matrix(y_real_class, y_pred_class, labels=['Excel·lent', 'Molt Bé', 'Bé', 'Suficient', 'Feble', 'Pobre'])

# Mostrar la matriu de confusió
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Excel·lent', 'Molt Bé', 'Bé', 'Suficient', 'Feble', 'Pobre'])
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Matriu de Confusió: Categories de Qualificació")
plt.show()

# Mètriques d'avaluació
report = classification_report(y_real_class, y_pred_class, target_names=['Excel·lent', 'Molt Bé', 'Bé', 'Suficient', 'Feble', 'Pobre'])

# Mostrar les mètriques
print("\nInforme de Classificació:")
print(report)

print("\nInforme de Classificació:")