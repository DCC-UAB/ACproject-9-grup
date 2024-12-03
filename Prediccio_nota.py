import pandas as pd             # data analysis
import numpy as np              # linear algebra + array handling
import sklearn                  # machine learning
import matplotlib.pyplot as plt # visualization
import seaborn as sns           # visualization
import pdpbox 
from sklearn.linear_model import Lasso, ARDRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler  # Para escalar los datos

# Cargar los datos
data = pd.read_csv('student-mat.csv')

# Reducir las variables y centrarse en las relevantes
cols = ['age', 'address', 'famsize', 'Pstatus', 
        'Medu', 'Fedu', 'studytime', 'schoolsup',
        'famsup', 'paid', 'activities', 'internet',
        'romantic', 'famrel', 'freetime', 'goout',
        'Dalc', 'Walc', 'traveltime', 'G3']
data = data[cols]

# Mapeo de variables categóricas a numéricas
mapping = {'address': {'U': 0, 'R': 1},
           'famsize': {'LE3': 0, 'GT3': 1},
           'Pstatus': {'T': 0, 'A': 1},
           'schoolsup': {'no': 0, 'yes': 1},
           'famsup': {'no': 0, 'yes': 1},
           'paid': {'no': 0, 'yes': 1},
           'activities': {'no': 0, 'yes': 1},
           'internet': {'no': 0, 'yes': 1},
           'romantic': {'no': 0, 'yes': 1}}

# Aplicar el mapeo a las columnas
for column in list(mapping.keys()):
    data[column] = data[column].map(mapping[column])

# Definir las variables predictoras y la variable objetivo
x = data.drop('G3', axis=1)
y = data['G3']

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = tts(x, y, train_size=0.7, random_state=101)

# Opcional: Escalar los datos (recomendable especialmente para SVR)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Definir los modelos
models = {
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regressor': SVR(),
    'AdaBoost Regressor': AdaBoostRegressor(),
    'LASSO': Lasso(),
    'Bayesian ARD Regressor': ARDRegression(),
    'ElasticNet': ElasticNet()
}

# Entrenar y evaluar los modelos
for name in models:
    model = models[name]
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f'{name}: {np.round(mae(y_test, prediction), 2)}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, prediction, color='blue', edgecolors='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Predicciones vs. Valores reales')
plt.xlabel('Valores reales (y_test)')
plt.ylabel('Predicciones')
plt.show()
