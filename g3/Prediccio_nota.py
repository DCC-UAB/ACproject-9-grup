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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Cargar los datos
data = pd.read_csv('student-mat.csv')

# Selección de características más importantes (identificadas previamente)
selected_features = ['goout', 'age', 'Medu', 'freetime', 'studytime', 'Walc', 'Fedu', 'famrel', 'G3']

# Reducir el dataset a las características seleccionadas
data = data[selected_features]

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

# Definir una métrica personalizada para la validación cruzada (MAE)
scorer = make_scorer(mae, greater_is_better=False)

# Evaluar cada modelo usando validación cruzada
for name, model in models.items():
    scores = cross_val_score(model, x, y, cv=100, scoring=scorer)  # 5 folds
    mean_score = -scores.mean()  # Invertimos el signo para que sea MAE positivo
    std_score = scores.std()
    print(f"{name}: MAE promedio: {np.round(mean_score, 2)}")

