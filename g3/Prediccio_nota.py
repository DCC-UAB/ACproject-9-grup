import pandas as pd             # data analysis
import numpy as np              # linear algebra + array handling
import matplotlib.pyplot as plt # visualization
import seaborn as sns           # visualization
from sklearn.linear_model import Lasso, ARDRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor as KNN 
from sklearn.tree import DecisionTreeRegressor as DT 
from sklearn.linear_model import Lasso
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.preprocessing import MinMaxScaler as scaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error as MAE 
from sklearn.model_selection import RandomizedSearchCV as RSCV 

# Cargar los datos
data = pd.read_csv('student-mat.csv')

# Selección de características más importantes (identificadas previamente)
selected_features = [
    "absences",
    "failures",
    "goout",
    "freetime",
    "age",
    "health",
    "G3"
]

# Reducir el dataset a las características seleccionadas
data = data[selected_features]

# Definir las variables predictoras y la variable objetivo
x = data.drop('G3', axis=1)
y = data['G3']

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = tts(x, y, train_size=0.75, random_state=1)

# Escalar los datos (recomendable especialmente para SVR)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Definir los modelos
models={'KNN': KNN(), 
        'DT': DT(), 
        'Lasso': Lasso(),
        'SVR': SVR(), 
        'RF': RF(random_state=1)}

parameters={'KNN':{'n_neighbors':[i for i in range(3,50)]}, 
           'DT':{'max_depth':[i for i in range(1,25)]}, 
           'Lasso':{'alpha':[i for i in range(50)], 'tol':[0.1,0.01,0.001,0.0001], 'max_iter':[j for j in range(100,1100,100)]},
           'SVR':{'kernel':['linear', 'poly', 'rbf'], 'C':[i for i in range(101)],
                  'epsilon':[0.0001, 0.001, 0.01, 0.1], 'gamma':['scale','auto']},
           'RF':{'n_estimators':[i for i in range(10,100)],  
                'max_depth':[i for i in range(1,25)], 'max_features':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}}

cval={}
for model in models.keys():
    cval[model]=RSCV(models[model], parameters[model], cv=10, scoring='neg_mean_absolute_error')
    cval[model].fit(x_train, y_train)
    print('The best parameters for '+model+' are: {}'.format(cval[model].best_params_))


models['KNN']=cval['KNN'].best_estimator_

models['DT']=cval['DT'].best_estimator_

models['Lasso']=cval['Lasso'].best_estimator_

models['SVR']=cval['SVR'].best_estimator_

models['RF']=cval['RF'].best_estimator_

y_pred={}
errors={}
for model in models.keys():
    y_pred[model]=models[model].predict(x_test)
    errors[model]=round(MAE(y_test, y_pred[model]),3)
    
plt.figure(figsize=(15,8))
ax=pd.Series(errors).sort_values().plot(kind='barh', rot=0, color=sns.color_palette('spring'), 
                    title='Mean Absolute Error by Different Models')

for container in ax.containers:
    ax.bar_label(container)    
plt.xlabel('Mean Absolute Error')
plt.ylabel('Models')
plt.show()