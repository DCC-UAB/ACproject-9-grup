import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Llegir el fitxer CSV
data = pd.read_csv("student-mat.csv")

# Comptar els valors de Walc
walc_counts = data['Walc'].value_counts().sort_index()

# Configuració del gràfic
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=data['Walc'], palette='magma')

# Afegir etiquetes a les barres amb els valors exactes
for container in ax.containers:
    ax.bar_label(container, fmt='%d', fontsize=10, padding=5)

# Ajustos visuals
ax.set_title('Distribution of Weekend Alcohol Consumption (Walc)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Weekend Alcohol Consumption (1=Low, 5=High)', fontsize=12)
ax.set_ylabel('Number of Students', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gràfic
plt.tight_layout()
plt.show()



#Filtratge
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 
        'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

data = data[cols]

mapping = {'address': {'U':0, 'R':1},
           'famsize': {'LE3':0, 'GT3':1},
           'Pstatus': {'T':0, 'A':1},
           'schoolsup':{'no':0,'yes':1},
           'famsup':{'no':0,'yes':1},
           'paid':{'no':0,'yes':1},
           'activities':{'no':0,'yes':1},
           'internet':{'no':0,'yes':1},
           'romantic':{'no':0,'yes':1}}

for column in list(mapping.keys()):
    data[column] = data[column].map(mapping[column])
    
print("Valors null?",data.isna().any().any())

count = {}
# Recorrem la columna "Walc" de cada fila
for value in data['Walc']:
    if value in count:
        count[value] += 1
    else:
        count[value] = 1

print("Valors de l'objectiu (Walc):",count)
    



    
