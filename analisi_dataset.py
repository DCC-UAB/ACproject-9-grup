import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Llegir el fitxer CSV
data = pd.read_csv("student-mat.csv")

walc_counts = data['Walc'].value_counts().sort_index()

print(walc_counts)


# Configuració del gràfic
plt.figure()
ax = sns.countplot(x=data['Walc'], palette='magma')

# Afegir etiquetes a les barres
ax.bar_label(ax.containers[0], fmt='%d', fontsize=10, padding=5)
# Ajustos visuals
ax.set_title('Distribution of Weekend Alcohol Consumption (Walc)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Weekend Alcohol Consumption (1=Low, 5=High)', fontsize=12)
ax.set_ylabel('Number of Students', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gràfic
plt.tight_layout()
#plt.show()



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
    



    
