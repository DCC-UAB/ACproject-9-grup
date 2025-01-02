import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Llegir el fitxer CSV
data = pd.read_csv("student-mat.csv")

# Comptar els valors de Walc
walc_counts = data['Walc'].value_counts().sort_index()

print("Valors de l'objectiu (Walc):",walc_counts)

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

    



    
