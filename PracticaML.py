import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Llegir el fitxer CSV
df = pd.read_csv("student-mat.csv")

# Eliminar la columna "Dalc"
df.drop("Dalc", axis=1, inplace=True)

# Dividir el DataFrame en train i test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separar les característiques (X) i l'etiqueta (Y)
X_train = train_df.drop(columns=['Walc'])
Y_train = train_df['Walc']
X_test = test_df.drop(columns=['Walc'])
Y_test = test_df['Walc']

# Identificar i codificar les columnes categòriques en X
cat_cols = X_train.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder()

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Definir la matriu de distàncies entre les classes (aquí només per 5 classes)
distance_matrix = np.array([[0, 1, 2, 3, 4], 
                            [1, 0, 1, 2, 3], 
                            [2, 1, 0, 1, 2], 
                            [3, 2, 1, 0, 1], 
                            [4, 3, 2, 1, 0]])

# Crear un diccionari de pesos segons la distància
# Assignar el pes per distància (els errors més grans tenen més pes)
weights = {i: {j: distance_matrix[i][j] for j in range(5)} for i in range(5)}

# Funció per calcular els pesos de les classes
def compute_class_weights(y_true, weights_matrix):
    class_weights = np.zeros(len(np.unique(y_true)))
    for i, true_class in enumerate(np.unique(y_true)):
        weight_sum = 0
        for pred_class in np.unique(y_true):
            weight_sum += weights_matrix[true_class-1][pred_class-1]  # Indexa les classes per 0
        class_weights[true_class-1] = weight_sum
    return class_weights

# Calcular els pesos de les classes basant-se en la matriu de distància
class_weights = compute_class_weights(Y_train, distance_matrix)
print(class_weights)

# Crear el model de Random Forest amb pesos de les classes
model = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42)

# Entrenar el model
model.fit(X_train, Y_train)

# Fer prediccions sobre el conjunt de test
y_pred = model.predict(X_test)

# Evaluar el model
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Report de classificació per veure les mètriques
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Classe Predicha')
plt.ylabel('Classe Real')
plt.title('Matrícula de Confusió')
plt.show()