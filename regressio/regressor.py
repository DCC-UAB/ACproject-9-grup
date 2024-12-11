import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Carregar dades preprocessades
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv")

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


# Model de regressió lineal
model = LinearRegression()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

train_predict = model.predict(X_train)
y_pred = model.predict(X_test)

print("Metriques regressor:")

#Error
train_predict = model.predict(X_train)

train_error = (y_train-train_predict).values
test_error = (y_test-y_pred).values


plt.plot(train_error, '.')
plt.plot(test_error, '.r')

plt.axhline(y=np.mean(train_error), color='b', linestyle='-', label='Mean train error')
plt.axhline(y=np.std(train_error), color='b', linestyle='--', label='Std train error')

plt.axhline(y=np.mean(test_error), color='r', linestyle='-', label='Mean test error')
plt.axhline(y=np.std(test_error), color='r', linestyle='--', label='Std test error')
plt.title("Train vs Test")
plt.xlabel("Instàncies de prediccions")
plt.ylabel("Error")
plt.legend()

#Errors normalitzats histograma
plt.figure()
plt.hist(train_error, density = True, histtype = 'step', label = "Train")
plt.hist(test_error,density = True, histtype = 'step', label = "Test")
plt.title("Histograma errors normalitzats Train vs Test")
plt.xlabel('Error')
plt.ylabel('Densitat')
plt.legend()


# Calcula MSE i MAE per al conjunt de train
train_mse = mean_squared_error(y_train, train_predict)
train_mae = mean_absolute_error(y_train, train_predict)

# Calcula MSE i MAE per al conjunt de test
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)

# Mostra els resultats
print("MSE Train:", train_mse)
print("MAE Train:", train_mae)
print("MSE Test:", test_mse)
print("MAE Test:", test_mae)


#CLASSIFIQUEM ELS VALORS PREDITS-----------------------------------------------------------------
# Convertim les prediccions a classes
y_pred_classes = np.array([assign_class(val) for val in y_pred])

# Calculant la matriu de confusió
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("\nMètriques valors classificats:")
# Mostrar la matriu de confusió
print("Matriu de confusio:")
print(conf_matrix)

# Càlcul de mètriques
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')  # 'weighted' pondera les mètriques per la seva freqüència
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

# Opcional: Visualitzar la matriu de confusió com una imatge
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriu de Confusio")
plt.colorbar()
tick_marks = np.arange(5)
plt.xticks(tick_marks, [1, 2, 3, 4, 5])
plt.yticks(tick_marks, [1, 2, 3, 4, 5])

# Etiquetes de les matrius
thresh = conf_matrix.max() / 2.
for i in range(5):
    for j in range(5):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('Classe real')
plt.xlabel('Classe predita')
plt.tight_layout()
plt.show()
