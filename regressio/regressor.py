import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_validate
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

kf = 5  # Nombre de folds

# Validació creuada i càlcul de les mètriques MSE i MAE
cv_results = cross_validate(model, X, y, cv=kf, scoring=None, return_train_score=False)

# Prediccions per a la validació creuada
y_pred = cross_val_predict(model, X, y, cv=kf)

# Calcula el MSE i MAE per cada fold de la validació creuada
mse_scores = []
mae_scores = []

for i in range(kf):
    start_idx = i * (len(y) // kf)
    end_idx = (i + 1) * (len(y) // kf)
    fold_y_true = y.iloc[start_idx:end_idx]
    fold_y_pred = y_pred[start_idx:end_idx]
    
    mse_scores.append(float(mean_squared_error(fold_y_true, fold_y_pred)))  # Convertir a float
    mae_scores.append(float(mean_absolute_error(fold_y_true, fold_y_pred)))  # Convertir a float

# Mostrar els resultats
print(f"MSE scores for each fold: {mse_scores}")
print(f"Average MSE: {np.mean(mse_scores)}")
print(f"MAE scores for each fold: {mae_scores}")
print(f"Average MAE: {np.mean(mae_scores)}")


# Crear gràfica per visualitzar les mètriques MSE i MAE
fig, ax = plt.subplots(figsize=(8, 6))

# Mostrar MSE i MAE
ax.bar(np.arange(kf) - 0.2, mse_scores, 0.4, label='MSE')
ax.bar(np.arange(kf) + 0.2, mae_scores, 0.4, label='MAE')

# Afegir títol i etiquetes
ax.set_title('MSE i MAE per Fold (Cross-Validation)')
ax.set_xlabel('Fold')
ax.set_ylabel('Valor')
ax.legend()



#CLASSIFIQUEM ELS VALORS PREDITS-----------------------------------------------------------------
# Convertim les prediccions a classes
y_pred_classes = np.array([assign_class(val) for val in y_pred])

# Calculant la matriu de confusió
conf_matrix = confusion_matrix(y, y_pred_classes)
print("\nMètriques valors classificats:")
# Mostrar la matriu de confusió
print("Matriu de confusio:")
print(conf_matrix)

# Càlcul de mètriques
accuracy = accuracy_score(y, y_pred_classes)
precision = precision_score(y, y_pred_classes, average='weighted')  # 'weighted' pondera les mètriques per la seva freqüència
recall = recall_score(y, y_pred_classes, average='weighted')
f1 = f1_score(y, y_pred_classes, average='weighted')

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
