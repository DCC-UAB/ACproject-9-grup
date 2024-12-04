# Imports necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from Prediccio_nota import cargar_y_preprocesar_datos

# Función para clasificar las notas
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

# Cargar datos procesados
X,X_train, X_test, y_train, y_test = cargar_y_preprocesar_datos('student-mat.csv')

# Convertir las notas a categorías
y_train_class = pd.Series(y_train).apply(classificar_nota)
y_test_class = pd.Series(y_test).apply(classificar_nota)

# Entrenar el modelo con categorías
rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train, y_train_class)

# Predicciones del modelo
y_pred_class = rf_classifier.predict(X_test)

# Generar la matriz de confusión
cm = confusion_matrix(y_test_class, y_pred_class, labels=['Excel·lent', 'Molt Bé', 'Bé', 'Suficient', 'Feble', 'Pobre'])

# Convertir la matriz de confusión a porcentajes
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Mostrar la matriz de confusión en porcentajes
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=['Excel·lent', 'Molt Bé', 'Bé', 'Suficient', 'Feble', 'Pobre'])
disp.plot(cmap='Blues', xticks_rotation='vertical', values_format=".1f")
plt.title("Matriu de Confusió en Percentatges")
plt.show()

# Informe de clasificación
report = classification_report(y_test_class, y_pred_class, target_names=['Excel·lent', 'Molt Bé', 'Bé', 'Suficient', 'Feble', 'Pobre'])
print("\nInforme de Classificació:")
print(report)

# Métrica adicional: Macro F1-score
f1_macro = f1_score(y_test_class, y_pred_class, average='macro')
print("\nF1-score macro:", f1_macro)
