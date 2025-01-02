import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_curve(model, X_test, Y_test, model_name):
    """
    Genera y muestra la curva ROC para un modelo específico.

    Args:
        model: El modelo entrenado.
        X_test: Datos de prueba (features).
        Y_test: Etiquetas verdaderas de prueba.
        model_name: Nombre del modelo (para el título del gráfico).
    """
    # Obtener las probabilidades de la clase positiva
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        raise ValueError(f"El modelo {model_name} no soporta predict_proba ni decision_function.")

    # Calcular FPR, TPR y AUC
    fpr, tpr, _ = roc_curve(Y_test, y_proba)
    auc = roc_auc_score(Y_test, y_proba)

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.title(f"Curva ROC - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

# Ejemplo de uso:
# plot_roc_curve(model_knn, X_test, Y_test, "KNN")
