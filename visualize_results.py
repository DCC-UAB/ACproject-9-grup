import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def visualize_results():
    # Cargar predicciones
    predictions = pd.read_csv("predictions.csv")

    # Scatter plot de valores reales vs predichos
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions["True"], predictions["Predicted"], alpha=0.7, color="blue", label="Predicciones")
    plt.plot(
        [predictions["True"].min(), predictions["True"].max()],
        [predictions["True"].min(), predictions["True"].max()],
        color="red",
        linestyle="--",
        label="Línea Ideal",
    )
    plt.title("Valores Reales vs Predicciones")
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.legend()
    plt.grid(True)
    plt.savefig("scatter_plot.png")
    plt.show()

    # Matriz de confusión
    cm = confusion_matrix(predictions["True"], predictions["Predicted"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Matriz de Confusión")
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    visualize_results()
