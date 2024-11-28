import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE


def prepare_data():
    # Cargar el dataset
    df = pd.read_csv("student-mat.csv")

    # Eliminar la columna innecesaria
    df.drop("Dalc", axis=1, inplace=True)

    # Separar características (X) y etiqueta (Y)
    X = df.drop(columns=['Walc'])
    Y = df['Walc']

    # Codificar variables categóricas
    cat_cols = X.select_dtypes(include=['object']).columns
    encoder = OrdinalEncoder()
    X[cat_cols] = encoder.fit_transform(X[cat_cols])

    # Dividir datos en train y test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Balancear clases con SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)

    # Guardar datos procesados
    X_train_balanced.to_csv("X_train_balanced.csv", index=False)
    Y_train_balanced.to_csv("Y_train_balanced.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    Y_test.to_csv("Y_test.csv", index=False)

    print("Datos preparados y guardados.")


if __name__ == "__main__":
    prepare_data()
