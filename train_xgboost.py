from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def train_xgboost():

    # Cargar datos balanceados
    X_train = pd.read_csv("X_train_balanced.csv")
    Y_train = pd.read_csv("Y_train_balanced.csv").squeeze()
    X_test = pd.read_csv("X_test.csv")
    Y_test = pd.read_csv("Y_test.csv").squeeze()

    # Seleccionar características significativas
    significant_features = ['sex', 'Fjob', 'studytime', 'famrel', 'goout', 'absences']
    X_train = X_train[significant_features]
    X_test = X_test[significant_features]

    # Entrenar modelo XGBoost
    xgb_model = XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)
    xgb_model.fit(X_train, Y_train)

    # Predicción y precisión
    predicted = xgb_model.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted)
    print(f"Precisión con XGBoost: {accuracy:.2f}")


if __name__ == "__main__":
    train_xgboost()
