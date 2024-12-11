import pandas as pd

# Llegir el fitxer CSV
data = pd.read_csv("student-mat.csv")

# Filtratge de les columnes
cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
        'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 
        'freetime', 'goout', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

data = data[cols]

# Comprovar valors nuls
print("Valors nulls?", data.isna().any().any())

# Mapeig de variables categòriques
mapping = {'school': {'GP': 0, 'MS': 1},
           'sex': {'F': 0, 'M': 1},
           'address': {'U': 0, 'R': 1},
           'famsize': {'LE3': 0, 'GT3': 1},
           'Pstatus': {'T': 0, 'A': 1},
           'schoolsup': {'no': 0, 'yes': 1},
           'famsup': {'no': 0, 'yes': 1},
           'paid': {'no': 0, 'yes': 1},
           'activities': {'no': 0, 'yes': 1},
           'nursery': {'no': 0, 'yes': 1},
           'higher': {'no': 0, 'yes': 1},
           'internet': {'no': 0, 'yes': 1},
           'romantic': {'no': 0, 'yes': 1}}

for column in mapping.keys():
    data[column] = data[column].map(mapping[column])

data['G1'] = data['G1'].str.replace("'", "").astype(float).astype(int)

# Dividir dades en X i y
X = data.drop(columns=["Walc"])
y = data["Walc"]

X.to_csv("X_preprocessed.csv", index=False)
y.to_csv("y_preprocessed.csv", index=False)