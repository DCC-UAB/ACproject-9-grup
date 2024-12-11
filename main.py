import subprocess
import os

def main():
    while True:
        print("\nMenu Principal del ML per un dataset d'estudiants.")
        print("1. Model d'Alcoholisme")
        print("2. Model de notes")
        print("3. Sortir")

        try:
            opcio = int(input("\nTria una opció (1-3): "))

            if opcio == 1:
                print("Has triat el model d'alcoholisme dels estudiants.")
                # Aquí cridaries la funció o mòdul corresponent
                executar_model_alcoholisme()
            elif opcio == 2:
                print("Has triat el model de notes dels estudiants.")
                # Aquí cridaries la funció o mòdul corresponent
                executar_notes_estudiants()
            elif opcio == 3:
                print("Sortint del programa. Adéu!")
                break
            else:
                print("Opció no vàlida. Si us plau, tria entre 1 i 3.")
        except ValueError:
            print("Entrada no vàlida. Si us plau, introdueix un número entre 1 i 3.")

def executar_model_alcoholisme():
    while True:
        print("\nModel d'Alcoholisme")
        print("1. Regressor")
        print("2. Random Forest")
        print("3. Tornar al menú principal")

        try:
            opcio = int(input("\nTria una opció (1-3): "))

            if opcio == 1:
                print("Has triat el model Regressor.")
                # Aquí afegeix la funcionalitat del model Regressor
                executar_regressor()
            elif opcio == 2:
                print("Has triat el model Random Forest.")
                # Aquí afegeix la funcionalitat del model Random Forest
                executar_random_forest()
            elif opcio == 3:
                print("Tornant al menú principal.")
                break
            else:
                print("Opció no vàlida. Si us plau, tria entre 1 i 3.")
        except ValueError:
            print("Entrada no vàlida. Si us plau, introdueix un número entre 1 i 3.")

def executar_regressor():
    # Placeholder per a la funcionalitat del model Regressor
    print("Execució del model Regressor")
    script_path = os.path.join("regressio", "regressor_plinomic.py")
    subprocess.run(["python", script_path])

def executar_random_forest():
    # Placeholder per a la funcionalitat del model Random Forest
    print("Execució del model Random Forest")
    script_path = os.path.join("randomforest", "randomforest_classificador.py")
    subprocess.run(["python", script_path])

def executar_notes_estudiants():
    # Placeholder per a la funcionalitat del sistema de notes
    print("Model d'estudiants en procés")
    script_path = os.path.join("g3", "prediccio_nota.py")
    subprocess.run(["python", script_path])

if __name__ == "__main__":
    main()
