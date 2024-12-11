import subprocess
import os

def main():
    while True:
        print("\nMenu Principal")
        print("1. Model d'Alcoholisme")
        print("2. Model de notes")
        print("3. Sortir")

        try:
            opcio = int(input("\nTria una opció (1-3): "))
            if opcio == 1 or opcio == 2:
                print("Preparant dataset...")
                subprocess.run(["python", "preprocessament.py"])
                if opcio == 1:
                    print("Has triat el model d'alcoholisme.")
                    executar_model_alcoholisme()
                else:
                    print("Has triat el model notes d'estudiants.")
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
                executar_regressor()
            elif opcio == 2:
                print("Has triat el model Random Forest.")
                executar_random_forest()
            elif opcio == 3:
                print("Tornant al menú principal.")
                break
            else:
                print("Opció no vàlida. Si us plau, tria entre 1 i 3.")
        except ValueError:
            print("Entrada no vàlida. Si us plau, introdueix un número entre 1 i 3.")

def executar_regressor():
    print("[Execució del model Regressor]")
    script_path = os.path.join("regressio", "regressor_polinomic.py")
    subprocess.run(["python", script_path])

def executar_random_forest():
    print("[Execució del model Random Forest]")
    script_path = os.path.join("randomforest", "randomforest_classificador.py")
    subprocess.run(["python", script_path])

def executar_notes_estudiants():
    print("[Execució del model de les notes d'estudiants")
    script_path = os.path.join("g3", "prediccio_nota.py")
    subprocess.run(["python", script_path])

if __name__ == "__main__":
    main()

