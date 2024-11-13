def convert_dat_to_dzn(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Estrai m e n
    m = int(lines[0].strip())
    n = int(lines[1].strip())

    # Estrai max_load
    max_load = list(map(int, lines[2].strip().split()))

    # Estrai weights
    weights = list(map(int, lines[3].strip().split()))

    # Estrai la matrice delle distanze
    distances = []
    for i in range(4, 4 + (n + 1)):
        row = list(map(int, lines[i].strip().split()))
        distances.append(row)

    # Scrivi i dati nel file .dzn
    with open(output_file, 'w') as file:
        file.write(f"m = {m};\n")
        file.write(f"n = {n};\n")
        file.write(f"max_load = {max_load};\n")
        file.write(f"weights = {weights};\n")
        
        # Formatta la matrice delle distanze nel modo richiesto
        file.write("distances = \n[|")
        file.write("\n | ".join(", ".join(map(str, row)) for row in distances))
        file.write("|];\n")

# Esempio di utilizzo:
n = "21"
convert_dat_to_dzn("instances/inst"+n+".dat", "dat_instances/inst"+n+".dzn")
