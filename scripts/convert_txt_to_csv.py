import csv

# Cesty k souborům
input_txt = "data/captions/captions.txt"
output_csv = "data/captions/captions.csv"

# Načti textový soubor
with open(input_txt, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

# Rozparsuj a přepiš do CSV
with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["file_name", "caption"])  # hlavička CSV

    for line in lines:
        if "|" in line:
            filename, caption = line.strip().split("|", 1)
            writer.writerow([filename, caption])
        else:
            print(f"⚠️ Přeskočena neplatná řádka: {line.strip()}")