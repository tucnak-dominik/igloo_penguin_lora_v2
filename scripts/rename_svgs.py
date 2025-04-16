import os

# Složka s obrázky
DIR = "data/images_raw"

# Seznam všech .svg souborů v adresáři
svg_files = [f for f in os.listdir(DIR) if f.lower().endswith(".svg")]
svg_files.sort()  # Seřadit pro stabilní přejmenování

# Přejmenování
for i, filename in enumerate(svg_files, start=1):
    src = os.path.join(DIR, filename)
    new_name = f"penguin_{i:02d}.svg"
    dst = os.path.join(DIR, new_name)
    os.rename(src, dst)
    print(f"{filename} → {new_name}")