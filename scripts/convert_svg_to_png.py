import os
import cairosvg

SRC_DIR = "data/images_raw"
DST_DIR = "data/images_png"
SIZE = 512

os.makedirs(DST_DIR, exist_ok=True)

for filename in os.listdir(SRC_DIR):
    if not filename.endswith(".svg"):
        continue

    input_path = os.path.join(SRC_DIR, filename)
    output_name = os.path.splitext(filename)[0] + ".png"
    output_path = os.path.join(DST_DIR, output_name)

    cairosvg.svg2png(url=input_path, write_to=output_path, output_width=SIZE, output_height=SIZE)
    print(f"✅ {filename} → {output_name}")