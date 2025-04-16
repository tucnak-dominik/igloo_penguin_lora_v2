import os

AUG_DIR = "data/images_augmented"
CAPTION_PATH = "data/captions/captions.txt"

# Mapa názvů tučňáků k jejich popiskům
caption_map = {
    "penguin_01": "a happy penguin raising its right flipper",
    "penguin_02": "a penguin wearing headphones",
    "penguin_03": "a penguin wearing headphones",
    "penguin_04": "a penguin holding a dashboard in its right flipper",
    "penguin_05": "a penguin raising its left flipper",
    "penguin_06": "a neutral penguin in basic pose",
    "penguin_07": "a penguin wearing glasses",
    "penguin_08": "an angry penguin",
    "penguin_09": "a grumpy penguin",
    "penguin_10": "a winking penguin",
    "penguin_11": "a crazy penguin with a wild expression",
    "penguin_12": "a sad penguin",
    "penguin_13": "a surprised penguin",
    "penguin_14": "a penguin raising its right flipper"
}

with open(CAPTION_PATH, "w") as f:
    for filename in sorted(os.listdir(AUG_DIR)):
        if not filename.endswith(".png"):
            continue

        base_name = "_".join(filename.split("_")[:2])  # např. penguin_01_aug_1 → penguin_01
        if base_name in caption_map:
            prompt = f"{caption_map[base_name]} in igloonet brand style"
            f.write(f"{filename}|{prompt}\n")

print(f"✅ Captions vygenerovány do {CAPTION_PATH}")
