import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random

SRC_DIR = "data/images_png"
DST_DIR = "data/images_augmented"
AUG_PER_IMAGE = 10
EXTENSIONS = [".png"]

os.makedirs(DST_DIR, exist_ok=True)

def is_image(filename):
    return any(filename.lower().endswith(ext) for ext in EXTENSIONS)

def augment_image(image):
    aug = image.copy()

    # Rotace
    if random.random() < 0.7:
        angle = random.uniform(-20, 20)
        aug = aug.rotate(angle, expand=True, fillcolor=(255, 255, 255, 0))

    # Zrcadlení
    if random.random() < 0.5:
        aug = ImageOps.mirror(aug)

    # Jas
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(aug)
        aug = enhancer.enhance(random.uniform(0.8, 1.2))

    # Kontrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(aug)
        aug = enhancer.enhance(random.uniform(0.8, 1.2))

    # Gaussian blur
    if random.random() < 0.3:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))

    # Resize back to 512x512 if needed
    aug = aug.resize((512, 512))
    return aug

def run_augmentation():
    for filename in os.listdir(SRC_DIR):
        if not is_image(filename):
            continue

        image_path = os.path.join(SRC_DIR, filename)
        image = Image.open(image_path).convert("RGBA")
        name, ext = os.path.splitext(filename)

        for i in range(AUG_PER_IMAGE):
            aug = augment_image(image)
            aug_name = f"{name}_aug_{i+1}.png"
            aug.save(os.path.join(DST_DIR, aug_name))

        print(f"✅ {filename}: vytvořeno {AUG_PER_IMAGE} variant")

if __name__ == "__main__":
    run_augmentation()
