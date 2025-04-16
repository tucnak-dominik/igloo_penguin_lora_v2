import os
from pathlib import Path

from diffusers import StableDiffusionPipeline
from diffusers.utils import export_to_video
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTokenizer

from diffusers import DDPMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from peft import LoraConfig
from diffusers.utils.torch_utils import randn_tensor

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

# ========== KONFIGURAČNÍ ČÁST ========== #
model_id = "runwayml/stable-diffusion-v1-5"
instance_prompt = "a photo of igloo penguin"
output_dir = "outputs/igloonet_penguin_lora_v2"
data_dir = "data/images_augmented"

resolution = 512
batch_size = 1
learning_rate = 1e-4
num_train_epochs = 6

# ========== PŘÍPRAVA ========== #
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# Dataset loading
image_paths = list(Path(data_dir).glob("*.png"))

# 🐞 DEBUG výpisy
print("📁 Načítám obrázky ze složky:", os.path.abspath(data_dir))
print("🖼️ Počet nalezených obrázků:", len(image_paths))

if not image_paths:
    raise ValueError(f"Nenalezeny žádné obrázky pro trénink ve složce: {data_dir}")

def load_images():
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((resolution, resolution))
        images.append(img)
    return images

train_images = load_images()
print(f"✅ Nahráno {len(train_images)} obrázků pro trénink.")

# ========== TRÉNINK ========== #
from diffusers import StableDiffusionLoRATrainer
from diffusers.training_utils import save_lora_weights

trainer = StableDiffusionLoRATrainer(
    pipeline=pipe,
    instance_prompt=instance_prompt,
    train_images=train_images,
    output_dir=output_dir,
    train_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    resolution=resolution,
)

print("🚀 Spouštím trénink...")
trainer.train()

# Uložení
save_lora_weights(pipe.unet, output_dir)
print(f"\n✅ Trénink dokončen! Výsledky uloženy v: {output_dir}")