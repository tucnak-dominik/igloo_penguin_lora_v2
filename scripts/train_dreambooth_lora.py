import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer
from peft import LoraConfig, get_peft_model

# ========== Nastavení ==========
model_id = "runwayml/stable-diffusion-v1-5"
instance_prompt = "a penguin"
validation_prompt = "a penguin in igloonet style"

instance_data_dir = "data/images_augmented"
captions_path = "data/captions/captions.csv"
output_dir = "outputs/igloo_penguin_lora"

resolution = 512
batch_size = 1
learning_rate = 1e-4
max_train_steps = 1000
num_epochs = 4

# ========== Načtení modelu ==========
print("📦 Načítám model...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
unet: UNet2DConditionModel = pipe.unet
vae: AutoencoderKL = pipe.vae

# ========== Přidání LoRA váh ==========
print("🧠 Přidávám LoRA váhy do UNet...")
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type="UNET"
)
unet = get_peft_model(unet, lora_config)

# ========== Načtení dat ==========
print(f"📁 Načítám obrázky ze složky: {instance_data_dir}")
image_paths = sorted(Path(instance_data_dir).glob("*.png"))
captions_df = pd.read_csv(captions_path)

transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
])

def load_images():
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).half().unsqueeze(0).to("cuda")
        images.append(tensor)
    return images

train_images = load_images()
print(f"🖼️ Počet nalezených obrázků: {len(train_images)}")

# ========== Trénink (mock) ==========
print("🚀 Spouštím trénink...")

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"🧪 Epoch {epoch + 1}/{num_epochs}")
    for i, (img_tensor, path) in enumerate(zip(train_images, image_paths)):
        optimizer.zero_grad()
        caption = captions_df[captions_df["file_name"] == path.name]["caption"].values[0]

        # Zde by mělo být zakódování promtu přes text encoder + noise prediction + loss
        # → Nahraď následující řádek reálným výpočtem loss při integraci s plným DreamBooth tréninkem
        loss = torch.rand(1).to("cuda")  # dummy loss

        loss.backward()
        optimizer.step()

        if (i + 1) % 25 == 0:
            print(f"  🔄 Step {i + 1}/{len(train_images)} - Loss: {loss.item():.4f}")

# ========== Uložení ==========
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"✅ Trénink dokončen. Model uložen do: {output_dir}")