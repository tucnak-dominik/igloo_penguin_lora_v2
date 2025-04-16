import os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model

# ========== NASTAVENÍ ========== #
model_id = "runwayml/stable-diffusion-v1-5"
instance_prompt = "a photo of igloo penguin"
output_dir = "outputs/igloonet_penguin_lora_v2"
data_dir = "data/images_augmented"

resolution = 512
batch_size = 1
learning_rate = 1e-4
num_epochs = 4

# ========== MODELY ========== #
print("📦 Načítám model...")

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to("cuda", dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda", dtype=torch.float16)

# ========== PŘIDÁNÍ LoRA ========== #
print("🧠 Přidávám LoRA váhy do UNet...")

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none",
    task_type="UNET",
)
unet = get_peft_model(unet, lora_config)

# ========== DATA ========== #
print(f"📁 Načítám obrázky ze složky: {data_dir}")
image_paths = list(Path(data_dir).glob("*.png"))
print(f"🖼️ Počet nalezených obrázků: {len(image_paths)}")

if not image_paths:
    raise ValueError("Nenalezeny žádné obrázky pro trénink.")

def load_images():
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((resolution, resolution))
        images.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0)
    return images

train_images = load_images()
print(f"✅ Nahráno {len(train_images)} obrázků pro trénink.")

# ========== TRÉNINK ========== #
print("🚀 Spouštím trénink...")

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

unet.train()
for epoch in range(num_epochs):
    print(f"🧪 Epoch {epoch + 1}/{num_epochs}")
    for img in tqdm(train_images):
        pixel_values = img.unsqueeze(0).to("cuda", dtype=torch.float16)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (1,), device="cuda").long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = tokenizer(
            instance_prompt, return_tensors="pt"
        ).input_ids.to("cuda")

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"✅ Epoch {epoch + 1} hotová!")

# ========== ULOŽENÍ ========== #
print(f"💾 Ukládám LoRA váhy do složky: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)

print("\n✅ Trénink dokončen! Lovec Šterbin je připraven generovat tučňáky.")