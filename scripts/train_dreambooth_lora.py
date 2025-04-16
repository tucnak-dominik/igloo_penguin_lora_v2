import os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer
from peft import LoraConfig, get_peft_model
from diffusers.utils import load_image
from accelerate import Accelerator

# ========== KONFIGURACE ========== #
model_id = "runwayml/stable-diffusion-v1-5"
instance_prompt = "a photo of igloo penguin"
output_dir = "outputs/igloonet_penguin_lora_v2"
data_dir = "data/images_augmented"
resolution = 512
batch_size = 1
learning_rate = 1e-4
num_train_epochs = 6

# ========== INIT ========== #
print("📦 Načítám model...")
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

tokenizer = CLIPTokenizer.from_pretrained(model_id)
unet = pipeline.unet
vae = pipeline.vae
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# ========== DATA ========== #
print(f"📁 Načítám obrázky ze složky: {data_dir}")
image_paths = list(Path(data_dir).glob("*.png"))
if not image_paths:
    raise ValueError("❌ Nenalezeny žádné obrázky pro trénink.")
print(f"🖼️ Počet nalezených obrázků: {len(image_paths)}")

def load_images():
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((resolution, resolution))
        images.append(img)
    return images

train_images = load_images()
print(f"✅ Nahráno {len(train_images)} obrázků pro trénink.")

# ========== PEFT KONFIGURACE ========== #
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",  # Hack – nebude se používat přímo
)

print("🧠 Přidávám LoRA váhy do UNet...")
unet = get_peft_model(unet, lora_config)

# ========== TRÉNINK ========== #
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
accelerator = Accelerator()

unet, optimizer = accelerator.prepare(unet, optimizer)

unet.train()

print("🚀 Začínám trénink...")
for epoch in range(num_train_epochs):
    total_loss = 0
    for image in tqdm(train_images, desc=f"Epoch {epoch+1}/{num_train_epochs}"):
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.
        image_tensor = image_tensor.to(accelerator.device, dtype=torch.float16)

        with torch.no_grad():
            latents = vae.encode(image_tensor).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=latents.device).long()

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = tokenizer(instance_prompt, return_tensors="pt").input_ids.to(latents.device)

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_images)
    print(f"📉 Epoch {epoch+1} dokončena. Průměrná loss: {avg_loss:.4f}")

# ========== UKLÁDÁNÍ ========== #
print(f"💾 Ukládám model do složky: {output_dir}")
pipeline.save_pretrained(output_dir)
print("✅ Trénink dokončen – jsi pán tučňáků, Lovče Šterbin!")