import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
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
tokenizer: CLIPTokenizer = pipe.tokenizer
text_encoder: CLIPTextModel = pipe.text_encoder
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

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

# ========== Trénink (reálný) ==========
print("🚀 Spouštím trénink...")

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"🧪 Epoch {epoch + 1}/{num_epochs}")
    for i, (img_tensor, path) in enumerate(zip(train_images, image_paths)):
        optimizer.zero_grad()
        caption = captions_df[captions_df["file_name"] == path.name]["caption"].values[0]

        # Tokenizace textu a extrakce hidden states
        inputs = tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to("cuda")
        encoder_hidden_states = text_encoder(**inputs).last_hidden_state

        # Získání latentního kódu z obrázku přes VAE
        with torch.no_grad():
            latents = vae.encode(img_tensor).latent_dist.sample() * 0.18215

        # Přidání náhodného šumu
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda").long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predikce šumu přes UNet
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        # Výpočet loss
        loss = F.mse_loss(noise_pred.float(), noise.float())

        loss.backward()
        optimizer.step()

        if (i + 1) % 25 == 0:
            print(f"  🔄 Step {i + 1}/{len(train_images)} - Loss: {loss.item():.4f}")

# ========== Uložení ==========
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"✅ Trénink dokončen. Model uložen do: {output_dir}")