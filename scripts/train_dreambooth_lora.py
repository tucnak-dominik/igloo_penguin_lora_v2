import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# 📦 Konfigurace
model_id = "runwayml/stable-diffusion-v1-5"
data_dir = "data/images_augmented"
output_dir = "outputs/igloonet_penguin_lora"
instance_prompt = "a photo of igloo penguin"
image_size = 512
learning_rate = 1e-4
batch_size = 1
epochs = 10

print("📦 Načítám model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

tokenizer = CLIPTokenizer.from_pretrained(model_id)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to("cuda", dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda", dtype=torch.float16)

# ⚙️ LoRA konfigurace
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none",
    task_type="CAUSAL_LM"
)

unet = get_peft_model(unet, lora_config)

# 📁 Dataset
def load_images(image_folder):
    image_paths = list(Path(image_folder).glob("*.png"))
    print(f"📁 Načítám obrázky ze složky: {image_folder}")
    print(f"🖼️ Počet nalezených obrázků: {len(image_paths)}")
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(preprocess(img))
    return torch.stack(images)

images = load_images(data_dir).to("cuda")

# 🧠 Trénink
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

print(f"🧠 Spouštím trénink na {len(images)} obrázcích...")
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        noise = torch.randn_like(batch).to("cuda")
        noisy_images = batch + noise * 0.1
        optimizer.zero_grad()
        output = unet(noisy_images, timestep=torch.tensor([10] * len(batch)).to("cuda"), encoder_hidden_states=None)
        loss = ((output.sample - batch) ** 2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"🔁 Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

# 💾 Uložení modelu
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"✅ Trénink dokončen – výstupní složka: {output_dir}")