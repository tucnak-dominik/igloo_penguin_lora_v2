import os
import torch
from PIL import Image
from datasets import Dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Cesty
IMAGE_DIR = "data/images_augmented"
CAPTIONS_PATH = "data/captions/captions.txt"
OUTPUT_DIR = "models/lora_output"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

# Konfigurace
IMAGE_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Načtení captionů
with open(CAPTIONS_PATH, "r") as f:
    lines = f.readlines()
    entries = [line.strip().split("|") for line in lines]

data = [{"image_path": os.path.join(IMAGE_DIR, img), "caption": caption} for img, caption in entries]

dataset = Dataset.from_list(data)

# Tokenizace
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pipeline pro načtení a použití UNetu
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
unet = pipe.unet

# LoRA konfigurace
lora_config = LoraConfig(r=4, lora_alpha=16, target_modules=["to_k", "to_q"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
peft_unet = get_peft_model(unet, lora_config)

# Dataset a transformace
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    image = transform(image)
    caption = example["caption"]
    input_ids = tokenizer(caption, padding="max_length", max_length=77, return_tensors="pt").input_ids[0]
    return {"pixel_values": image, "input_ids": input_ids}

# Předzpracování dat
processed = dataset.map(preprocess)
dataloader = DataLoader(processed, batch_size=BATCH_SIZE, shuffle=True)

# Tréninková smyčka
optimizer = torch.optim.AdamW(peft_unet.parameters(), lr=LR)
peft_unet.train()

print("\n🚀 Starting LoRA training on GCP GPU...")

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = batch["pixel_values"].to(DEVICE).unsqueeze(1)
        input_ids = batch["input_ids"].to(DEVICE)

        noise = torch.randn_like(images).to(DEVICE)
        timesteps = torch.randint(0, 1000, (images.size(0),), device=DEVICE).long()
        noisy_images = pipe.scheduler.add_noise(images, noise, timesteps)
        encoder_hidden_states = pipe.text_encoder(input_ids)[0]

        noise_pred = peft_unet(noisy_images, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1}: loss = {total_loss / len(dataloader):.4f}")

# Uložení modelu
peft_unet.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Model uložen do složky {OUTPUT_DIR}")
