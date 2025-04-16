import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Config
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs/igloonet_penguin_unet"
IMAGE_DIR = "data/images_png"
EPOCHS = 3
BATCH_SIZE = 1
LR = 1e-5

# Dataset class
class PenguinDataset(Dataset):
    def __init__(self, image_dir, image_size=(512, 512)):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

# Load pretrained pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    safety_checker=None
)

pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# Load UNet and apply LoRA
unet = pipe.unet
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type="UNET"
)
unet = get_peft_model(unet, lora_config)

# Prepare dataset and dataloader
dataset = PenguinDataset(IMAGE_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Set up training
optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
unet.train()

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = batch
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (BATCH_SIZE,), dtype=torch.long)
        noisy_images = pipe.scheduler.add_noise(images, noise, timesteps)

        encoder_hidden_states = torch.randn((BATCH_SIZE, 77, 768))
        noise_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Loss: {avg_loss:.4f}")

# Save the fine-tuned UNet
os.makedirs(OUTPUT_DIR, exist_ok=True)
unet.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Training complete – model saved to {OUTPUT_DIR}!")