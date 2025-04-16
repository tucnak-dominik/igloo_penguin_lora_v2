import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse


class CustomDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, resolution):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.resolution = resolution

        df = pd.read_csv(caption_file)
        self.items = df.to_dict(orient="records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = os.path.join(self.image_dir, item["file_name"])
        caption = item["caption"].strip()

        if caption == "":
            caption = "a penguin"

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        tokens = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    lora_config = LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    unet = get_peft_model(unet, lora_config).to(device)

    dataset = CustomDataset(
        image_dir=args.instance_data_dir,
        caption_file=args.caption_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    unet.train()
    for epoch in range(args.max_train_steps // len(dataloader)):
        print(f"🧪 Epoch {epoch + 1}/{args.max_train_steps // len(dataloader)}")
        for step, batch in enumerate(tqdm(dataloader, desc="🔄 Training")):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )[0]

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            if torch.isnan(loss):
                print("⚠️ Loss je NaN, přeskočeno")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"   🔄 Step {step}/{len(dataloader)} - Loss: {loss.item():.4f}")

    unet.save_pretrained(args.output_dir)
    print("✅ Trénink dokončen a model uložen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, default="a penguin")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--validation_prompt", type=str, default="a penguin in igloonet style")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--caption_file", type=str, required=True)
    args = parser.parse_args()

    train(args)
