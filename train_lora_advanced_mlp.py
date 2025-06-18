"""
Train Advanced MLP Audio-to-Image Model with LoRA on Audio-Image Pairs

This script trains a Stable Diffusion UNet with LoRA adapters and an advanced MLP projection layer
to map CLAP audio embeddings to the image latent space. It uses paired audio embeddings and images.

Key features:
- Loads audio embeddings and corresponding images from specified directories.
- Uses an advanced MLP (with LayerNorm and ReLU) to project audio embeddings to the UNet cross-attention space.
- Adds LoRA adapters to the UNet for efficient fine-tuning.
- Trains using a DDPMScheduler and MSE loss on the predicted noise.
- Supports mixed precision training (fp16, bf16, or no).
- Periodically saves model checkpoints and final weights in safetensors format.

Arguments and configuration are set at the top of the script.

Usage:
    python train_lora_advanced_mlp.py

You may need to adjust the configuration section for your dataset and training preferences.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers import __version__ as diffusers_version
from peft import LoraConfig
from diffusers.optimization import get_scheduler
from PIL import Image
import os
import glob
from tqdm.auto import tqdm
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any

# --- Configuration ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
AUDIO_EMBEDDINGS_DIR = "art-from-audio-dataset/audio_embeddings/"
IMAGE_DIR = "art-from-audio-dataset/cover_images/"
OUTPUT_DIR = "lora_advanced_mlp_output" # New output directory
IMAGE_EXTENSION = ".jpg"

LEARNING_RATE = 1e-4
LORA_RANK = 4 # You might consider increasing this later if results are still not diverse enough
BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 50 # Keeping epochs, adjust as needed
IMAGE_RESOLUTION = 512
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "bf16"
SAVE_MODEL_EPOCHS = 10
WEIGHT_DECAY = 1e-2 # Added weight decay

os.makedirs(OUTPUT_DIR, exist_ok=True)

class AudioImageDataset(Dataset):
    def __init__(self, embeddings_dir: str, image_dir: str, image_extension: str, resolution: int = 512):
        self.embeddings_dir = embeddings_dir
        self.image_dir = image_dir
        self.image_extension = image_extension
        self.embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.pt")))
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding_path = self.embedding_files[idx]
        base_name = os.path.splitext(os.path.basename(embedding_path))[0]
        image_path = os.path.join(self.image_dir, base_name + self.image_extension)
        try:
            audio_embedding = torch.load(embedding_path, map_location="cpu")
            if audio_embedding.ndim == 1 and audio_embedding.shape[0] == 512:
                audio_embedding = audio_embedding.unsqueeze(0)
            elif audio_embedding.shape == torch.Size([1, 1, 512]):
                audio_embedding = audio_embedding.squeeze(1)
            elif audio_embedding.shape != torch.Size([1, 512]):
                print(f"Warning: Unexpected shape {audio_embedding.shape} for {embedding_path}. Skipping.")
                return None
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return {"pixel_values": image_tensor, "condition_embedding": audio_embedding}
        except Exception as e:
            print(f"Error loading item {embedding_path} or {image_path}: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    condition_embeddings = torch.stack([b["condition_embedding"].squeeze(0) for b in batch])
    return {"pixel_values": pixel_values, "condition_embedding": condition_embeddings}

def save_trained_weights(unet_model, audio_projection_model, save_path):
    weights_to_save = {}
    for name, param in unet_model.named_parameters():
        if "lora" in name and param.requires_grad:
            weights_to_save[f"unet_lora.{name}"] = param.data.cpu().clone()
    if audio_projection_model is not None:
        for k, v in audio_projection_model.state_dict().items():
            weights_to_save[f"audio_projection.{k}"] = v.cpu().clone()
    try:
        from safetensors.torch import save_file
        save_file(weights_to_save, save_path)
        print(f"Saved weights to {save_path}")
    except ImportError:
        print("safetensors not installed, falling back to .bin format")
        fallback_path = save_path.replace(".safetensors", ".bin")
        torch.save(weights_to_save, fallback_path)
        print(f"Saved weights to {fallback_path}")

def main():
    print(f"Using Diffusers version: {diffusers_version}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_mixed_precision = MIXED_PRECISION.lower()
    if device.type == 'cuda':
        if current_mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("BF16 not supported on this GPU, falling back to FP16.")
            current_mixed_precision = "fp16"
        if current_mixed_precision not in ["no", "fp16", "bf16"]:
            print(f"Invalid mixed_precision '{MIXED_PRECISION}', falling back to 'no'.")
            current_mixed_precision = "no"
    else:
        if current_mixed_precision != "no":
            print("Mixed precision is only supported on CUDA, setting to 'no'.")
        current_mixed_precision = "no"

    target_dtype_models = torch.float32
    target_dtype_inputs = torch.float32
    scaler = None

    if current_mixed_precision == "fp16":
        scaler = torch.amp.GradScaler('cuda')
        target_dtype_models = torch.float16
        target_dtype_inputs = torch.float16
        print("Using FP16 mixed precision.")
    elif current_mixed_precision == "bf16":
        target_dtype_models = torch.bfloat16
        target_dtype_inputs = torch.bfloat16
        print("Using BF16 mixed precision.")
    else:
        print("Not using mixed precision.")

    print("Loading models...")
    # Optionally load a different VAE here if stripes persist
    # vae_model_name = "stabilityai/sd-vae-ft-mse"
    # try:
    #     vae = AutoencoderKL.from_pretrained(vae_model_name).to(device, dtype=target_dtype_models)
    #     print(f"Loaded custom VAE: {vae_model_name}")
    # except Exception as e:
    #     print(f"Could not load custom VAE {vae_model_name}, falling back. Error: {e}")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device, dtype=target_dtype_models)
    
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device, dtype=target_dtype_models)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    audio_embedding_dim = 512
    unet_cross_attention_dim = unet.config.cross_attention_dim
    
    # --- Advanced Audio Projection Layer ---
    hidden_projection_dim = 768 # Increased hidden dimension (was (512+768)//2 = 640)
    audio_projection = nn.Sequential(
        nn.Linear(audio_embedding_dim, hidden_projection_dim),
        nn.LayerNorm(hidden_projection_dim), # Added LayerNorm
        nn.ReLU(), # Or nn.GELU()
        nn.Linear(hidden_projection_dim, unet_cross_attention_dim)
    ).to(device, dtype=target_dtype_models)

    print(f"Using Advanced MLP audio projection with LayerNorm: {audio_embedding_dim} -> LN -> ReLU -> {hidden_projection_dim} -> {unet_cross_attention_dim}")
    num_audio_proj_params = sum(p.numel() for p in audio_projection.parameters() if p.requires_grad)
    print(f"Audio projection MLP trainable parameters: {num_audio_proj_params}")
    # --- End of Advanced Audio Projection Layer ---

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
        bias="none",
    )
    unet.add_adapter(lora_config)
    num_lora_params = sum(p.numel() for n, p in unet.named_parameters() if "lora" in n and p.requires_grad)
    print(f"LoRA setup complete. Trainable LoRA parameters in UNet: {num_lora_params}")

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if audio_projection is not None:
        params_to_optimize.extend(list(audio_projection.parameters()))

    if not params_to_optimize:
        print("No parameters to optimize.")
        return

    print("Preparing dataset...")
    dataset = AudioImageDataset(AUDIO_EMBEDDINGS_DIR, IMAGE_DIR, IMAGE_EXTENSION, IMAGE_RESOLUTION)
    if len(dataset) == 0:
        print(f"No data found. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(
        params_to_optimize, 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY # Added weight decay
    )
    print(f"Using AdamW optimizer with LR: {LEARNING_RATE} and Weight Decay: {WEIGHT_DECAY}")

    steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION_STEPS
    max_steps = NUM_TRAIN_EPOCHS * steps_per_epoch
    
    # Optional: Add warmup steps
    num_warmup_steps_val = int(0.05 * max_steps) # 5% warmup
    print(f"Scheduler: cosine with {num_warmup_steps_val} warmup steps over {max_steps} total steps.")

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_val, 
        num_training_steps=max_steps
    )

    print(f"Starting training for {NUM_TRAIN_EPOCHS} epochs. Saving every {SAVE_MODEL_EPOCHS} epochs.")
    global_step = 0

    for epoch in range(NUM_TRAIN_EPOCHS):
        unet.train()
        audio_projection.train()
        progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{NUM_TRAIN_EPOCHS}")
        epoch_loss_total = 0.0

        for step, batch in enumerate(dataloader):
            if batch is None:
                if steps_per_epoch == len(dataloader): progress_bar.update(1)
                continue

            pixel_values = batch["pixel_values"].to(device, dtype=target_dtype_inputs)
            audio_embeddings = batch["condition_embedding"].to(device) 

            if audio_embeddings.ndim == 2:
                audio_embeddings = audio_embeddings.unsqueeze(1)

            current_embeddings_for_proj = audio_embeddings.squeeze(1)
            projected = audio_projection(current_embeddings_for_proj.to(audio_projection[0].weight.dtype))
            final_embeddings = projected.unsqueeze(1).to(dtype=target_dtype_inputs)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=target_dtype_models if current_mixed_precision != "no" else None, enabled=(current_mixed_precision != "no")):
                    latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast(device_type=device.type, dtype=target_dtype_models if current_mixed_precision != "no" else None, enabled=(current_mixed_precision != "no")):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=final_embeddings).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            loss_for_backward = loss / GRADIENT_ACCUMULATION_STEPS
            epoch_loss_total += loss.item()

            if scaler:
                scaler.scale(loss_for_backward).backward()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step +=1
            else:
                loss_for_backward.backward()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step +=1
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

        avg_epoch_loss = epoch_loss_total / len(dataloader) if len(dataloader) > 0 else 0
        progress_bar.set_postfix(avg_loss=avg_epoch_loss, lr=lr_scheduler.get_last_lr()[0])
        progress_bar.close()
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % SAVE_MODEL_EPOCHS == 0 or (epoch + 1) == NUM_TRAIN_EPOCHS:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.safetensors")
            print(f"Saving checkpoint to {checkpoint_path}...")
            save_trained_weights(unet, audio_projection, checkpoint_path)

    print("Training complete.")
    final_path = os.path.join(OUTPUT_DIR, "final_lora_advanced_mlp_weights.safetensors") # New final weights name
    print(f"Saving final model weights to {final_path}...")
    save_trained_weights(unet, audio_projection, final_path)

if __name__ == "__main__":
    main()