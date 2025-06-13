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

# --- Configuration (Same as original train_lora_diffusion.py) ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
AUDIO_EMBEDDINGS_DIR = "art-from-audio-dataset/audio_embeddings/"
IMAGE_DIR = "art-from-audio-dataset/cover_images/"
OUTPUT_DIR = "lora_mlp_output" # Changed output directory for the new model
IMAGE_EXTENSION = ".jpg"

LEARNING_RATE = 1e-4
LORA_RANK = 4
BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 50
IMAGE_RESOLUTION = 512
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "bf16" # "no", "fp16", "bf16"
SAVE_MODEL_EPOCHS = 10

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
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding_path = self.embedding_files[idx]
        base_name = os.path.splitext(os.path.basename(embedding_path))[0]
        image_path = os.path.join(self.image_dir, base_name + self.image_extension)
        try:
            # Load audio embedding, ensure it's [1, 512]
            audio_embedding = torch.load(embedding_path, map_location="cpu")
            if audio_embedding.ndim == 1 and audio_embedding.shape[0] == 512: # Shape [512]
                audio_embedding = audio_embedding.unsqueeze(0) # -> [1, 512]
            elif audio_embedding.shape == torch.Size([1, 1, 512]): # Shape [1, 1, 512] (from some CLAP versions)
                 audio_embedding = audio_embedding.squeeze(1) # -> [1, 512]
            elif audio_embedding.shape != torch.Size([1, 512]):
                print(f"Warning: Unexpected shape {audio_embedding.shape} for {embedding_path}. Expected [1, 512] or [512] or [1,1,512]. Skipping.")
                return None # Skip this item

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return {"pixel_values": image_tensor, "condition_embedding": audio_embedding}
        except Exception as e:
            print(f"Error loading item {embedding_path} or {image_path}: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None] # Filter out None items
    if not batch:
        return None # Or raise an error, or return an empty dict
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    # Squeeze out the batch dim from individual embeddings before stacking
    condition_embeddings = torch.stack([b["condition_embedding"].squeeze(0) for b in batch])
    return {"pixel_values": pixel_values, "condition_embedding": condition_embeddings}


def save_trained_weights(unet_model, audio_projection_model, save_path):
    weights_to_save = {}
    # Save LoRA weights from UNet
    for name, param in unet_model.named_parameters():
        if "lora" in name and param.requires_grad:
            weights_to_save[f"unet_lora.{name}"] = param.data.cpu().clone()

    # Save audio projection model weights
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

    # Mixed precision setup
    current_mixed_precision = MIXED_PRECISION.lower()
    if device.type == 'cuda':
        if current_mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("BF16 not supported on this GPU, falling back to FP16.")
            current_mixed_precision = "fp16"
        if current_mixed_precision not in ["no", "fp16", "bf16"]:
            print(f"Invalid mixed_precision '{MIXED_PRECISION}', falling back to 'no'.")
            current_mixed_precision = "no"
    else: # CPU
        if current_mixed_precision != "no":
            print("Mixed precision is only supported on CUDA, setting to 'no'.")
        current_mixed_precision = "no"

    target_dtype_models = torch.float32
    target_dtype_inputs = torch.float32
    scaler = None # For fp16

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
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device, dtype=target_dtype_models)
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device, dtype=target_dtype_models)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    # --- Modified Audio Projection Layer ---
    audio_embedding_dim = 512 # Assuming CLAP embeddings are 512-dim
    unet_cross_attention_dim = unet.config.cross_attention_dim
    
    # Define the MLP for audio projection
    # Intermediate dimension can be tuned
    hidden_projection_dim = (audio_embedding_dim + unet_cross_attention_dim) // 2 
    audio_projection = nn.Sequential(
        nn.Linear(audio_embedding_dim, hidden_projection_dim),
        nn.ReLU(),
        nn.Linear(hidden_projection_dim, unet_cross_attention_dim)
    ).to(device, dtype=target_dtype_models) # Ensure dtype matches models

    print(f"Using MLP audio projection: {audio_embedding_dim} -> {hidden_projection_dim} -> {unet_cross_attention_dim}")
    num_audio_proj_params = sum(p.numel() for p in audio_projection.parameters() if p.requires_grad)
    print(f"Audio projection MLP trainable parameters: {num_audio_proj_params}")
    # --- End of Modified Audio Projection Layer ---

    vae.requires_grad_(False)
    unet.requires_grad_(False) # Freeze UNet, only LoRA layers will be trainable

    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK, # Often set to r or 2*r
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"], # Common targets for attention blocks
        bias="none", # or "all" or "lora_only"
    )
    unet.add_adapter(lora_config)
    num_lora_params = sum(p.numel() for n, p in unet.named_parameters() if "lora" in n and p.requires_grad)
    print(f"LoRA setup complete. Trainable LoRA parameters in UNet: {num_lora_params}")


    # Parameters to optimize: LoRA parameters from UNet and all parameters from audio_projection
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if audio_projection is not None: # Should always be true now with MLP
        params_to_optimize.extend(list(audio_projection.parameters()))

    if not params_to_optimize:
        print("No parameters to optimize. Check LoRA setup and audio projection.")
        return

    print("Preparing dataset...")
    dataset = AudioImageDataset(AUDIO_EMBEDDINGS_DIR, IMAGE_DIR, IMAGE_EXTENSION, IMAGE_RESOLUTION)
    if len(dataset) == 0:
        print(f"No data found in {AUDIO_EMBEDDINGS_DIR} and {IMAGE_DIR}. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True) # num_workers can be tuned

    optimizer = torch.optim.AdamW(params_to_optimize, lr=LEARNING_RATE)

    # Calculate total training steps
    # Note: len(dataloader) gives number of batches.
    # If dataset size is not a multiple of BATCH_SIZE, last batch might be smaller.
    # For simplicity, we assume full batches or that dataloader handles partials correctly.
    steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION_STEPS
    max_steps = NUM_TRAIN_EPOCHS * steps_per_epoch

    lr_scheduler = get_scheduler(
        "cosine", # common choice
        optimizer=optimizer,
        num_warmup_steps=0, # Can add warmup steps, e.g., int(0.05 * max_steps)
        num_training_steps=max_steps
    )

    print(f"Starting training for {NUM_TRAIN_EPOCHS} epochs ({max_steps} steps). Saving every {SAVE_MODEL_EPOCHS} epochs.")
    global_step = 0

    for epoch in range(NUM_TRAIN_EPOCHS):
        unet.train()
        if audio_projection: # Should always be true
            audio_projection.train()

        progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{NUM_TRAIN_EPOCHS}")
        epoch_loss_total = 0.0

        for step, batch in enumerate(dataloader):
            if batch is None: # From collate_fn if all items in batch failed
                print(f"Skipping empty batch at epoch {epoch+1}, step {step+1}")
                if steps_per_epoch == len(dataloader): # if not using grad accum
                    progress_bar.update(1)
                continue

            pixel_values = batch["pixel_values"].to(device, dtype=target_dtype_inputs)
            audio_embeddings = batch["condition_embedding"].to(device) # Expected shape [B, 512] from collate_fn

            # Reshape audio_embeddings if necessary for UNet (expects [B, Seq_Len, Dim])
            # Here, Seq_Len is 1 for a single global audio embedding per image
            if audio_embeddings.ndim == 2: # [B, 512]
                audio_embeddings = audio_embeddings.unsqueeze(1) # -> [B, 1, 512]

            # Project audio embeddings
            final_embeddings = audio_embeddings.to(target_dtype_inputs) # Default if no projection
            if audio_projection:
                # audio_projection expects [B, Dim] or [B, Seq, Dim] if it handles seq internally.
                # Our MLP expects [B, Dim_in] or [B*Seq, Dim_in] then reshape.
                # Simpler: apply to [B, 1, 512], MLP will apply to last dim.
                # Or squeeze, project, unsqueeze:
                current_embeddings_for_proj = audio_embeddings.squeeze(1) # -> [B, 512]
                projected = audio_projection(current_embeddings_for_proj.to(audio_projection[0].weight.dtype)) # Ensure dtype match for MLP input
                final_embeddings = projected.unsqueeze(1).to(dtype=target_dtype_inputs) # -> [B, 1, Proj_Dim]

            # VAE: Encode images to latent space
            with torch.no_grad(): # VAE is frozen
                # Autocast for VAE if using mixed precision
                with torch.amp.autocast(device_type=device.type, dtype=target_dtype_models if current_mixed_precision != "no" else None, enabled=(current_mixed_precision != "no")):
                    latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet forward pass (predict noise)
            # Autocast for UNet
            with torch.amp.autocast(device_type=device.type, dtype=target_dtype_models if current_mixed_precision != "no" else None, enabled=(current_mixed_precision != "no")):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=final_embeddings).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") # Ensure loss calculation in fp32

            # Backpropagation
            loss_for_backward = loss / GRADIENT_ACCUMULATION_STEPS
            epoch_loss_total += loss.item()

            if scaler: # FP16
                scaler.scale(loss_for_backward).backward()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0) # Clip gradients
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step +=1
            else: # FP32 or BF16
                loss_for_backward.backward()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0) # Clip gradients
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
    final_path = os.path.join(OUTPUT_DIR, "final_lora_mlp_projection_weights.safetensors")
    print(f"Saving final model weights to {final_path}...")
    save_trained_weights(unet, audio_projection, final_path)

if __name__ == "__main__":
    main()