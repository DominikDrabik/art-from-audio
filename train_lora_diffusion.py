"""
Trains a Stable Diffusion model fine-tuned with LoRA for audio-conditioned image generation.

This script loads pre-extracted audio embeddings (from CLAP) and corresponding cover images.
It then fine-tunes a U-Net (from a pre-trained Stable Diffusion model) using LoRA (Low-Rank Adaptation)
and an additional linear projection layer to adapt the audio embeddings to the U-Net's
cross-attention dimension. The trained LoRA weights and the projection layer are saved.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers import __version__ as diffusers_version # Import __version__ directly
from peft import LoraConfig
from diffusers.optimization import get_scheduler
from PIL import Image
import os
import glob
from tqdm.auto import tqdm
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any # For type hinting

# --- Configuration ---
MODEL_NAME: str = "runwayml/stable-diffusion-v1-5" # Base model for U-Net, VAE, etc.
AUDIO_EMBEDDINGS_DIR: str = "art-from-audio-dataset/audio_embeddings/" # Directory for .pt audio embeddings
IMAGE_DIR: str = "art-from-audio-dataset/cover_images/" # Directory for .jpg cover images
OUTPUT_DIR: str = "lora_output" # Directory to save trained LoRA weights
IMAGE_EXTENSION: str = ".jpg" # Extension for cover image files

# Training Hyperparameters
LEARNING_RATE: float = 1e-4
LORA_RANK: int = 4 # Rank for LoRA decomposition
BATCH_SIZE: int = 32
NUM_TRAIN_EPOCHS: int = 30
IMAGE_RESOLUTION: int = 512 # Resolution images will be resized to
GRADIENT_ACCUMULATION_STEPS: int = 1
MIXED_PRECISION: str = "bf16" # "no", "fp16", "bf16" (bf16 preferred if available)
SAVE_MODEL_EPOCHS: int = 10 # Save a checkpoint every N epochs

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dataset Class ---
class AudioImageDataset(Dataset):
    """
    A PyTorch Dataset to load audio embeddings and corresponding images.

    Args:
        embeddings_dir (str): Directory containing .pt audio embedding files.
        image_dir (str): Directory containing image files.
        image_extension (str): File extension for the images (e.g., ".jpg").
        resolution (int): The resolution to resize images to.
    """
    def __init__(self, embeddings_dir: str, image_dir: str, image_extension: str, resolution: int = 512):
        self.embeddings_dir = embeddings_dir
        self.image_dir = image_dir
        self.image_extension = image_extension
        self.embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.pt"))) # Sort for reproducibility

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), antialias=True), # Added antialias for better quality
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.embedding_files)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        embedding_path = self.embedding_files[idx]
        base_name = os.path.splitext(os.path.basename(embedding_path))[0]
        image_path = os.path.join(self.image_dir, base_name + self.image_extension)

        try:
            audio_embedding: torch.Tensor = torch.load(embedding_path, map_location="cpu") # Load to CPU to save GPU VRAM
            
            # Expected CLAP embedding shape is [1, 512].
            # This ensures consistency, as clap_feature_extractor.py saves them in this format.
            if audio_embedding.ndim == 1 and audio_embedding.shape[0] == 512:
                audio_embedding = audio_embedding.unsqueeze(0) # Reshape [512] to [1, 512]
            elif audio_embedding.shape == torch.Size([1,1,512]):
                audio_embedding = audio_embedding.squeeze(1) # Reshape [1,1,512] to [1,512]
            elif audio_embedding.shape != torch.Size([1, 512]):
                print(f"Warning: Unexpected embedding shape for {embedding_path}: {audio_embedding.shape}. Expected [1, 512]. Skipping item.")
                # If a more complex case like [N, 512] occurs, it indicates an issue upstream
                # or a change in how embeddings are saved. For now, we strictly expect [1, 512].
                return None

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return {"pixel_values": image_tensor, "condition_embedding": audio_embedding}
        except FileNotFoundError:
            print(f"Error: Image file not found for {embedding_path} at {image_path}. Skipping item.")
            return None
        except Exception as e:
            print(f"Error loading item {embedding_path} or {image_path}: {e}. Skipping item.")
            return None

def collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function for the DataLoader. Filters out None items (due to loading errors)
    and stacks the tensors from the batch.
    """
    # Filter out None items from the batch (e.g., if an image or embedding failed to load)
    batch = [b for b in batch if b is not None]
    if not batch: # If all items in the batch failed
        return None 

    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    condition_embeddings = torch.stack([b["condition_embedding"] for b in batch])
    return {"pixel_values": pixel_values, "condition_embedding": condition_embeddings}


# --- Model Loading and Training Logic ---
def main():
    """Main function to set up models, data, and run the training loop."""
    # 0. Print Diffusers version
    print(f"Using Diffusers version: {diffusers_version}")

    # 1. Device and Mixed Precision Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine the mixed precision strategy
    current_mixed_precision = MIXED_PRECISION.lower()
    if device.type == 'cuda':
        if current_mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("Warning: bf16 configured but not supported. Falling back to fp16.")
            current_mixed_precision = "fp16"
        if current_mixed_precision == "fp16" and not torch.cuda.is_available(): # Should not happen if device.type is cuda
            print("Warning: fp16 configured but CUDA not available. Using 'no' mixed precision.")
            current_mixed_precision = "no"
        
        if current_mixed_precision == "bf16":
            print("Using bf16 mixed precision on CUDA.")
        elif current_mixed_precision == "fp16":
            print("Using fp16 mixed precision on CUDA.")
        else:
            current_mixed_precision = "no" # Default to no if invalid or unsupported for CUDA
            print("Using no mixed precision on CUDA (or invalid configuration).")
            
    elif device.type == 'cpu':
        if current_mixed_precision in ["fp16", "bf16"]:
            print(f"Warning: {current_mixed_precision} not supported on CPU. Using 'no' mixed precision.")
        current_mixed_precision = "no"

    target_dtype_models = torch.float32
    target_dtype_inputs = torch.float32
    scaler: Optional[torch.amp.GradScaler] = None

    if current_mixed_precision == "fp16":
        scaler = torch.amp.GradScaler('cuda')
        target_dtype_models = torch.float16
        target_dtype_inputs = torch.float16
    elif current_mixed_precision == "bf16":
        target_dtype_models = torch.bfloat16
        target_dtype_inputs = torch.bfloat16
    
    # 2. Load pre-trained models (VAE, UNet, Scheduler)
    print("Loading models...")
    # Load VAE, UNet, and Scheduler from the base model
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    
    # Audio Projection Layer Setup
    audio_embedding_dim = 512 # CLAP output dimension
    unet_cross_attention_dim = unet.config.cross_attention_dim # e.g., 768 for SD 1.5
    
    audio_projection: Optional[nn.Linear] = None
    if audio_embedding_dim != unet_cross_attention_dim:
        print(f"Audio embedding dim ({audio_embedding_dim}) != UNet cross-attention dim ({unet_cross_attention_dim}). Adding projection layer.")
        audio_projection = nn.Linear(audio_embedding_dim, unet_cross_attention_dim)
        # Projection layer weights are float32 by default and trained as such.
        # It's moved to device, but its internal computations are float32 unless explicitly cast during forward pass if needed.
        audio_projection.to(device) 
    else:
        print("Audio embedding dim matches UNet cross-attention dim. No projection layer needed.")

    # Freeze VAE and base U-Net parameters (LoRA parameters will be trainable)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Move models to device and set dtype according to mixed precision strategy
    vae.to(device, dtype=target_dtype_models)
    unet.to(device, dtype=target_dtype_models)
    print(f"Models loaded. VAE and UNet on {device} with dtype {target_dtype_models}.")

    # 3. Add LoRA layers to UNet
    print(f"Setting up LoRA with rank {LORA_RANK}...")
    # Define LoRA configuration. These target_modules are common for SD UNet attention blocks.
    # `bias="none"` is often used for LoRA to keep the number of trainable parameters low.
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK, # alpha equals rank
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0", # Attention projection layers
            "proj_in", "proj_out" # Cross-attention input/output projections in ResTransformer blocks
            # Add other layers like "ff.net.0.proj", "ff.net.2" if targeting feed-forward networks
        ],
        bias="none",
    )
    unet.add_adapter(lora_config)
    print("LoRA adapters added to U-Net.")

    # Collect parameters to optimize: LoRA parameters + audio_projection parameters
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if not params_to_optimize:
        print("Warning: No LoRA parameters found in UNet for optimizer. Check LoRA setup.")
    else:
        print(f"Found {len(params_to_optimize)} LoRA parameters in UNet for optimization.")

    if audio_projection is not None:
        params_to_optimize.extend(list(audio_projection.parameters()))
        print(f"Added {len(list(audio_projection.parameters()))} audio projection parameters for optimization.")
    
    if not params_to_optimize:
        print("Error: No parameters to optimize. Training cannot proceed.")
        return

    # 4. Prepare Dataset and DataLoader
    print("Preparing dataset...")
    dataset = AudioImageDataset(
        embeddings_dir=AUDIO_EMBEDDINGS_DIR,
        image_dir=IMAGE_DIR,
        image_extension=IMAGE_EXTENSION,
        resolution=IMAGE_RESOLUTION
    )
    if len(dataset) == 0:
        print("Error: No data found in dataset. Please check data paths and embedding generation.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    print(f"Dataset loaded with {len(dataset)} samples. DataLoader configured.")

    # 5. Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.AdamW(params_to_optimize, lr=LEARNING_RATE)

    num_update_steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION_STEPS
    max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="cosine", 
        optimizer=optimizer,
        num_warmup_steps=0, 
        num_training_steps=max_train_steps,
    )
    print("Optimizer and LR Scheduler configured.")

    # 6. Training Loop
    print("Starting training...")
    global_step = 0
    for epoch in range(NUM_TRAIN_EPOCHS):
        unet.train() # Ensure LoRA layers are in training mode
        if audio_projection is not None: # Ensure projection layer is also in training mode
            audio_projection.train()
            
        progress_bar = tqdm(total=num_update_steps_per_epoch, desc=f"Epoch {epoch+1}/{NUM_TRAIN_EPOCHS}")
        epoch_loss_total = 0.0
        
        for step, batch in enumerate(dataloader):
            if batch is None: # Batch could be None if collate_fn returned None (e.g., all items failed)
                print(f"Skipping empty/failed batch at epoch {epoch+1}, step {step+1}")
                continue

            pixel_values = batch["pixel_values"].to(device, dtype=target_dtype_inputs)
            audio_embeddings_batch = batch["condition_embedding"].to(device) # Shape: [batch, 1, 512]

            # Ensure audio_embeddings_batch is [batch_size, 1, embedding_dim (512)]
            if audio_embeddings_batch.ndim == 2: # If it's [batch, 512]
                audio_embeddings_batch = audio_embeddings_batch.unsqueeze(1)
            
            # The audio_projection layer has float32 parameters.
            # Its computation is done in float32 for precision, then cast to target_dtype_inputs for UNet.
            final_audio_embeddings_for_unet = audio_embeddings_batch.to(target_dtype_inputs)
            if audio_projection is not None:
                # Compute projection in float32, then cast its output for UNet
                projected_embeddings = audio_projection(audio_embeddings_batch.float())
                final_audio_embeddings_for_unet = projected_embeddings.to(dtype=target_dtype_inputs)

            # Prepare latents
            with torch.no_grad():
                # VAE is on target_dtype_models. Input pixel_values are target_dtype_inputs.
                # Autocast for VAE encoding if mixed precision is active.
                with torch.amp.autocast(device_type=device.type, dtype=target_dtype_models if current_mixed_precision != "no" else None, enabled=(current_mixed_precision != "no")):
                    latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor # Apply VAE scaling factor

            # Sample noise and add to latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Forward pass through UNet with autocasting for mixed precision
            with torch.amp.autocast(device_type=device.type, dtype=target_dtype_models if current_mixed_precision != "no" else None, enabled=(current_mixed_precision != "no")):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=final_audio_embeddings_for_unet).sample
                # Calculate loss in float32 for stability, regardless of model/input dtypes
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") 
            
            loss_for_backward = loss / GRADIENT_ACCUMULATION_STEPS
            epoch_loss_total += loss.item() # Accumulate loss for epoch average

            # Backward pass and optimizer step
            if scaler is not None: # For fp16
                scaler.scale(loss_for_backward).backward()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else: # For bf16 or no mixed precision
                loss_for_backward.backward()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

        avg_epoch_loss = epoch_loss_total / len(dataloader)
        progress_bar.set_postfix(avg_loss=avg_epoch_loss, lr=lr_scheduler.get_last_lr()[0])
        progress_bar.close()
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % SAVE_MODEL_EPOCHS == 0 or (epoch + 1) == NUM_TRAIN_EPOCHS:
            if unet.has_adapter: # Check if adapter exists before trying to save
                checkpoint_save_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{(epoch+1)}.safetensors")
                save_trained_weights(unet, audio_projection, checkpoint_save_path)
            else:
                print(f"Skipping checkpoint save at epoch {epoch+1}: No LoRA adapter found on U-Net.")

    print("Training complete.")
    if unet.has_adapter:
        final_checkpoint_path = os.path.join(OUTPUT_DIR, "final_lora_and_projection_weights.safetensors")
        save_trained_weights(unet, audio_projection, final_checkpoint_path)
        print(f"Final trained weights saved to {final_checkpoint_path}")
    else:
        print("Skipping final save: No LoRA adapter found on U-Net.")


# Helper function to save trained weights (LoRA from UNet and audio projection layer)
def save_trained_weights(unet_model: UNet2DConditionModel, 
                         audio_projection_model: Optional[nn.Linear], 
                         save_path: str):
    """
    Saves the LoRA weights from the UNet and the weights of the audio projection layer 
    into a single .safetensors file.

    Args:
        unet_model: The UNet model containing LoRA adapters.
        audio_projection_model: The audio projection layer (if used).
        save_path: Path to save the .safetensors file.
    """
    weights_to_save = {}
    # 1. Save LoRA weights from UNet, prefixing keys
    has_unet_lora_weights = False
    if hasattr(unet_model, 'peft_config'): # Check if it's a PeftModel
        for name, param in unet_model.named_parameters():
            if "lora" in name and param.requires_grad:
                weights_to_save[f"unet_lora.{name}"] = param.data.cpu().clone()
                has_unet_lora_weights = True    
        if not has_unet_lora_weights:
            print("Warning: UNet seems to be a PeftModel but no LoRA parameters found to save.")
    else:
        print("Warning: UNet does not have PEFT config, cannot save LoRA weights directly this way. Ensure add_adapter was called.")

    # 2. Save audio_projection_model weights (if it exists), prefixing keys
    if audio_projection_model is not None:
        audio_projection_state_dict = audio_projection_model.state_dict()
        for k, v in audio_projection_state_dict.items():
            weights_to_save[f"audio_projection.{k}"] = v.cpu().clone()
        print(f"Audio projection weights prepared for saving.")
    else:
        print("Info: No audio projection model to save.")
    
    if weights_to_save:
        try:
            from safetensors.torch import save_file
            save_file(weights_to_save, save_path)
            print(f"Saved trained weights to {save_path} using safetensors.")
        except ImportError:
            print("Warning: `safetensors` library not found. Saving with torch.save() as .bin instead.")
            bin_save_path = save_path.replace(".safetensors", ".bin")
            torch.save(weights_to_save, bin_save_path)
            print(f"Saved trained weights to {bin_save_path}.")
        except Exception as e:
            print(f"Error saving weights to {save_path}: {e}")
    else:
        print("Warning: Nothing to save. No UNet LoRA or Audio Projection weights were collected.")


if __name__ == "__main__":
    main()
