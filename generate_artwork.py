"""
Generates album artwork conditioned on an audio file using a pre-trained
LoRA-fine-tuned Stable Diffusion model and CLAP audio embeddings.

This script performs the following steps:
1. Loads a CLAP model to extract an embedding from the input audio file.
2. Loads a base Stable Diffusion model (VAE, UNet, text_encoder, tokenizer, scheduler).
3. Sets up an audio projection layer (if used during training).
4. Adds LoRA adapters to the UNet (matching the training configuration).
5. Loads the trained LoRA weights (for UNet) and audio projection layer weights 
   from a specified checkpoint file.
6. Prepares the audio embedding (projects and reshapes) to be used as conditional input.
7. If classifier-free guidance is used, generates unconditional embeddings.
8. Runs the Stable Diffusion pipeline to generate an image.
9. Saves the generated image.
"""
import torch
import torch.nn as nn
from PIL import Image
from transformers import ClapModel, ClapProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import LoraConfig # For re-creating LoRA structure before loading weights
from safetensors.torch import load_file # For loading custom checkpoint
import os
import librosa
import argparse
from typing import Optional, Any # For type hinting

# --- Configuration (Defaults for argparse) ---
BASE_MODEL_NAME_DEFAULT: str = "runwayml/stable-diffusion-v1-5"
CLAP_MODEL_NAME_DEFAULT: str = "laion/larger_clap_music_and_speech"
CHECKPOINT_PATH_DEFAULT: str = "lora_output/final_lora_and_projection_weights.safetensors"
OUTPUT_IMAGE_PATH_DEFAULT: str = "generated_artwork.png"
LORA_RANK_DEFAULT: int = 4

# --- Constants ---
AUDIO_EMBEDDING_DIM: int = 512 # CLAP output dimension, must match training
UNET_CROSS_ATTENTION_DIM: int = 768 # SD 1.5 UNet cross-attention dimension, must match training
CLAP_TARGET_SAMPLE_RATE: int = 48000 # CLAP model's expected sample rate

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Function for CLAP Embedding ---
def get_clap_audio_embedding(audio_path: str,
                             clap_model: ClapModel,
                             clap_processor: ClapProcessor,
                             device: str, 
                             target_sr: int = CLAP_TARGET_SAMPLE_RATE) -> torch.Tensor:
    """
    Loads an audio file, extracts, and returns its CLAP embedding.

    Args:
        audio_path: Path to the audio file.
        clap_model: The loaded CLAP model.
        clap_processor: The loaded CLAP processor.
        device: The device to run CLAP model on.
        target_sr: The sample rate to resample the audio to.

    Returns:
        A torch.Tensor representing the audio embedding (shape [1, AUDIO_EMBEDDING_DIM]).

    Raises:
        FileNotFoundError: If the audio file is not found.
        RuntimeError: If there's an error loading the audio file.
        ValueError: If the extracted embedding has an unexpected shape.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        audio_waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {audio_path}: {e}")

    inputs = clap_processor(audios=[audio_waveform], return_tensors="pt", sampling_rate=target_sr, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): # Ensure no gradients are computed for CLAP model
        audio_embeddings = clap_model.get_audio_features(**inputs)

    # Ensure embedding is [1, AUDIO_EMBEDDING_DIM]
    if audio_embeddings.ndim == 1 and audio_embeddings.shape[0] == AUDIO_EMBEDDING_DIM:
        audio_embeddings = audio_embeddings.unsqueeze(0)
    
    if audio_embeddings.shape != torch.Size([1, AUDIO_EMBEDDING_DIM]):
        raise ValueError(f"Unexpected CLAP embedding shape: {audio_embeddings.shape}. Expected [1, {AUDIO_EMBEDDING_DIM}]")
        
    return audio_embeddings

# --- Main Inference Logic ---
def main(args: argparse.Namespace):
    """Orchestrates the full inference pipeline from audio to generated image."""
    print(f"Using device: {DEVICE}")
    
    model_dtype = torch.float32
    if DEVICE == "cuda" and args.mixed_precision_inference:
        if args.mixed_precision_inference.lower() == "bf16" and torch.cuda.is_bf16_supported():
            print("Using bfloat16 for inference models on CUDA.")
            model_dtype = torch.bfloat16
        elif args.mixed_precision_inference.lower() == "fp16":
            print("Using float16 for inference models on CUDA.")
            model_dtype = torch.float16
        else:
            print(f"Mixed precision '{args.mixed_precision_inference}' not supported or invalid for CUDA. Using float32.")

    # 1. Load CLAP Model
    print(f"Loading CLAP model: {args.clap_model_name}...")
    try:
        clap_model = ClapModel.from_pretrained(args.clap_model_name).to(DEVICE, dtype=model_dtype).eval()
        clap_processor = ClapProcessor.from_pretrained(args.clap_model_name)
    except Exception as e:
        print(f"Error loading CLAP model: {e}"); return

    # 2. Load Diffusion Model Components
    print(f"Loading Diffusion model components from {args.base_model_name}...")
    try:
        vae = AutoencoderKL.from_pretrained(args.base_model_name, subfolder="vae").to(DEVICE, dtype=model_dtype).eval()
        unet = UNet2DConditionModel.from_pretrained(args.base_model_name, subfolder="unet").to(DEVICE, dtype=model_dtype).eval()
        text_encoder = CLIPTextModel.from_pretrained(args.base_model_name, subfolder="text_encoder").to(DEVICE, dtype=model_dtype).eval()
        tokenizer = CLIPTokenizer.from_pretrained(args.base_model_name, subfolder="tokenizer")
        scheduler = DDPMScheduler.from_pretrained(args.base_model_name, subfolder="scheduler")
    except Exception as e:
        print(f"Error loading base diffusion model components: {e}"); return

    # 3. Setup Audio Projection Layer (weights will be loaded later)
    audio_projection: Optional[nn.Linear] = None
    if AUDIO_EMBEDDING_DIM != UNET_CROSS_ATTENTION_DIM:
        print(f"Setting up Audio Projection Layer ({AUDIO_EMBEDDING_DIM} -> {UNET_CROSS_ATTENTION_DIM})...")
        audio_projection = nn.Linear(AUDIO_EMBEDDING_DIM, UNET_CROSS_ATTENTION_DIM)
        # audio_projection will be moved to device and set to .eval() after weight loading
    
    # 4. Add LoRA Adapters to UNet (structure only, weights loaded from checkpoint)
    print(f"Adding LoRA adapters (rank={args.lora_rank}) to UNet...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
        bias="none",
    )
    unet.add_adapter(lora_config) # UNet is already in .eval() mode
    print("LoRA adapter structure added to U-Net.")

    # 5. Load Trained Weights from Checkpoint
    print(f"Loading trained weights from {args.checkpoint_path}...")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}"); return
    try:
        state_dict = load_file(args.checkpoint_path, device="cpu") # Load all to CPU first
    except Exception as e:
        print(f"Error loading checkpoint file '{args.checkpoint_path}' with safetensors: {e}"); return

    # Separate and load weights for UNet LoRA and Audio Projection
    unet_lora_state_dict = {}
    audio_projection_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("unet_lora."):
            unet_lora_state_dict[key.replace("unet_lora.", "", 1)] = value
        elif key.startswith("audio_projection."):
            audio_projection_state_dict[key.replace("audio_projection.", "", 1)] = value

    if unet_lora_state_dict:
        print("Loading UNet LoRA weights...")
        missing, unexpected = unet.load_state_dict(unet_lora_state_dict, strict=False)
        
        missing_lora_keys_list = [k for k in missing if "lora" in k]
        if missing_lora_keys_list: 
            print(f"Warning: Missing LoRA-specific keys in UNet: {missing_lora_keys_list}")
        
        unexpected_lora_keys_list = [k for k in unexpected if "lora" in k]
        if unexpected_lora_keys_list: 
            print(f"Warning: Unexpected LoRA-specific keys in UNet: {unexpected_lora_keys_list}")
        print("UNet LoRA weights loaded.")
    else:
        print("Warning: No 'unet_lora' weights found in checkpoint. UNet will use base weights + randomly initialized LoRA adapters.")

    if audio_projection is not None and audio_projection_state_dict:
        print("Loading Audio Projection weights...")
        try:
            audio_projection.load_state_dict(audio_projection_state_dict)
            print("Audio Projection weights loaded.")
        except Exception as e:
            print(f"Error loading audio_projection state_dict: {e}")
    elif audio_projection is not None and not audio_projection_state_dict:
        print("Warning: Audio projection layer exists but no 'audio_projection' weights found in checkpoint. Using random init.")
    
    if audio_projection is not None:
        audio_projection.to(DEVICE, dtype=torch.float32).eval() # Projection layer typically stays float32 for stability
    print("All models and layers configured and on device in evaluation mode.")

    # 6. Create Stable Diffusion Pipeline with our modified components
    print("Creating Stable Diffusion Pipeline...")
    pipeline = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
        scheduler=scheduler, safety_checker=None, feature_extractor=None, 
        requires_safety_checker=False,
    )
    # Pipeline is not explicitly moved to device or dtype here, as its components already are.
    # Operations within pipeline context will use component dtypes.
    print("Stable Diffusion Pipeline created.")

    # 7. Load and Process Audio File to get CLAP embedding
    print(f"Processing audio from: {args.audio_file_path}...")
    try:
        # CLAP model is already on DEVICE and in eval mode, potentially with mixed precision.
        clap_audio_embedding = get_clap_audio_embedding(args.audio_file_path, clap_model, clap_processor, DEVICE)
        print(f"CLAP audio embedding extracted, shape: {clap_audio_embedding.shape}")
    except Exception as e:
        print(f"Error processing audio file: {e}"); return

    # 8. Project Audio Embedding and Prepare for Pipeline
    with torch.no_grad():
        # audio_projection (if used) is on DEVICE, float32, and in eval mode.
        # Input clap_audio_embedding is on DEVICE, cast to float32 for projection.
        projected_audio_embedding = audio_projection(clap_audio_embedding.float()) if audio_projection else clap_audio_embedding
    
    # Cast to the target model_dtype for the pipeline. Shape: [1, UNET_CROSS_ATTENTION_DIM or AUDIO_EMBEDDING_DIM]
    current_prompt_embeds = projected_audio_embedding.to(dtype=model_dtype)

    negative_prompt_embeds: Optional[torch.Tensor] = None
    if args.guidance_scale > 1.0:
        print("Generating unconditional embeddings for guidance...")
        uncond_tokens = [""] 
        max_length = tokenizer.model_max_length
        # Tokenizer and text_encoder are on DEVICE and in eval mode, possibly with mixed precision.
        uncond_input = tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            unconditional_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE), attention_mask=uncond_input.attention_mask.to(DEVICE))[0]
        negative_prompt_embeds = unconditional_embeddings.to(dtype=model_dtype)
        print(f"Unconditional embeddings generated, shape: {negative_prompt_embeds.shape}")

        seq_len = negative_prompt_embeds.shape[1]
        current_prompt_embeds = current_prompt_embeds.unsqueeze(1).repeat(1, seq_len, 1)
    else:
        current_prompt_embeds = current_prompt_embeds.unsqueeze(1)

    print(f"Final prompt_embeds shape for pipeline: {current_prompt_embeds.shape}")
    if negative_prompt_embeds is not None: print(f"Final negative_prompt_embeds shape: {negative_prompt_embeds.shape}")

    # 9. Generate Image
    print(f"Generating image (guidance: {args.guidance_scale}, steps: {args.num_inference_steps}, seed: {args.seed})...")
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed) if args.seed is not None else None
    try:
        # Use autocast for the pipeline call if mixed precision for models is active.
        # Inputs (current_prompt_embeds, negative_prompt_embeds) are already cast to model_dtype.
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE if DEVICE != "cpu" else "cpu", enabled=(model_dtype != torch.float32 and DEVICE=="cuda"), dtype=model_dtype if model_dtype != torch.float32 else None):
            image_output = pipeline(
                prompt_embeds=current_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            image: Image.Image = image_output.images[0]
    except Exception as e:
        print(f"Error during image generation: {e}"); return

    # 10. Save Image
    try:
        output_dir = os.path.dirname(args.output_image_path)
        if output_dir and not os.path.exists(output_dir): # Ensure output directory exists if specified
            os.makedirs(output_dir, exist_ok=True)
        image.save(args.output_image_path)
        print(f"Generated image saved to {args.output_image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate artwork from audio using a trained LoRA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument("--audio_file_path", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH_DEFAULT, help="Path to the trained .safetensors checkpoint.")
    parser.add_argument("--output_image_path", type=str, default=OUTPUT_IMAGE_PATH_DEFAULT, help="Path to save the generated image.")
    
    parser.add_argument("--base_model_name", type=str, default=BASE_MODEL_NAME_DEFAULT, help="Base Stable Diffusion model name.")
    parser.add_argument("--clap_model_name", type=str, default=CLAP_MODEL_NAME_DEFAULT, help="CLAP model name.")
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK_DEFAULT, help="Rank of LoRA layers (must match training).")
    
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale. Use <=1.0 for no/less guidance.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation (for reproducibility).")
    parser.add_argument("--mixed_precision_inference", type=str, default=None, choices=["fp16", "bf16"], help="Enable mixed precision (fp16/bf16) for inference models on CUDA.")
    
    cli_args = parser.parse_args()
    main(cli_args)

    print(f"\n--- Inference Script Execution Finished ---")
    if os.path.exists(cli_args.output_image_path):
        print(f"To view the image, open: {cli_args.output_image_path}")
    print("Example command (ensure paths and rank match your setup):")
    # Corrected multi-line f-string for example command
    example_command = (
        f"python generate_artwork.py --audio_file_path path/to/your/audio.mp3 "
        f"--checkpoint_path {CHECKPOINT_PATH_DEFAULT} "
        f"--output_image_path desired_artwork.png --lora_rank {LORA_RANK_DEFAULT}"
    )
    print(example_command) 
