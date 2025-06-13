import torch
import torch.nn as nn
from PIL import Image
from transformers import ClapModel, ClapProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import LoraConfig
from safetensors.torch import load_file
import librosa
import os
import argparse
from typing import Optional, Dict, Any

# --- Constants ---
AUDIO_EMBEDDING_DIM = 512
UNET_CROSS_ATTENTION_DIM = 768
CLAP_TARGET_SAMPLE_RATE = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_clap_audio_embedding(audio_path, clap_model, clap_processor, device) -> torch.Tensor:
    audio_waveform, _ = librosa.load(audio_path, sr=CLAP_TARGET_SAMPLE_RATE, mono=True)
    inputs = clap_processor(audios=[audio_waveform], return_tensors="pt", sampling_rate=CLAP_TARGET_SAMPLE_RATE, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_embeddings = clap_model.get_audio_features(**inputs)
    if audio_embeddings.ndim == 1:
        audio_embeddings = audio_embeddings.unsqueeze(0)
    elif audio_embeddings.ndim == 3 and audio_embeddings.shape[1] == 1 :
        audio_embeddings = audio_embeddings.squeeze(1)
    return audio_embeddings

def load_models(
    checkpoint_path: str,
    base_model_name: str,
    clap_model_name: str,
    lora_rank: int,
    mixed_precision: Optional[str] = None
) -> Dict[str, Any]:
    model_dtype = torch.float32
    if DEVICE == "cuda":
        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            print("Using BF16 for inference models.")
        elif mixed_precision == "fp16":
            model_dtype = torch.float16
            print("Using FP16 for inference models.")
        else:
            print("Using FP32 for inference models.")
    else:
        print("Using FP32 for inference models (CPU).")

    clap_model = ClapModel.from_pretrained(clap_model_name).to(DEVICE, dtype=model_dtype).eval()
    clap_processor = ClapProcessor.from_pretrained(clap_model_name)

    # Optionally load the same custom VAE used in training if you changed it
    # vae_model_name_inf = "stabilityai/sd-vae-ft-mse" # Or the one used in training
    # try:
    #     vae = AutoencoderKL.from_pretrained(vae_model_name_inf).to(DEVICE, dtype=model_dtype).eval()
    #     print(f"Loaded custom VAE for inference: {vae_model_name_inf}")
    # except Exception as e:
    #     print(f"Could not load custom VAE {vae_model_name_inf}, falling back. Error: {e}")
    vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae").to(DEVICE, dtype=model_dtype).eval()

    unet = UNet2DConditionModel.from_pretrained(base_model_name, subfolder="unet").to(DEVICE, dtype=model_dtype).eval()
    text_encoder = CLIPTextModel.from_pretrained(base_model_name, subfolder="text_encoder").to(DEVICE, dtype=model_dtype).eval()
    tokenizer = CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler")

    # --- Advanced Audio Projection Layer (must match training) ---
    hidden_projection_dim = 768 # Must match the value used in training
    audio_projection = nn.Sequential(
        nn.Linear(AUDIO_EMBEDDING_DIM, hidden_projection_dim),
        nn.LayerNorm(hidden_projection_dim), # Added LayerNorm
        nn.ReLU(), # Or nn.GELU() if used in training
        nn.Linear(hidden_projection_dim, UNET_CROSS_ATTENTION_DIM)
    ).to(DEVICE, dtype=model_dtype).eval()
    print(f"Defined Advanced MLP audio projection for inference: {AUDIO_EMBEDDING_DIM} -> LN -> ReLU -> {hidden_projection_dim} -> {UNET_CROSS_ATTENTION_DIM}")

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"], bias="none",
    )
    unet.add_adapter(lora_config)
    print(f"Added LoRA adapter to UNet with rank {lora_rank}.")

    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = load_file(checkpoint_path, device="cpu")
    unet_lora_weights = {k.replace("unet_lora.", "", 1): v.to(DEVICE, dtype=model_dtype) for k, v in state_dict.items() if k.startswith("unet_lora.")}
    audio_proj_weights = {k.replace("audio_projection.", "", 1): v.to(DEVICE, dtype=model_dtype) for k, v in state_dict.items() if k.startswith("audio_projection.")}

    if unet_lora_weights:
        unet.load_state_dict(unet_lora_weights, strict=False)
        print("Successfully loaded LoRA weights into UNet.")
    else: print("Warning: No LoRA weights found for UNet.")
    if audio_proj_weights and audio_projection:
        audio_projection.load_state_dict(audio_proj_weights, strict=True)
        print("Successfully loaded weights into Advanced MLP audio_projection.")
    else: print("Warning: No audio_projection weights found or audio_projection not defined.")
        
    pipeline = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False
    ).to(DEVICE)

    return {
        "pipeline": pipeline, "clap_model": clap_model, "clap_processor": clap_processor,
        "audio_projection": audio_projection, "tokenizer": tokenizer, "text_encoder": text_encoder,
        "model_dtype": model_dtype,
    }

def generate_image_for_audio(audio_path: str, output_path: str, models: Dict[str, Any], args: Any):
    pipeline = models["pipeline"]
    audio_projection_mlp = models["audio_projection"]
    model_dtype = models["model_dtype"]

    with torch.no_grad():
        audio_embedding = get_clap_audio_embedding(audio_path, models["clap_model"], models["clap_processor"], DEVICE)
        audio_embedding = audio_embedding.to(dtype=audio_projection_mlp[0].weight.dtype)
        projected_embedding = audio_projection_mlp(audio_embedding)
        prompt_embeds = projected_embedding.unsqueeze(1).to(dtype=model_dtype)

        negative_prompt_embeds = None
        if args.guidance_scale > 1.0:
            tokens = [""]
            inputs = models["tokenizer"](tokens, padding="max_length", max_length=models["tokenizer"].model_max_length, truncation=True, return_tensors="pt")
            uncond_embeds = models["text_encoder"](inputs.input_ids.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE))[0]
            negative_prompt_embeds = uncond_embeds.to(dtype=model_dtype)
            if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                prompt_embeds = prompt_embeds.repeat(1, negative_prompt_embeds.shape[1], 1)
            
        generator = torch.Generator(device=DEVICE).manual_seed(args.seed) if args.seed is not None else None
        with torch.amp.autocast(device_type=DEVICE, dtype=model_dtype if model_dtype != torch.float32 else None, enabled=(model_dtype != torch.float32)):
            image = pipeline(
                prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, generator=generator,
            ).images[0]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Saved image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate artwork with Advanced MLP projection.")
    parser.add_argument("--audio_file_path", type=str, required=True)
    parser.add_argument("--output_image_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="lora_advanced_mlp_output/final_lora_advanced_mlp_weights.safetensors") # Updated default
    parser.add_argument("--base_model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--clap_model_name", type=str, default="laion/larger_clap_music_and_speech")
    parser.add_argument("--lora_rank", type=int, default=4) # Should match training
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision_inference", type=str, choices=["fp16", "bf16", "no"], default="no")
    args = parser.parse_args()
    if args.mixed_precision_inference == "no": args.mixed_precision_inference = None
    
    print("Loading models with Advanced MLP...")
    models_dict = load_models(
        checkpoint_path=args.checkpoint_path, base_model_name=args.base_model_name,
        clap_model_name=args.clap_model_name, lora_rank=args.lora_rank,
        mixed_precision=args.mixed_precision_inference
    )
    print("Models loaded. Generating image...")
    generate_image_for_audio(args.audio_file_path, args.output_image_path, models_dict, args)
    print("Image generation complete.")