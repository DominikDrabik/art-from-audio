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

    vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae").to(DEVICE, dtype=model_dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(base_model_name, subfolder="unet").to(DEVICE, dtype=model_dtype).eval()
    text_encoder = CLIPTextModel.from_pretrained(base_model_name, subfolder="text_encoder").to(DEVICE, dtype=model_dtype).eval()
    tokenizer = CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler")

    hidden_projection_dim = (AUDIO_EMBEDDING_DIM + UNET_CROSS_ATTENTION_DIM) // 2
    audio_projection = nn.Sequential(
        nn.Linear(AUDIO_EMBEDDING_DIM, hidden_projection_dim),
        nn.ReLU(),
        nn.Linear(hidden_projection_dim, UNET_CROSS_ATTENTION_DIM)
    ).to(DEVICE, dtype=model_dtype).eval()
    print(f"Defined MLP audio projection for inference: {AUDIO_EMBEDDING_DIM} -> {hidden_projection_dim} -> {UNET_CROSS_ATTENTION_DIM}")


    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian", 
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
        bias="none",
    )
    unet.add_adapter(lora_config)
    print(f"Added LoRA adapter to UNet with rank {lora_rank}.")

    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = load_file(checkpoint_path, device="cpu") 

    unet_lora_weights = {k.replace("unet_lora.", "", 1): v.to(DEVICE, dtype=model_dtype) for k, v in state_dict.items() if k.startswith("unet_lora.")}
    audio_proj_weights = {k.replace("audio_projection.", "", 1): v.to(DEVICE, dtype=model_dtype) for k, v in state_dict.items() if k.startswith("audio_projection.")}

    if not unet_lora_weights:
        print("Warning: No LoRA weights found in the checkpoint for UNet.")
    else:
        unet.load_state_dict(unet_lora_weights, strict=False) # strict=False for LoRA
        print("Successfully loaded LoRA weights into UNet.")

    if not audio_proj_weights:
        print("Warning: No audio_projection weights found in the checkpoint.")
    elif audio_projection:
        audio_projection.load_state_dict(audio_proj_weights, strict=True)
        print("Successfully loaded weights into MLP audio_projection.")

    pipeline = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False
    ).to(DEVICE) 

    return {
        "pipeline": pipeline,
        "clap_model": clap_model,
        "clap_processor": clap_processor,
        "audio_projection": audio_projection, 
        "tokenizer": tokenizer, 
        "text_encoder": text_encoder, 
        "model_dtype": model_dtype, 
    }

def generate_image_for_audio(audio_path: str, output_path: str, models: Dict[str, Any], args: Any):
    pipeline = models["pipeline"]
    clap_model = models["clap_model"]
    clap_processor = models["clap_processor"]
    audio_projection_mlp = models["audio_projection"]
    model_dtype = models["model_dtype"]

    with torch.no_grad():
        audio_embedding = get_clap_audio_embedding(audio_path, clap_model, clap_processor, DEVICE)
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
                print(f"Repeating prompt_embeds sequence length from {prompt_embeds.shape[1]} to {negative_prompt_embeds.shape[1]}")
                prompt_embeds = prompt_embeds.repeat(1, negative_prompt_embeds.shape[1], 1)
            
        generator = torch.Generator(device=DEVICE).manual_seed(args.seed) if args.seed is not None else None

        with torch.amp.autocast(device_type=DEVICE, dtype=model_dtype if model_dtype != torch.float32 else None, enabled=(model_dtype != torch.float32)):
            image = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Saved image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate artwork from a single audio file using a trained LoRA model with MLP projection.")
    parser.add_argument("--audio_file_path", type=str, required=True, help="Path to the input audio file (.mp3, .wav).")
    parser.add_argument("--output_image_path", type=str, required=True, help="Path to save the generated image.")
    parser.add_argument("--checkpoint_path", type=str, default="lora_mlp_output/final_lora_mlp_projection_weights.safetensors", help="Path to the LoRA and MLP projection weights checkpoint (.safetensors).")
    parser.add_argument("--base_model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model name from Hugging Face.")
    parser.add_argument("--clap_model_name", type=str, default="laion/larger_clap_music_and_speech", help="CLAP model name from Hugging Face.")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of the LoRA adaptation.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation (optional).")
    parser.add_argument("--mixed_precision_inference", type=str, choices=["fp16", "bf16", "no"], default="no", help="Mixed precision for inference ('fp16', 'bf16', or 'no').")

    args = parser.parse_args()

    if args.mixed_precision_inference == "no":
        args.mixed_precision_inference = None

    print("Loading models...")
    models_dict = load_models(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model_name,
        clap_model_name=args.clap_model_name,
        lora_rank=args.lora_rank,
        mixed_precision=args.mixed_precision_inference
    )
    print("Models loaded. Generating image...")
    generate_image_for_audio(args.audio_file_path, args.output_image_path, models_dict, args)
    print("Image generation complete.")