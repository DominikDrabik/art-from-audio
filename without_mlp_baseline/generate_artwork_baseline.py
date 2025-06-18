### ✅ generate_artwork.py (MODULAR + CLI-Compatible)

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

# --- Helper: Extract CLAP Embedding ---
def get_clap_audio_embedding(audio_path, clap_model, clap_processor, device) -> torch.Tensor:
    audio_waveform, _ = librosa.load(audio_path, sr=CLAP_TARGET_SAMPLE_RATE, mono=True)
    inputs = clap_processor(audios=[audio_waveform], return_tensors="pt", sampling_rate=CLAP_TARGET_SAMPLE_RATE, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_embeddings = clap_model.get_audio_features(**inputs)
    if audio_embeddings.ndim == 1:
        audio_embeddings = audio_embeddings.unsqueeze(0)
    return audio_embeddings

# --- Load All Models Once ---
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
        elif mixed_precision == "fp16":
            model_dtype = torch.float16

    # Load CLAP
    clap_model = ClapModel.from_pretrained(clap_model_name).to(DEVICE, dtype=model_dtype).eval()
    clap_processor = ClapProcessor.from_pretrained(clap_model_name)

    # Load SD Components
    vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae").to(DEVICE, dtype=model_dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(base_model_name, subfolder="unet").to(DEVICE, dtype=model_dtype).eval()
    text_encoder = CLIPTextModel.from_pretrained(base_model_name, subfolder="text_encoder").to(DEVICE, dtype=model_dtype).eval()
    tokenizer = CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler")

    # Audio Projection
    audio_projection = None
    if AUDIO_EMBEDDING_DIM != UNET_CROSS_ATTENTION_DIM:
        audio_projection = nn.Linear(AUDIO_EMBEDDING_DIM, UNET_CROSS_ATTENTION_DIM).to(DEVICE).eval()

    # Add LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
        bias="none",
    )
    unet.add_adapter(lora_config)

    # Load Checkpoint
    state_dict = load_file(checkpoint_path, device="cpu")
    unet_lora = {k.replace("unet_lora.", "", 1): v for k, v in state_dict.items() if k.startswith("unet_lora.")}
    audio_proj_sd = {k.replace("audio_projection.", "", 1): v for k, v in state_dict.items() if k.startswith("audio_projection.")}
    unet.load_state_dict(unet_lora, strict=False)
    if audio_projection and audio_proj_sd:
        audio_projection.load_state_dict(audio_proj_sd)

    pipeline = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False
    )

    return {
        "pipeline": pipeline,
        "clap_model": clap_model,
        "clap_processor": clap_processor,
        "audio_projection": audio_projection,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "model_dtype": model_dtype,
    }

# --- Inference Function ---
def generate_image_for_audio(audio_path: str, output_path: str, models: Dict[str, Any], args: Any):
    with torch.no_grad():
        embedding = get_clap_audio_embedding(audio_path, models["clap_model"], models["clap_processor"], DEVICE)
        embedding = embedding.to(torch.float32)
        if models["audio_projection"]:
            embedding = models["audio_projection"](embedding)
        prompt_embeds = embedding.to(dtype=models["model_dtype"]).unsqueeze(1)

        negative_prompt_embeds = None
        if args.guidance_scale > 1.0:
            tokens = [""]
            inputs = models["tokenizer"](tokens, padding="max_length", max_length=models["tokenizer"].model_max_length, truncation=True, return_tensors="pt")
            uncond_embeds = models["text_encoder"](inputs.input_ids.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE))[0]
            negative_prompt_embeds = uncond_embeds.to(dtype=models["model_dtype"])
            prompt_embeds = prompt_embeds.repeat(1, negative_prompt_embeds.shape[1], 1)

        generator = torch.Generator(device=DEVICE).manual_seed(args.seed) if args.seed else None
        with torch.amp.autocast(device_type=DEVICE, enabled=(models["model_dtype"] != torch.float32)):
            image = models["pipeline"](
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Saved image to: {output_path}")

# --- CLI Wrapper ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file_path", type=str, required=True)
    parser.add_argument("--output_image_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="lora_output/final_lora_and_projection_weights.safetensors")
    parser.add_argument("--base_model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--clap_model_name", type=str, default="laion/larger_clap_music_and_speech")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision_inference", type=str, choices=["fp16", "bf16"], default=None)
    args = parser.parse_args()

    models = load_models(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model_name,
        clap_model_name=args.clap_model_name,
        lora_rank=args.lora_rank,
        mixed_precision=args.mixed_precision_inference
    )
    generate_image_for_audio(args.audio_file_path, args.output_image_path, models, args)
