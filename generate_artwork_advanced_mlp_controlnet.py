"""
Single Artwork Generation from Audio using Advanced MLP and (optionally) ControlNet

This script generates a single image from an audio file using a Stable Diffusion pipeline with an advanced MLP audio projection.
Optionally, ControlNet can be used for image conditioning. The script loads all required models, projects the audio embedding,
and saves the generated image.

Parameters:
    --audio_file_path (str, required): Path to the input audio file.
    --output_image_path (str, required): Path to save the generated image.
    --checkpoint_path (str): Path to LoRA/MLP weights checkpoint.
    --base_model_name (str): HuggingFace model name for Stable Diffusion base.
    --clap_model_name (str): HuggingFace model name for CLAP audio encoder.
    --lora_rank (int): LoRA rank for UNet adapter.
    --num_inference_steps (int): Number of diffusion steps.
    --guidance_scale (float): Classifier-free guidance scale.
    --seed (int, optional): Random seed for reproducibility.
    --mixed_precision_inference (str): Use "fp16", "bf16", or "no" (default) for model inference precision.
    --controlnet_model_name (str, optional): ControlNet model name. If not set but --condition_image_path is provided, a default is used.
    --condition_image_path (str, optional): Path to an image for ControlNet conditioning.
    --controlnet_conditioning_scale (float): ControlNet conditioning scale.

Usage:
    python generate_artwork_advanced_mlp_controlnet.py --audio_file_path <audio> --output_image_path <image> [other options]

"""
import torch
import torch.nn as nn
from PIL import Image
from transformers import ClapModel, ClapProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler, ControlNetModel, StableDiffusionControlNetPipeline
from peft import LoraConfig
from safetensors.torch import load_file
import librosa
import os
import argparse
from typing import Optional, Dict, Any
import cv2
import numpy as np 

AUDIO_EMBEDDING_DIM = 512
UNET_CROSS_ATTENTION_DIM = 768
CLAP_TARGET_SAMPLE_RATE = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny" 

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
    mixed_precision: Optional[str] = None,
    controlnet_model_name: Optional[str] = None 
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

    hidden_projection_dim = 768
    audio_projection = nn.Sequential(
        nn.Linear(AUDIO_EMBEDDING_DIM, hidden_projection_dim),
        nn.LayerNorm(hidden_projection_dim),
        nn.ReLU(),
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
        
    controlnet = None
    pipeline_to_use = None 

    if controlnet_model_name: 
        print(f"Loading ControlNet model: {controlnet_model_name}")
        try:
            controlnet = ControlNetModel.from_pretrained(controlnet_model_name, torch_dtype=model_dtype).to(DEVICE).eval()
            pipeline_to_use = StableDiffusionControlNetPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
                scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False
            ).to(DEVICE)
            print("Using StableDiffusionControlNetPipeline.")
        except Exception as e:
            print(f"Failed to load ControlNet model {controlnet_model_name} or pipeline: {e}. Falling back to standard pipeline.")
            controlnet = None 
            pipeline_to_use = StableDiffusionPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False
            ).to(DEVICE)
            print("Fell back to StableDiffusionPipeline.")
    else:
        pipeline_to_use = StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False
        ).to(DEVICE)
        print("Using StableDiffusionPipeline (no ControlNet).")

    return {
        "pipeline": pipeline_to_use, "clap_model": clap_model, "clap_processor": clap_processor,
        "audio_projection": audio_projection, "tokenizer": tokenizer, "text_encoder": text_encoder,
        "model_dtype": model_dtype,
        "controlnet": controlnet, 
        "loaded_controlnet_model_name": controlnet_model_name if controlnet else None 
    }

def generate_image_for_audio(audio_path: str, output_path: str, models: Dict[str, Any], args: Any):
    pipeline = models["pipeline"]
    audio_projection_mlp = models["audio_projection"]
    model_dtype = models["model_dtype"]
    controlnet = models.get("controlnet")
    loaded_controlnet_model_name = models.get("loaded_controlnet_model_name") 

    control_image_input = None
    if controlnet and args.condition_image_path:
        if not os.path.exists(args.condition_image_path):
            print(f"Warning: Condition image not found at {args.condition_image_path}. Proceeding without ControlNet image conditioning.")
        else:
            print(f"Loading condition image from: {args.condition_image_path}")
            condition_image_pil = Image.open(args.condition_image_path).convert("RGB")
            
            if loaded_controlnet_model_name and "canny" in loaded_controlnet_model_name.lower():
                print("Applying Canny edge detection to condition image...")
                condition_image_cv = np.array(condition_image_pil)
                canny_image = cv2.Canny(condition_image_cv, 100, 200)
                control_image_input = Image.fromarray(canny_image).convert("RGB")
                print("Canny preprocessing applied.")
            else:
                control_image_input = condition_image_pil
                if loaded_controlnet_model_name:
                    print(f"Passing condition image directly (or implement specific preprocessing for {loaded_controlnet_model_name}).")
                else:
                    print("Passing condition image directly (ControlNet model name not specified for specific preprocessing).")
    
    with torch.no_grad():
        audio_embedding = get_clap_audio_embedding(audio_path, models["clap_model"], models["clap_processor"], DEVICE)
        audio_embedding = audio_embedding.to(dtype=audio_projection_mlp[0].weight.dtype) # Ensure dtype match for MLP
        projected_embedding = audio_projection_mlp(audio_embedding)
        prompt_embeds = projected_embedding.unsqueeze(1).to(dtype=model_dtype) # Shape: [batch_size, 1, cross_attention_dim]

        negative_prompt_embeds = None
        if args.guidance_scale > 1.0:
            tokens = [""] 
            uncond_input = models["tokenizer"](
                tokens,
                padding="max_length",
                max_length=models["tokenizer"].model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = models["text_encoder"](
                uncond_input.input_ids.to(DEVICE),
                attention_mask=uncond_input.attention_mask.to(DEVICE) if 'attention_mask' in uncond_input else None
            )[0].to(dtype=model_dtype) 

            if prompt_embeds.shape[0] != negative_prompt_embeds.shape[0]:
                if prompt_embeds.shape[0] == 1 and negative_prompt_embeds.shape[0] > 1:
                     prompt_embeds = prompt_embeds.repeat(negative_prompt_embeds.shape[0], 1, 1)
                elif negative_prompt_embeds.shape[0] == 1 and prompt_embeds.shape[0] > 1:
                     negative_prompt_embeds = negative_prompt_embeds.repeat(prompt_embeds.shape[0], 1, 1)
                else:
                    # If batch sizes are >1 and mismatched, this is a harder problem to auto-resolve.
                    # For now, we assume one of them is 1 if they mismatch, or they are already equal.
                    pass


            if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                print(f"Expanding prompt_embeds sequence length from {prompt_embeds.shape[1]} to {negative_prompt_embeds.shape[1]} to match negative_prompt_embeds.")
                prompt_embeds = prompt_embeds.repeat(1, negative_prompt_embeds.shape[1], 1)
        
        generator = torch.Generator(device=DEVICE).manual_seed(args.seed) if args.seed is not None else None
        
        pipeline_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "generator": generator,
        }

        if control_image_input and isinstance(pipeline, StableDiffusionControlNetPipeline):
            pipeline_kwargs["image"] = control_image_input
            if hasattr(args, 'controlnet_conditioning_scale') and args.controlnet_conditioning_scale is not None:
                 pipeline_kwargs["controlnet_conditioning_scale"] = args.controlnet_conditioning_scale
            print(f"Generating with ControlNet using condition image. Scale: {args.controlnet_conditioning_scale if hasattr(args, 'controlnet_conditioning_scale') else 'default'}")
        else:
            if controlnet and not control_image_input: 
                 print("ControlNet model loaded, but no condition image provided or found. Generating without ControlNet image conditioning.")
            elif not controlnet: 
                 print("Generating without ControlNet.")

        with torch.amp.autocast(device_type=DEVICE, dtype=model_dtype if model_dtype != torch.float32 else None, enabled=(model_dtype != torch.float32)):
            image = pipeline(**pipeline_kwargs).images[0]
        
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        image.save(output_path)
        print(f"Saved image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate artwork with Advanced MLP projection.")
    parser.add_argument("--audio_file_path", type=str, required=True)
    parser.add_argument("--output_image_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="lora_advanced_mlp_output/final_lora_advanced_mlp_weights.safetensors")
    parser.add_argument("--base_model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--clap_model_name", type=str, default="laion/larger_clap_music_and_speech")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision_inference", type=str, choices=["fp16", "bf16", "no"], default="no")
    parser.add_argument("--controlnet_model_name", type=str, default=None, 
                        help=f"Optional ControlNet model name. If --condition_image_path is set and this is not, defaults to '{DEFAULT_CONTROLNET_MODEL}'.")
    parser.add_argument("--condition_image_path", type=str, default=None, help="Optional path to an image for ControlNet conditioning.")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale.")

    args = parser.parse_args()
    if args.mixed_precision_inference == "no": 
        args.mixed_precision_inference = None
    
    actual_controlnet_model_to_load = args.controlnet_model_name
    if args.condition_image_path and not args.controlnet_model_name:
        print(f"Condition image ('{args.condition_image_path}') provided and no ControlNet model specified. Using default: {DEFAULT_CONTROLNET_MODEL}")
        actual_controlnet_model_to_load = DEFAULT_CONTROLNET_MODEL
    elif not args.condition_image_path and args.controlnet_model_name:
        print(f"Warning: ControlNet model ('{args.controlnet_model_name}') specified, but no --condition_image_path provided. ControlNet will be loaded but not used with an image condition.")

    print("Loading models with Advanced MLP...")
    models_dict = load_models(
        checkpoint_path=args.checkpoint_path, base_model_name=args.base_model_name,
        clap_model_name=args.clap_model_name, lora_rank=args.lora_rank,
        mixed_precision=args.mixed_precision_inference,
        controlnet_model_name=actual_controlnet_model_to_load
    )
    print("Models loaded. Generating image...")
    generate_image_for_audio(args.audio_file_path, args.output_image_path, models_dict, args)
    print("Image generation complete.")