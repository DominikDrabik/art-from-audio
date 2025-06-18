"""
Batch Artwork Generation from Audio using Advanced MLP and (optionally) ControlNet

This script processes a folder of audio files (.mp3 or .wav), generating an image for each using a Stable Diffusion pipeline
with an advanced MLP audio projection and optional ControlNet conditioning. Models are loaded once for efficiency, and
output images are saved to a specified or automatically determined folder.

Parameters:
    --audio_folder_path (str, required): Path to the folder containing audio files.
    --output_folder_path (str, optional): Path to save generated images. If not set, a default is used.
    --checkpoint_path (str): Path to LoRA/MLP weights checkpoint.
    --base_model_name (str): HuggingFace model name for Stable Diffusion base.
    --clap_model_name (str): HuggingFace model name for CLAP audio encoder.
    --lora_rank (int): LoRA rank for UNet adapter.
    --num_inference_steps (int): Number of diffusion steps.
    --guidance_scale (float): Classifier-free guidance scale.
    --seed (int, optional): Random seed for reproducibility.
    --mixed_precision_inference (str): Use "fp16", "bf16", or "no" (default) for model inference precision.
    --controlnet_model_name (str, optional): ControlNet model name. If not set but --condition_image_path is provided, a default is used.
    --condition_image_path (str, optional): Path to a single image for ControlNet conditioning (applied to all audio files).
    --controlnet_conditioning_scale (float): ControlNet conditioning scale.

Usage:
    python generate_artwork_folder_advanced_mlp_controlnet.py --audio_folder_path <folder> [other options]

"""
import os
import argparse
from glob import glob
from generate_artwork_advanced_mlp_controlnet import load_models, generate_image_for_audio, DEFAULT_CONTROLNET_MODEL

def generate_images_from_folder_advanced_mlp(args):
    input_dir = args.audio_folder_path
    if not os.path.isdir(input_dir):
        print(f"Error: Provided path is not a valid directory: {input_dir}")
        return

    output_dir_suffix = "_adv_mlp_images"
    
    determined_controlnet_for_folder = args.controlnet_model_name
    if args.condition_image_path and not args.controlnet_model_name:
        determined_controlnet_for_folder = DEFAULT_CONTROLNET_MODEL
        
    if determined_controlnet_for_folder and args.condition_image_path:
        output_dir_suffix += "_controlnet"
        
    output_dir_name = os.path.basename(input_dir.rstrip("/\\")) + output_dir_suffix
    output_dir = os.path.join(os.path.dirname(input_dir.rstrip("/\\")) or ".", output_dir_name)

    if args.output_folder_path:
        output_dir = args.output_folder_path
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_dir}")

    audio_files = sorted(glob(os.path.join(input_dir, "*.mp3")) + glob(os.path.join(input_dir, "*.wav")))
    if not audio_files:
        print(f"No audio files found in folder: {input_dir}")
        return
    print(f"Found {len(audio_files)} audio files to process.")

    print(f"Loading models once for all audio files (using Advanced MLP projection)...")
    mixed_precision_val = args.mixed_precision_inference if args.mixed_precision_inference != "no" else None
    
    controlnet_model_for_loading = args.controlnet_model_name
    if args.condition_image_path and not args.controlnet_model_name:
        print(f"Condition image ('{args.condition_image_path}') provided for folder processing, and no ControlNet model specified. Using default: {DEFAULT_CONTROLNET_MODEL}")
        controlnet_model_for_loading = DEFAULT_CONTROLNET_MODEL
    elif not args.condition_image_path and args.controlnet_model_name:
        print(f"Warning: ControlNet model ('{args.controlnet_model_name}') specified, but no --condition_image_path provided for folder processing. ControlNet will be loaded but not used with an image condition for any file.")

    models_dict = load_models(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model_name,
        clap_model_name=args.clap_model_name,
        lora_rank=args.lora_rank,
        mixed_precision=mixed_precision_val,
        controlnet_model_name=controlnet_model_for_loading
    )
    print("Models loaded successfully.")

    for i, audio_path in enumerate(audio_files):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_image_path = os.path.join(output_dir, f"{base_name}.png")
        
        print(f"\nProcessing file {i+1}/{len(audio_files)}: {audio_path} -> {output_image_path}")
        try:
            generate_image_for_audio(audio_path, output_image_path, models_dict, args)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")


    print("\nBatch image generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate artwork with Advanced MLP projection.")
    parser.add_argument("--audio_folder_path", type=str, required=True)
    parser.add_argument("--output_folder_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="lora_advanced_mlp_output/final_lora_advanced_mlp_weights.safetensors")
    parser.add_argument("--base_model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--clap_model_name", type=str, default="laion/larger_clap_music_and_speech")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision_inference", type=str, choices=["fp16", "bf16", "no"], default="no")
    parser.add_argument("--controlnet_model_name", type=str, default=None,
                        help=f"Optional ControlNet model name. If --condition_image_path is set and this is not, defaults to '{DEFAULT_CONTROLNET_MODEL}'. Applied to all images in folder.")
    parser.add_argument("--condition_image_path", type=str, default=None, help="Optional path to a single image for ControlNet conditioning. Applied to all images in folder.")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale.")

    args = parser.parse_args()
    if args.mixed_precision_inference == "no": # Ensure None is passed if "no"
        args.mixed_precision_inference = None
        
    generate_images_from_folder_advanced_mlp(args)