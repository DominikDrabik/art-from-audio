import os
import argparse
from glob import glob
from with_simple_mlp.generate_artwork_simple_mlp import load_models, generate_image_for_audio

def generate_images_from_folder_mlp(args):
    input_dir = args.audio_folder_path
    if not os.path.isdir(input_dir):
        print(f"Error: Provided path is not a valid directory: {input_dir}")
        return

    output_dir_suffix = "_mlp_images"
    output_dir = os.path.join(os.path.dirname(input_dir.rstrip("/\\")), os.path.basename(input_dir.rstrip("/\\")) + output_dir_suffix)
    if args.output_folder_path: 
        output_dir = args.output_folder_path
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_dir}")


    audio_files = sorted(glob(os.path.join(input_dir, "*.mp3")) + glob(os.path.join(input_dir, "*.wav")))
    if not audio_files:
        print(f"No audio files (.mp3, .wav) found in folder: {input_dir}")
        return
    print(f"Found {len(audio_files)} audio files to process.")

    print(f"Loading models once for all audio files (using MLP projection)...")
    mixed_precision_val = args.mixed_precision_inference if args.mixed_precision_inference != "no" else None
    
    models_dict = load_models(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model_name,
        clap_model_name=args.clap_model_name,
        lora_rank=args.lora_rank,
        mixed_precision=mixed_precision_val
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
    parser = argparse.ArgumentParser(description="Batch generate artwork for audio files in a folder using LoRA with MLP projection.")
    parser.add_argument("--audio_folder_path", type=str, required=True, help="Path to the folder containing audio files (.mp3, .wav).")
    parser.add_argument("--output_folder_path", type=str, default=None, help="Optional: Specific path to save generated images. If None, defaults to 'audio_folder_path_mlp_images'.")
    parser.add_argument("--checkpoint_path", type=str, default="lora_mlp_output/final_lora_mlp_projection_weights.safetensors", help="Path to the LoRA and MLP projection weights checkpoint (.safetensors).")
    parser.add_argument("--base_model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model name.")
    parser.add_argument("--clap_model_name", type=str, default="laion/larger_clap_music_and_speech", help="CLAP model name.")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of the LoRA adaptation.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation (optional, applied to all images).")
    parser.add_argument("--mixed_precision_inference", type=str, choices=["fp16", "bf16", "no"], default="no", help="Mixed precision for inference ('fp16', 'bf16', or 'no').")

    args = parser.parse_args()
    generate_images_from_folder_mlp(args)