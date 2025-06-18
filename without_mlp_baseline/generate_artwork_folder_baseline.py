import os
import argparse
from glob import glob
from without_mlp_baseline.generate_artwork_baseline import load_models, generate_image_for_audio

def generate_images_from_folder(args):
    input_dir = args.audio_folder_path
    assert os.path.isdir(input_dir), f"Provided path is not a valid directory: {input_dir}"

    output_dir = input_dir.rstrip("/\\") + "_mlp_images"
    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted(glob(os.path.join(input_dir, "*.mp3")) + glob(os.path.join(input_dir, "*.wav")))
    if not audio_files:
        print(f"No audio files found in folder: {input_dir}")
        return

    print(f"Loading models once for all audio files...")
    models = load_models(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model_name,
        clap_model_name=args.clap_model_name,
        lora_rank=args.lora_rank,
        mixed_precision=args.mixed_precision_inference
    )

    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        print(f"Processing: {audio_path} -> {output_path}")
        generate_image_for_audio(audio_path, output_path, models, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate artwork for audio files in a folder.")
    parser.add_argument("--audio_folder_path", type=str, required=True, help="Path to folder containing audio files.")
    parser.add_argument("--checkpoint_path", type=str, default="lora_output/final_lora_and_projection_weights.safetensors")
    parser.add_argument("--base_model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--clap_model_name", type=str, default="laion/larger_clap_music_and_speech")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision_inference", type=str, choices=["fp16", "bf16"], default=None)

    args = parser.parse_args()
    generate_images_from_folder(args)
