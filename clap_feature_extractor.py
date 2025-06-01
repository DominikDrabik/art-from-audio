"""
Extracts audio embeddings from a directory of audio files using a pre-trained CLAP model.

This script processes .mp3 files from a specified audio directory, 
extracts their embeddings using the LAION CLAP model, and saves 
the embeddings as PyTorch .pt files in a specified output directory.
"""
import glob
import os
from typing import Tuple, Optional

import librosa
import torch
from transformers import ClapModel, AutoProcessor

# --- Configuration ---
MODEL_ID = "laion/larger_clap_music_and_speech"
AUDIO_DIR = "art-from-audio-dataset/audio_snippets/" # Source directory for .mp3 files
EMBEDDINGS_DIR = "art-from-audio-dataset/audio_embeddings/" # Target directory for .pt embeddings
TARGET_SAMPLE_RATE = 48000 # CLAP model's expected sample rate

def load_model_and_processor() -> Tuple[Optional[ClapModel], Optional[AutoProcessor]]:
    """Loads the pre-trained CLAP model and processor from Hugging Face."""
    print(f"Loading CLAP model and processor: {MODEL_ID}...")
    try:
        model = ClapModel.from_pretrained(MODEL_ID)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded model and processor for {MODEL_ID}.")
        return model, processor
    except Exception as e:
        print(f"Error loading CLAP model or processor: {e}")
        return None, None

def extract_audio_embeddings(audio_file_path: str, model: ClapModel, processor: AutoProcessor) -> Optional[torch.Tensor]:
    """
    Extracts embeddings from a single audio file.

    Args:
        audio_file_path: Path to the audio file.
        model: The loaded CLAP model.
        processor: The loaded CLAP processor.

    Returns:
        Audio embeddings as a torch.Tensor, or None if an error occurs.
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return None

    try:
        print(f"Processing audio file: {audio_file_path}...")
        waveform, sample_rate = librosa.load(audio_file_path, sr=TARGET_SAMPLE_RATE, mono=True)
        
        inputs = processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            audio_features = model.get_audio_features(**inputs)
        
        print(f"  Successfully extracted embedding with shape {audio_features.shape}.")
        return audio_features
    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {e}")
        return None

def main():
    """Main function to orchestrate loading models and processing audio files."""
    model, processor = load_model_and_processor()

    if model and processor:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        print(f"Embeddings will be saved to: {EMBEDDINGS_DIR}")

        audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
        
        if not audio_files:
            print(f"No .mp3 files found in {AUDIO_DIR}. Please check the directory and file extensions.")
            return
        
        print(f"Found {len(audio_files)} .mp3 files in {AUDIO_DIR}.")

        for i, audio_file in enumerate(audio_files):
            print(f"--- Processing file {i+1}/{len(audio_files)}: {os.path.basename(audio_file)} ---")
            embeddings = extract_audio_embeddings(audio_file, model, processor)
            
            if embeddings is not None:
                base_filename = os.path.basename(audio_file)
                name_without_ext = os.path.splitext(base_filename)[0]
                save_path = os.path.join(EMBEDDINGS_DIR, f"{name_without_ext}.pt")
                
                try:
                    torch.save(embeddings, save_path)
                    print(f"  Successfully saved embedding to {save_path}")
                except Exception as e:
                    print(f"  Error saving embedding for {audio_file} to {save_path}: {e}")
            else:
                print(f"  Skipping saving for {audio_file} due to previous error.")
            # Removed the extra newline print for cleaner logs

    print("Audio embedding extraction process complete.")

if __name__ == "__main__":
    main() 
