"""
Jamendo Track Downloader and CLAP Audio Embedding Extractor

Requirements:
- A Jamendo API key (set as JAMENDO_API_KEY in a .env file).

This script downloads a set of music tracks and their cover images from the Jamendo API
based on a specified tag (genre), date range, and popularity. For each track, it:
- Downloads the audio file (MP3)
- Downloads the album or track image
- Extracts a CLAP audio embedding using the laion/larger_clap_music_and_speech model
- Saves the embedding as a .pt file

Configuration (API key, tag, date range, etc.) is set at the top of the script.

Usage:
    python download_jamendo_tracks_and_embeddings.py

Arguments are set as constants in the script. You may need to set your JAMENDO_API_KEY as an environment variable.

Output:
    - Audio files in the TRACK_DIR directory
    - Images in the IMAGE_DIR directory
    - Audio embeddings in the EMBED_DIR directory
"""

import os
import re
import requests
import torch
import librosa
from transformers import ClapModel, AutoProcessor
from typing import Optional, List, Dict

API_KEY       = os.getenv("JAMENDO_API_KEY")
BASE_URL      = "https://api.jamendo.com/v3.0"
TAG           = "jazz"                        
DATE_BETWEEN  = "2018-01-01_2025-06-11"      
NUM_TRACKS    = 5                             
ORDER_FIELD   = "popularity_total"           

TRACK_DIR       = "tracks"
IMAGE_DIR       = "images"
EMBED_DIR       = "audio_embeddings"

CLAP_MODEL_ID     = "laion/larger_clap_music_and_speech"
TARGET_SAMPLE_RATE = 48000

def jamendo_get(endpoint: str, params: Dict) -> Dict:
    params.update({"client_id": API_KEY, "format": "json"})
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def sanitize(text: str) -> str:
    return re.sub(r"[^\w-]", "_", text)


def load_clap() -> Optional[tuple]:
    try:
        model = ClapModel.from_pretrained(CLAP_MODEL_ID)
        processor = AutoProcessor.from_pretrained(CLAP_MODEL_ID)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"Error loading CLAP: {e}")
        return None, None


def extract_embedding(audio_path: str, model: ClapModel, processor: AutoProcessor) -> Optional[torch.Tensor]:
    try:
        waveform, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        inputs = processor(audios=waveform, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = model.get_audio_features(**inputs)
        return emb.squeeze(0)
    except Exception as e:
        print(f"Failed embedding for {audio_path}: {e}")
        return None


if __name__ == "__main__":
    for d in (TRACK_DIR, IMAGE_DIR, EMBED_DIR):
        os.makedirs(d, exist_ok=True)

    clap_model, clap_processor = load_clap()
    if not clap_model or not clap_processor:
        raise RuntimeError("Could not load CLAP model/processor.")

    params = {
        "tags": TAG,
        "datebetween": DATE_BETWEEN,
        "order": ORDER_FIELD,
        "limit": NUM_TRACKS,
        "include": "musicinfo"
    }
    data = jamendo_get("/tracks/", params)
    tracks = data.get("results", [])
    if not tracks:
        print("No tracks returned.")
        exit(0)

    for tr in tracks:
        tid   = tr["id"]
        name  = sanitize(tr.get("name", f"track{tid}"))
        genre = TAG

        audio_url = tr.get("audiodownload") if tr.get("audiodownload_allowed") else tr.get("audio")
        audio_fname = f"{name}_{tid}_{genre}.mp3"
        audio_path  = os.path.join(TRACK_DIR, audio_fname)
        print(f"Downloading audio: {audio_fname}")
        download_file(audio_url, audio_path)

        img_url = tr.get("album_image") or tr.get("image")
        img_ext = img_url.split("?")[0].rsplit(".", 1)[-1] if "." in img_url.split("?")[0] else "jpg"
        img_fname = f"{name}_{tid}_{genre}.{img_ext}"
        img_path  = os.path.join(IMAGE_DIR, img_fname)
        print(f"Downloading image: {img_fname}")
        download_file(img_url, img_path)

        emb = extract_embedding(audio_path, clap_model, clap_processor)
        if emb is not None:
            embed_fname = f"{name}_{tid}_{genre}.pt"
            embed_path  = os.path.join(EMBED_DIR, embed_fname)
            torch.save(emb, embed_path)
            print(f"Saved embedding: {embed_fname}")

    print("Done!")
