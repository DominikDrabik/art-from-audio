import os
import io
import requests
import re
import subprocess
import numpy as np
import librosa
import torch
from transformers import ClapModel, AutoProcessor
import imageio_ffmpeg as _ff
import pandas as pd


# === Configuration ===
GENRE_IDS = {
    "heavy_metal": 464,
    "jazz": 129,
    "reggae": 144,
}
NUM_TRACKS = 1
TARGET_SR = 48_000
MODEL_ID  = "laion/larger_clap_music_and_speech"

# Directories
EMBED_DIR = "embeddings"
IMAGE_DIR = "images"
MANIFEST_CSV = "manifest.csv"

os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

devnull = subprocess.DEVNULL
ffmpeg_bin = _ff.get_ffmpeg_exe()
model = ClapModel.from_pretrained(MODEL_ID).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)


def sanitize(text: str) -> str:
    """Replace non-word chars with underscore"""
    return "_".join(
        [t for t in re.split(r"[^\w]", text) if t]
    ).lower()


def decode_mp3_to_wave(audio_bytes: bytes, target_sr: int = TARGET_SR):
    """
    Decode MP3 bytes into a mono waveform at target_sr using ffmpeg via subprocess.
    Returns: (y: np.ndarray, sr: int)
    """
    proc = subprocess.Popen(
        [ffmpeg_bin, "-i", "pipe:0", "-f", "wav", "-ar", str(target_sr), "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=devnull
    )
    wav_bytes, _ = proc.communicate(audio_bytes)
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=target_sr, mono=True)
    return y, sr


def get_audio_embedding(waveform: np.ndarray, sr: int) -> torch.Tensor:
    """Run the CLAP model to get an audio embedding"""
    inputs = processor(audios=waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    return emb.squeeze(0)


def get_top_tracks_for_genre(genre_id: int, limit: int):
    """Fetch top tracks chart for a genre"""
    url = f"https://api.deezer.com/chart/{genre_id}/tracks"
    resp = requests.get(url, params={"limit": limit}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])

manifest = []
for genre_name, genre_id in GENRE_IDS.items():
    tracks = get_top_tracks_for_genre(genre_id, NUM_TRACKS)
    for tr in tracks:
        title      = tr.get("title", "unknown")
        print("Doing this for track: ", title)
        track_id   = tr.get("id")
        preview    = tr.get("preview")
        print("This is the preview url: ", preview)
        cover_url  = tr.get("album", {}).get("cover_xl")
        
        base = f"{sanitize(title)}_{track_id}_{genre_name}"
        
        audio_bytes = requests.get(preview, timeout=10).content
        waveform, sr = decode_mp3_to_wave(audio_bytes)
        
        emb = get_audio_embedding(waveform, sr)
        emb_path = os.path.join(EMBED_DIR, base + ".pt")
        torch.save(emb, emb_path)
        
        img_resp = requests.get(cover_url, timeout=10)
        img_resp.raise_for_status()
        ext = cover_url.split(".")[-1].split("?")[0]
        img_path = os.path.join(IMAGE_DIR, base + f".{ext}")
        with open(img_path, "wb") as f:
            f.write(img_resp.content)
        
        manifest.append({
            "track_id": track_id,
            "title": title,
            "genre": genre_name,
            "embedding_path": emb_path,
            "image_path": img_path
        })

# Save manifest to CSV
pd.DataFrame(manifest).to_csv(MANIFEST_CSV, index=False)
print(f"Saved manifest with {len(manifest)} entries to {MANIFEST_CSV}")
