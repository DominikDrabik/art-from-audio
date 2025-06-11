"""
Jamendo Audio-Cover Dataset Collector

This script collects audio tracks and their corresponding cover art from the Jamendo API.
It aims to build a dataset of (audio_snippet, cover_image) pairs suitable for training
multi-modal models.

Requirements:
- A Jamendo API key (set as JAMENDO_API_KEY in a .env file).
- Python libraries: requests, Pillow, pydub, python-dotenv.
"""

import csv
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydub import AudioSegment
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

API_KEY = "6acebb65"  # Replace with os.getenv("JAMENDO_API_KEY") for production

if not API_KEY:
    print("ERROR: JAMENDO_API_KEY not found.")
    exit(1)

BASE_URL = "https://api.jamendo.com/v3.0"
OUTPUT_DIR = "art-from-audio-dataset"
AUDIO_SNIPPETS_DIR = os.path.join(OUTPUT_DIR, "audio_snippets")
COVER_IMAGES_DIR = os.path.join(OUTPUT_DIR, "cover_images")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

TARGET_LICENSES_STRING = "by,by-sa,by-nc,by-nc-sa"
IMAGE_TARGET_WIDTH_PX = 300
AUDIO_SNIPPET_DURATION_S = 30

TRACKS_PER_API_CALL = 3
TRACKS_TO_COLLECT_PER_GENRE = 3
TARGET_GENRES = ["jazz", "metal", "techno"]
API_REQUEST_DELAY_S = 1

def setup_directories():
    os.makedirs(AUDIO_SNIPPETS_DIR, exist_ok=True)
    os.makedirs(COVER_IMAGES_DIR, exist_ok=True)
    print(f"Data will be saved in: {os.path.abspath(OUTPUT_DIR)}")

def init_metadata_file():
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            fieldnames = [
                "jamendo_track_id", "track_name", "artist_name", "album_name",
                "audio_filename", "image_filename", "jamendo_license_cc", "genre",
                "original_audio_duration_s", "actual_snippet_duration_s",
                "image_original_width_px", "image_original_height_px",
                "image_saved_width_px", "image_saved_height_px"
            ]
            writer.writerow(fieldnames)
        print(f"Metadata file created: {METADATA_FILE}")
    else:
        print(f"Metadata file already exists: {METADATA_FILE}")

def make_api_request(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    params['client_id'] = API_KEY
    params['format'] = 'json'
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def download_file_content(url: str) -> Optional[bytes]:
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Download error: {e}")
        return None

def create_audio_snippet(audio_bytes: bytes, output_path: str, duration_s: int) -> Tuple[bool, float]:
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        snippet_duration_ms = duration_s * 1000
        snippet = audio_segment[:snippet_duration_ms]
        snippet.export(output_path, format="mp3")
        return True, len(snippet) / 1000.0
    except Exception as e:
        print(f"Snippet creation failed: {e}")
        return False, 0.0

def resize_and_save_image(image_bytes: bytes, output_path: str, target_width_px: int) -> Tuple[bool, int, int, int, int]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size
        aspect_ratio = orig_h / orig_w
        target_h = int(target_width_px * aspect_ratio)
        img_resized = img.resize((target_width_px, target_h), Image.Resampling.LANCZOS)
        img_resized.save(output_path, "JPEG", quality=90)
        return True, orig_w, orig_h, target_width_px, target_h
    except Exception as e:
        print(f"Image processing failed: {e}")
        return False, 0, 0, 0, 0

def append_to_metadata(data: Dict[str, Any], fields: List[str]):
    with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({k: data.get(k) for k in fields})

if __name__ == "__main__":
    setup_directories()

    METADATA_FIELDS = [
        "jamendo_track_id", "track_name", "artist_name", "album_name",
        "audio_filename", "image_filename", "jamendo_license_cc", "genre",
        "original_audio_duration_s", "actual_snippet_duration_s",
        "image_original_width_px", "image_original_height_px",
        "image_saved_width_px", "image_saved_height_px"
    ]
    init_metadata_file()

    existing_ids = set()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_ids = {row["jamendo_track_id"] for row in reader if row.get("jamendo_track_id")}

    total_collected = 0

    for genre in TARGET_GENRES:
        print(f"\n--- Collecting for genre: {genre} ---")
        offset = 0
        collected = 0

        while collected < TRACKS_TO_COLLECT_PER_GENRE:
            print(f"\nFetching tracks... Offset: {offset}, Collected: {collected}/{TRACKS_TO_COLLECT_PER_GENRE}")
            params = {
                "limit": TRACKS_PER_API_CALL,
                "offset": offset,
                "license_cc": TARGET_LICENSES_STRING,
                "hasimage": "true",
                "album_datebetween": "2000-01-01_2024-12-31",
                "audioformat": "mp32",
                "tags": genre,
                "order": "popularity_week"
            }

            response = make_api_request("/tracks", params)
            if not response or not response.get("results"):
                print("No more results or API error.")
                break

            for track in response["results"]:
                if collected >= TRACKS_TO_COLLECT_PER_GENRE:
                    break

                track_id = str(track.get("id"))
                if not track_id or track_id in existing_ids:
                    continue

                audio_url = track.get("audio")  # <-- Use preview only
                image_url = track.get("album_image")
                if not audio_url or not image_url:
                    continue

                audio_bytes = download_file_content(audio_url)
                if not audio_bytes:
                    continue

                audio_filename = f"{track_id}.mp3"
                audio_path = os.path.join(AUDIO_SNIPPETS_DIR, audio_filename)
                success, duration = create_audio_snippet(audio_bytes, audio_path, AUDIO_SNIPPET_DURATION_S)
                if not success:
                    continue

                image_bytes = download_file_content(image_url)
                if not image_bytes:
                    continue

                image_filename = f"{track_id}.jpg"
                image_path = os.path.join(COVER_IMAGES_DIR, image_filename)
                img_success, orig_w, orig_h, resized_w, resized_h = resize_and_save_image(image_bytes, image_path, IMAGE_TARGET_WIDTH_PX)
                if not img_success:
                    continue

                metadata = {
                    "jamendo_track_id": track_id,
                    "track_name": track.get("name", "Unknown"),
                    "artist_name": track.get("artist_name", "Unknown"),
                    "album_name": track.get("album_name", "Unknown"),
                    "audio_filename": audio_filename,
                    "image_filename": image_filename,
                    "jamendo_license_cc": track.get("license_ccurl", "").split("/licenses/")[-1].split("/")[0],
                    "genre": genre,
                    "original_audio_duration_s": float(track.get("duration", 0)),
                    "actual_snippet_duration_s": duration,
                    "image_original_width_px": orig_w,
                    "image_original_height_px": orig_h,
                    "image_saved_width_px": resized_w,
                    "image_saved_height_px": resized_h
                }
                append_to_metadata(metadata, METADATA_FIELDS)
                existing_ids.add(track_id)
                collected += 1
                total_collected += 1

                print(f"  ✅ Track {track_id} saved.")

                time.sleep(0.1)

            offset += TRACKS_PER_API_CALL
            time.sleep(API_REQUEST_DELAY_S)

        print(f"--- Finished genre {genre}. Collected {collected} tracks. ---")

    print(f"\n--- Collection Finished ---")
    print(f"Total new tracks collected: {total_collected}")
    print(f"Audio snippets in: {AUDIO_SNIPPETS_DIR}")
    print(f"Cover images in: {COVER_IMAGES_DIR}")
