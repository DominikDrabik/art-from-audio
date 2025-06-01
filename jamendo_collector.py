"""
Jamendo Audio-Cover Dataset Collector

This script collects audio tracks and their corresponding cover art from the Jamendo API.
It aims to build a dataset of (audio_snippet, cover_image) pairs suitable for training
multi-modal models.

Features:
- Fetches tracks based on specified Creative Commons licenses.
- Downloads audio (full track if available via 'audiodownload' URL, otherwise preview) and cover images.
- Creates fixed-length audio snippets (e.g., 30 seconds) from the downloaded audio and saves these snippets.
  (Note: Only the snippet is saved, not the full downloaded audio track).
- Resizes cover images to a consistent dimension.
- Saves all data into structured directories (`art-from-audio-dataset/`).
- Records metadata (track ID, names, filenames, licenses, etc.) in a CSV file.
- Skips tracks already present in the metadata file to avoid re-processing.

Requirements:
- A Jamendo API key (set as JAMENDO_API_KEY in a .env file).
- Python libraries: requests, Pillow, pydub, python-dotenv.
  (Install via: pip install requests Pillow pydub python-dotenv)

Usage:
1. Create a .env file in the same directory as this script.
2. Add your Jamendo API key to the .env file: 
   JAMENDO_API_KEY="your_actual_api_key_here"
3. Run the script: python jamendo_collector.py
4. Adjust configuration parameters (TOTAL_TRACKS_TO_COLLECT, etc.) as needed.
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
load_dotenv() # Load environment variables from .env file

API_KEY: Optional[str] = os.getenv("JAMENDO_API_KEY")

if not API_KEY:
    print("ERROR: JAMENDO_API_KEY not found in environment variables.")
    print("Please ensure it is set in your .env file (e.g., JAMENDO_API_KEY=\"your_key\")")
    print("and that the .env file is in the same directory as this script.")
    exit(1)

BASE_URL: str = "https://api.jamendo.com/v3.0"
OUTPUT_DIR: str = "art-from-audio-dataset"
AUDIO_SNIPPETS_DIR: str = os.path.join(OUTPUT_DIR, "audio_snippets")
COVER_IMAGES_DIR: str = os.path.join(OUTPUT_DIR, "cover_images")
METADATA_FILE: str = os.path.join(OUTPUT_DIR, "metadata.csv")

# Licenses: focus on those generally permissive for dataset creation.
# (CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA). Avoid 'ND' (NoDerivatives).
TARGET_LICENSES_STRING: str = "by,by-sa,by-nc,by-nc-sa" # Comma-separated for API
IMAGE_TARGET_WIDTH_PX: int = 300  # Target width for cover art
AUDIO_SNIPPET_DURATION_S: int = 30  # Desired audio snippet duration in seconds

# Collection parameters
TRACKS_PER_API_CALL: int = 50  # Number of tracks to fetch per API call
TOTAL_TRACKS_TO_COLLECT: int = 250 # Target number of tracks to collect
API_REQUEST_DELAY_S: int = 1  # Seconds to wait between API calls

# --- Helper Functions ---

def setup_directories():
    """Creates necessary output directories if they don't exist."""
    os.makedirs(AUDIO_SNIPPETS_DIR, exist_ok=True)
    os.makedirs(COVER_IMAGES_DIR, exist_ok=True)
    print(f"Data will be saved in: {os.path.abspath(OUTPUT_DIR)}")

def init_metadata_file():
    """Initializes the CSV metadata file with headers if it doesn't exist."""
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Define the exact order of columns for the CSV
            fieldnames = [
                "jamendo_track_id", "track_name", "artist_name", "album_name",
                "audio_filename", "image_filename", "jamendo_license_cc",
                "original_audio_duration_s", "actual_snippet_duration_s", 
                "image_original_width_px", "image_original_height_px",
                "image_saved_width_px", "image_saved_height_px"
            ]
            writer.writerow(fieldnames)
        print(f"Metadata file created: {METADATA_FILE} with headers: {fieldnames}")
    else:
        print(f"Metadata file already exists: {METADATA_FILE}")

def make_api_request(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Makes a request to the Jamendo API and handles common errors including rate limiting."""
    params['client_id'] = API_KEY
    params['format'] = 'json'
    response: Optional[requests.Response] = None
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10) # Added timeout
        response.raise_for_status()
        json_response = response.json()
        if "results" not in json_response and response.status_code == 200:
            # This case might indicate an API change or unexpected successful but empty response
            print(f"API Info: Request to {endpoint} successful (Status {response.status_code}), but no 'results' key in JSON.")
            print(f"Full API Response Headers: {json_response.get('headers')}")
            print(f"Full API Response (first 500 chars if large): {str(json_response)[:500]}")
        return json_response
    except requests.exceptions.HTTPError as http_err:
        print(f"API HTTP error: {http_err}")
        if response is not None:
            print(f"Status Code: {response.status_code}")
            print(f"Response Text (first 500 chars): {response.text[:500]}...")
            if response.status_code == 429: # Too Many Requests - Jamendo specific handling might vary
                retry_after = int(response.headers.get("Retry-After", 60)) # Use Retry-After header if available
                print(f"Rate limited by API. Waiting for {retry_after} seconds before retrying...")
                time.sleep(retry_after)
                return make_api_request(endpoint, params) # Recursive retry
        return None
    except requests.exceptions.Timeout:
        print(f"API request to {endpoint} timed out after 10 seconds. Retrying once after delay...")
        time.sleep(5) # Wait a bit before retrying a timeout
        return make_api_request(endpoint, params) # Simple retry, could be more sophisticated
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None
    except ValueError as json_err: # Handles errors in response.json() parsing (e.g. if not valid JSON)
        print(f"API JSON parsing error: {json_err}")
        if response is not None:
            print(f"Status Code: {response.status_code}")
            print(f"Received non-JSON response (first 500 chars): {response.text[:500]}...")
        return None

def download_file_content(url: str) -> Optional[bytes]:
    """Downloads file content from a URL and returns it as bytes."""
    try:
        response = requests.get(url, stream=True, timeout=10) # Added timeout
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading content from {url}: {e}")
        return None

def create_audio_snippet(audio_bytes: bytes, output_path: str, duration_s: int) -> Tuple[bool, float]:
    """Creates an audio snippet from raw audio bytes and saves it as MP3."""
    try:
        audio_format = "mp3" # Assuming Jamendo previews are MP3
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
        
        snippet_duration_ms = duration_s * 1000
        actual_snippet_segment = audio_segment[:snippet_duration_ms] # Slice to desired duration or less if shorter
        
        actual_snippet_segment.export(output_path, format="mp3")
        return True, len(actual_snippet_segment) / 1000.0 # Return success and actual snippet duration in seconds
    except Exception as e:
        print(f"Error creating audio snippet for {output_path}: {e}")
        return False, 0.0

def resize_and_save_image(image_bytes: bytes, output_path: str, target_width_px: int) -> Tuple[bool, int, int, int, int]:
    """Resizes an image from raw bytes and saves it as JPEG, maintaining aspect ratio."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB") # Ensure RGB for JPEG saving
        
        original_width, original_height = img.size
        aspect_ratio = original_height / original_width
        target_height_px = int(target_width_px * aspect_ratio)
        
        img_resized = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
        img_resized.save(output_path, "JPEG", quality=90)
        return True, original_width, original_height, target_width_px, target_height_px
    except Exception as e:
        print(f"Error processing image for {output_path}: {e}")
        return False, 0, 0, 0, 0

def append_to_metadata(data_dict: Dict[str, Any], field_order: List[str]):
    """Appends a dictionary of data as a new row to the CSV file, respecting field_order."""
    # Ensure all fields in field_order are in data_dict, add None if missing for safety
    row_to_write = {field: data_dict.get(field) for field in field_order}
    with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        # Check if file is empty to write headers (though init_metadata_file should handle this for new files)
        if f.tell() == 0:
            writer.writeheader() 
        writer.writerow(row_to_write)

# --- Main Collection Logic ---
if __name__ == "__main__":
    # API_KEY check is already done at the top
    setup_directories()
    
    # Define metadata field order here to be used by init and append
    METADATA_FIELD_ORDER = [
        "jamendo_track_id", "track_name", "artist_name", "album_name",
        "audio_filename", "image_filename", "jamendo_license_cc",
        "original_audio_duration_s", "actual_snippet_duration_s", 
        "image_original_width_px", "image_original_height_px",
        "image_saved_width_px", "image_saved_height_px"
    ]
    init_metadata_file() # This will use the new field order if creating the file

    existing_track_ids = set()
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "jamendo_track_id" not in reader.fieldnames:
                    print(f"Warning: Metadata file {METADATA_FILE} is empty or missing 'jamendo_track_id' header. Will not skip existing tracks.")
                else:
                    for row in reader:
                        if row.get('jamendo_track_id'): # Check if key exists and is not empty
                            existing_track_ids.add(row['jamendo_track_id'])
            print(f"Found {len(existing_track_ids)} existing tracks in metadata. These will be skipped.")
        except Exception as e:
            print(f"Error reading existing metadata: {e}. Starting fresh or with partial skip list.")

    offset = 0
    tracks_collected_this_run = 0

    while tracks_collected_this_run < TOTAL_TRACKS_TO_COLLECT:
        print(f"\nFetching tracks... Offset: {offset}, Collected this run: {tracks_collected_this_run}/{TOTAL_TRACKS_TO_COLLECT}")
        params = {
            "limit": TRACKS_PER_API_CALL,
            "offset": offset,
            "license_cc": TARGET_LICENSES_STRING,
            "hasimage": "true",
            "album_datebetween": "2000-01-01_2024-12-31", # Example: filter by release window
            "audioformat": "mp3", # Request mp3 audio
            # "imagesize": IMAGE_TARGET_WIDTH_PX, # Requesting specific size can be an option
            "order": "popularity_week" # Or 'random', 'releasedate_desc' for different sorting
        }
        
        api_response_data = make_api_request("/tracks", params)

        if not api_response_data or "results" not in api_response_data or not api_response_data["results"]:
            print("No more tracks found with current filters or API error. Exiting collection loop.")
            if api_response_data and "headers" in api_response_data:
                 print(f"API Response Headers (summary): {api_response_data['headers']}")
            break

        tracks_on_page = api_response_data["results"]
        
        for track_data in tracks_on_page:
            if tracks_collected_this_run >= TOTAL_TRACKS_TO_COLLECT:
                break

            track_id = str(track_data.get("id", ""))
            track_name = track_data.get("name", "Unknown Track")
            artist_name = track_data.get("artist_name", "Unknown Artist")
            album_name = track_data.get("album_name", "Unknown Album")
            license_cc = track_data.get("license_ccurl", "").split("/licenses/")[-1].split("/")[0] # Extract license type
            original_audio_duration = float(track_data.get("duration", 0))

            # Basic validation
            if not track_id or not track_data.get("album_image") or not track_data.get("audiodownload"):
                # Using "audiodownload" for potentially higher quality if available, fallback to "audio" for preview
                # print(f"Skipping track '{track_name}' (ID: {track_id}) due to missing essential data (ID, image, or audio URL).")
                continue

            if track_id in existing_track_ids:
                continue 

            print(f"\nProcessing Track ID: {track_id}, Name: {track_name}, Artist: {artist_name}")
            print(f"  License: {license_cc}, Duration: {original_audio_duration}s")

            audio_url = track_data.get("audiodownload", track_data.get("audio")) # Prefer full download, fallback to preview
            image_url = track_data.get("album_image") # Jamendo usually provides reasonably sized images

            audio_filename = f"{track_id}.mp3"
            image_filename = f"{track_id}.jpg"
            audio_snippet_output_path = os.path.join(AUDIO_SNIPPETS_DIR, audio_filename)
            image_output_path = os.path.join(COVER_IMAGES_DIR, image_filename)

            # Download and process audio
            print(f"  Downloading audio from: {audio_url}...")
            audio_bytes = download_file_content(audio_url)
            if not audio_bytes:
                print(f"  Failed to download audio for {track_id}. Skipping."); continue
            
            snippet_created, actual_snippet_len_s = create_audio_snippet(audio_bytes, audio_snippet_output_path, AUDIO_SNIPPET_DURATION_S)
            if not snippet_created:
                print(f"  Failed to create audio snippet for {track_id}. Skipping."); continue
            print(f"  Audio snippet created: {audio_snippet_output_path} ({actual_snippet_len_s:.2f}s)")

            # Download and process image
            print(f"  Downloading image from: {image_url}...")
            image_bytes = download_file_content(image_url)
            if not image_bytes:
                print(f"  Failed to download image for {track_id}. Skipping."); continue

            img_saved, orig_w, orig_h, saved_w, saved_h = resize_and_save_image(image_bytes, image_output_path, IMAGE_TARGET_WIDTH_PX)
            if not img_saved:
                print(f"  Failed to process image for {track_id}. Skipping."); continue
            print(f"  Image saved: {image_output_path} (Resized to {saved_w}x{saved_h} from {orig_w}x{orig_h})")

            # Record metadata
            metadata_entry = {
                "jamendo_track_id": track_id,
                "track_name": track_name,
                "artist_name": artist_name,
                "album_name": album_name,
                "audio_filename": audio_filename,
                "image_filename": image_filename,
                "jamendo_license_cc": license_cc,
                "original_audio_duration_s": original_audio_duration,
                "actual_snippet_duration_s": actual_snippet_len_s,
                "image_original_width_px": orig_w,
                "image_original_height_px": orig_h,
                "image_saved_width_px": saved_w,
                "image_saved_height_px": saved_h
            }
            append_to_metadata(metadata_entry, METADATA_FIELD_ORDER)
            existing_track_ids.add(track_id) # Add to set to avoid re-processing in this run if API returns duplicates
            tracks_collected_this_run += 1
            
            time.sleep(0.1) # Small delay per item processed

        if not tracks_on_page or len(tracks_on_page) < TRACKS_PER_API_CALL:
            print("Reached end of available tracks with current filters or API returned fewer than requested.")
            break 

        offset += TRACKS_PER_API_CALL
        time.sleep(API_REQUEST_DELAY_S) # Be polite to the API

    print(f"\n--- Jamendo Data Collection Finished ---")
    print(f"Collected {tracks_collected_this_run} new tracks in this session.")
    print(f"Total unique tracks in metadata file ({METADATA_FILE}): {len(existing_track_ids)}")
    print(f"Audio snippets are in: {AUDIO_SNIPPETS_DIR}")
    print(f"Cover images are in: {COVER_IMAGES_DIR}")
    