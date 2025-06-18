# Audio-Conditioned Latent Diffusion for Album Cover Artwork Generation

This project, developed for the "Multimodal Machine Learning" course (Spring 2025, IIIS, Tsinghua University), explores generating album cover artwork from audio. It implements and compares several architectures for conditioning a Stable Diffusion model with CLAP audio embeddings, including a baseline, a simple MLP projection, and an advanced MLP projection with optional ControlNet guidance.

## Key Features

- **Multiple Architectures**: Train and infer with three different audio projection layers:
    1.  **Baseline**: A simple linear projection.
    2.  **Simple MLP**: A two-layer MLP with a ReLU activation.
    3.  **Advanced MLP**: A more robust MLP with LayerNorm and ReLU.
- **ControlNet Integration**: The advanced model supports ControlNet, allowing for additional image-based structural guidance (e.g., from Canny edges, depth maps).
- **Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) to fine-tune only a fraction of the UNet's parameters, making training accessible.
- **Flexible Inference**: Generate single images or process entire folders in batches.
- **Comprehensive Workflow**: Includes scripts for data collection, feature extraction, training, and inference.

## Project Structure

The repository is organized to separate the different model variants.

```
.
├── .env                            # (User-created) For storing API keys
├── .gitignore
├── environment.yml                 # Conda environment file
├── requirements.txt                # Pip requirements file
│
├── jamendo_collector_with_ganres.py # Script to download audio/images from Jamendo
├── extract_and_embed.py            # Alternative script to download and create embeddings
├── clap_feature_extractor.py       # Script to create embeddings from existing audio
│
├── without_mlp_baseline/           # --- Variant 1: Baseline Model ---
│   ├── train_lora_baseline.py
│   ├── generate_artwork_baseline.py
│   └── generate_artwork_folder_baseline.py
│
├── with_simple_mlp/                # --- Variant 2: Simple MLP Model ---
│   ├── train_lora_simple_mlp.py
│   ├── generate_artwork_simple_mlp.py
│   └── generate_artwork_folder_simple_mlp.py
│
├── train_lora_advanced_mlp.py      # --- Variant 3: Advanced MLP Model ---
├── generate_artwork_advanced_mlp_controlnet.py
├── generate_artwork_folder_advanced_mlp_controlnet.py
│
├── art-from-audio-dataset/         # (Gitignored) For storing your dataset
│   ├── audio_snippets/             # .mp3 or .wav audio files
│   ├── cover_images/               # .jpg or .png image files
│   └── audio_embeddings/           # (Generated) .pt CLAP embeddings
│
└── lora_*_output/                  # (Gitignored) Saved model weights from training
```

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd art-from-audio
```

### 2. Install Dependencies

It is highly recommended to use a virtual environment.

**Using pip:**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

# Install PyTorch with your specific CUDA version first for best results
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
```

### 3. Set Up API Key (for Data Collection)

If you plan to use the `jamendo_collector_with_ganres.py` script, you need a Jamendo API key.

1.  Create a file named `.env` in the project's root directory.
2.  Add your API key to the file like this:
    ```
    JAMENDO_API_KEY=your_api_key_here
    ```

## Full Workflow

### Step 1: Get Data

You can either use the provided script to download data or use your own audio/image pairs.

**Option A: Download from Jamendo**
Run the collector script. It will create the `art-from-audio-dataset` directory and populate `audio_snippets` and `cover_images`.
```bash
python jamendo_collector_with_ganres.py
```
**Prerequisite**: This script uses `pydub` to process audio, which requires **FFmpeg**. Please ensure it is installed and accessible in your system's PATH.
- **Windows**: Download from the [official FFmpeg site](https://ffmpeg.org/download.html) and add the `bin` folder to your PATH.
- **Linux (Ubuntu/Debian)**: `sudo apt-get install ffmpeg`
- **macOS (with Homebrew)**: `brew install ffmpeg`

*Note: You can configure genres, number of tracks, etc., inside the script.*

**Option B: Use Your Own Data**
1.  Create the directories: `mkdir -p art-from-audio-dataset/audio_snippets art-from-audio-dataset/cover_images`
2.  Place your audio files (`.mp3`, `.wav`) in `audio_snippets/`.
3.  Place your image files (`.jpg`, `.png`) in `cover_images/`.
4.  **Crucially**, ensure audio and image filenames match (e.g., `my_song.mp3` and `my_song.jpg`).

### Step 2: Extract Audio Embeddings

Before training, you must process your audio files to create CLAP embeddings.

```bash
python clap_feature_extractor.py
```
This script will read from `audio_snippets/` and save `.pt` embedding files into `audio_embeddings/`. This step is required for all training variants.

### Step 3: Train a Model

Choose one of the three variants to train. Training scripts are configured by editing the constants at the top of the file (e.g., `LEARNING_RATE`, `BATCH_SIZE`).

- **To train the Advanced MLP model (Recommended):**
  ```bash
  python train_lora_advanced_mlp.py
  ```
  *Outputs will be saved in `lora_advanced_mlp_output/`.*

- **To train the Simple MLP model:**
  ```bash
  python with_simple_mlp/train_lora_simple_mlp.py
  ```
  *Outputs will be saved in `lora_mlp_output/`.*

- **To train the Baseline model:**
  ```bash
  python without_mlp_baseline/train_lora_baseline.py
  ```
  *Outputs will be saved in `lora_output/`.*

### Step 4: Generate Artwork (Inference)

Use the corresponding generation script for your trained model. The **Advanced MLP** scripts are the most feature-rich and are detailed below.

#### **Advanced MLP Model Generation**

These scripts support all features, including ControlNet.

**A. Generate a Single Image**

```bash
python generate_artwork_advanced_mlp_controlnet.py \
    --audio_file_path "path/to/your/song.mp3" \
    --output_image_path "path/to/save/artwork.png" \
    --checkpoint_path "lora_advanced_mlp_output/final_lora_advanced_mlp_weights.safetensors" \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --mixed_precision_inference fp16
```

**B. Generate a Single Image with ControlNet**

To add structural guidance, provide a condition image and specify a ControlNet model.

```bash
python generate_artwork_advanced_mlp_controlnet.py \
    --audio_file_path "path/to/your/song.mp3" \
    --output_image_path "path/to/save/artwork_canny.png" \
    --checkpoint_path "lora_advanced_mlp_output/final_lora_advanced_mlp_weights.safetensors" \
    --condition_image_path "path/to/your/canny_edges.jpg" \
    --controlnet_model_name "lllyasviel/sd-controlnet-canny" \
    --controlnet_conditioning_scale 0.8
```

**C. Generate for a Folder of Audio Files**

This will process all audio files in a directory, saving the outputs to a new folder.

```bash
python generate_artwork_folder_advanced_mlp_controlnet.py \
    --audio_folder_path "path/to/your/audio_folder" \
    --checkpoint_path "lora_advanced_mlp_output/final_lora_advanced_mlp_weights.safetensors" \
    --num_inference_steps 30
```
*(You can also add the ControlNet arguments to apply the same condition to all images).*

#### **Advanced Inference Parameters**

| Parameter                       | Description                                                                                      |
|---------------------------------|--------------------------------------------------------------------------------------------------|
| `--audio_file_path`             | Path to the input audio file (`.mp3` or `.wav`).                                                 |
| `--audio_folder_path`           | Path to a folder of audio files for batch generation.                                            |
| `--output_image_path`           | Path to save the single generated image.                                                         |
| `--output_folder_path`          | (Optional) Path to save batch-generated images. Defaults to a new folder next to the input.      |
| `--checkpoint_path`             | **Required.** Path to the trained `.safetensors` model weights.                                  |
| `--lora_rank`                   | The LoRA rank used during training. **Must match.** (Default: `4`).                              |
| `--num_inference_steps`         | Number of diffusion steps. (Default: `30`).                                                      |
| `--guidance_scale`              | Classifier-Free Guidance scale. Higher values adhere more to the audio. (Default: `7.5`).        |
| `--seed`                        | (Optional) A number for the random seed to ensure reproducible results.                          |
| `--mixed_precision_inference`   | Use `"fp16"`, `"bf16"`, or `"no"` for inference. Can speed up generation on GPUs. (Default: `"no"`). |
| `--condition_image_path`        | (Optional) Path to an image for ControlNet conditioning.                                         |
| `--controlnet_model_name`       | (Optional) Hugging Face name of the ControlNet model. Defaults to Canny if not set.              |
| `--controlnet_conditioning_scale`| How much to weight the ControlNet conditioning. (Default: `1.0`).                                |
| `--base_model_name`             | (Optional) Base Stable Diffusion model if not the default.                                       |
| `--clap_model_name`             | (Optional) CLAP model if not the default.                                                        |

---

*This README provides a comprehensive guide to setting up and running the project. For more specific details, please refer to the comments and argument parsers within each Python script.*