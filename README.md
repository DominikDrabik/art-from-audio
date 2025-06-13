# Audio-Conditioned Latent Diffusion for Album Cover Artwork Generation

This project was developed as part of the "Multimodal Machine Learning" course (Spring 2025) taught by Prof. Hang Zhao at the Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University. It explores the generation of album cover artwork conditioned on audio input using latent diffusion models.

The system leverages pre-trained models for audio feature extraction (CLAP) and image generation (Stable Diffusion), connecting them via a fine-tuning process that incorporates an audio embedding projection layer and Low-Rank Adaptation (LoRA) for efficient training.

## Architecture and Technical Details

The core pipeline of this project can be summarized as follows:

1.  **Audio Input**: An audio file (e.g., `.mp3`) serves as the primary input.
2.  **Audio Feature Extraction**: The CLAP model (`laion/larger_clap_music_and_speech`) processes the audio to generate a fixed-size (512-dimensional) embedding that captures its semantic and acoustic content. This is handled by `clap_feature_extractor.py`.
3.  **Conditional Image Generation**: A pre-trained Stable Diffusion v1.5 model (`runwayml/stable-diffusion-v1-5`) is fine-tuned to generate images based on these audio embeddings.
    *   **Audio Embedding Projection**: A crucial step involves projecting the 512-dimensional CLAP audio embedding to the 768-dimensional space expected by the Stable Diffusion UNet's cross-attention mechanism. This is achieved using a trainable `torch.nn.Linear(512, 768)` layer. Both this projection layer and the LoRA adapters are trained.
    *   **Low-Rank Adaptation (LoRA)**: To efficiently fine-tune the large Stable Diffusion UNet, LoRA is applied. Specifically, LoRA adapters (with a configurable rank, e.g., 4 or 8) are added to the UNet's attention-related layers (e.g., `to_q`, `to_k`, `to_v`, `to_out.0`, `proj_in`, `proj_out`). Only these LoRA parameters and the audio projection layer's parameters are updated during training, keeping the base Stable Diffusion model weights frozen.
    *   **Training (`train_lora_diffusion.py`)**:
        *   The `AudioImageDataset` class prepares (audio embedding, image) pairs for training. Images are resized and normalized.
        *   Training utilizes mixed precision (preferring `bfloat16` if available, otherwise `float16` on CUDA) for speed and memory efficiency.
        *   The loss is computed as the MSE between the UNet's predicted noise and the actual noise added to the latents.
        *   The AdamW optimizer is used.
        *   Checkpoints containing the trained UNet LoRA weights and the audio projection layer weights are saved periodically (e.g., as `.safetensors` files).
4.  **Inference (`generate_artwork.py`)**:
    *   For a new audio input, its CLAP embedding is extracted and then projected using the trained audio projection layer.
    *   The base Stable Diffusion model components (VAE, UNet, text encoder, tokenizer, scheduler) are loaded. The trained LoRA adapters are applied to the UNet, and their weights (along with the audio projection layer's weights) are loaded from the saved checkpoint.
    *   The projected audio embedding serves as the conditional input (`prompt_embeds`) to the Stable Diffusion pipeline.
    *   **Classifier-Free Guidance (CFG)**: This technique is employed to enhance the adherence of the generated image to the conditional audio input.
        *   **Mechanism**: When CFG is active (controlled by `guidance_scale > 1.0`), the model internally contrasts two predictions at each diffusion step:
            1.  A *conditional* prediction based on your audio embedding (`prompt_embeds`).
            2.  An *unconditional* prediction, generated as if there were no specific audio input. In this project, these unconditional embeddings (`negative_prompt_embeds`) are derived from an empty text prompt processed by the Stable Diffusion's text encoder.
        *   **Guidance Scale**: The `guidance_scale` parameter (default is 7.5 in `generate_artwork.py`) determines how strongly the final generation is "nudged" away from the unconditional prediction towards the audio-conditioned one. Values closer to 1.0 reduce or disable CFG's effect.
        *   **Embedding Shape Adaptation**: Since the audio embedding is a single vector representing the whole audio, and the unconditional text embeddings have a sequence length (e.g., 77 for CLIP), the audio-derived `prompt_embeds` are repeated along the sequence dimension to match the shape of the `negative_prompt_embeds`. This ensures compatibility for the CFG process and is automatically handled by `generate_artwork.py` when CFG is active.
    *   The pipeline then generates an image through the reverse diffusion process.

## Project Structure

```
.
├── art-from-audio-dataset/       # (Gitignored) Root for dataset (user-created)
│   ├── audio_snippets/           # Input .mp3 audio files (user-provided)
│   ├── cover_images/             # Corresponding .jpg cover images (user-provided)
│   └── audio_embeddings/         # (Generated) .pt CLAP embeddings
├── lora_output/                  # (Gitignored) Saved LoRA & projection weights (generated by training)
├── .gitignore                    # Specifies intentionally untracked files
├── clap_feature_extractor.py     # Script to extract CLAP embeddings from audio
├── environment.yml               # Conda environment specification
├── generate_artwork.py           # Script to generate artwork using trained model
├── jamendo_collector.py          # Example script for collecting audio-cover pairs
├── README.md                     # This file
├── requirements.txt              # pip requirements (for non-Conda users or Colab)
└── train_lora_diffusion.py       # Script to train the LoRA-conditioned diffusion model
```


## Setup

### 1. Clone the Repository

```bash
git clone git@github.com:nikitabutsch/art-from-audio.git
cd art-from-audio
```

### 2. Create Conda Environment

It is highly recommended to use Conda for managing Python dependencies.

```bash
conda env create -f environment.yml
conda activate art-from-audio
```

**Alternatively (without Conda):**

If you are not using Conda, you can create a standard Python virtual environment and install packages using `requirements.txt`.


```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX # Replace cuXXX with your CUDA version e.g. cu118 or cu121
pip install -r requirements.txt
```
*(The `pytorch` entry in `requirements.txt` is often generic; explicit installation as shown above is more reliable, especially for CUDA.)*

## Data Preparation

The model requires a dataset of audio files and their corresponding cover images.

1.  **Create Dataset Directories**:
    In the project's root directory, create the necessary folder structure:
    ```bash
    mkdir -p art-from-audio-dataset/audio_snippets art-from-audio-dataset/cover_images
    ```
    *(These directories are gitignored, as the dataset is typically large and user-specific).*

2.  **Add Audio Files**:
    Place your audio files (e.g., `.mp3`, `.wav`) into the `art-from-audio-dataset/audio_snippets/` directory.

3.  **Add Cover Images**:
    Place the corresponding cover images (e.g., `.jpg`, `.png`) into the `art-from-audio-dataset/cover_images/` directory.

    **Crucial Naming Convention**: The base filename (without the extension) of an audio file and its corresponding image **must match exactly**.
    *   Example: `song_01.mp3` must have its cover image as `song_01.jpg`.

4.  **(Optional) Data Collection with `jamendo_collector.py`**:
    The `jamendo_collector.py` script is provided as an example of how to download audio-image pairs from the Jamendo music platform.
    *   Requires a Jamendo API key (see script comments for setup).
    *   It downloads audio snippets and cover art, saving them to the correct directories and creating a `metadata.csv`.
    *   Requires 
    *   You may need to adapt this script or use other methods for your data collection needs. Ensure collected data follows the naming convention.

## Running the Scripts

### 1. Extract Audio Embeddings (`clap_feature_extractor.py`)

This script processes all audio files in `art-from-audio-dataset/audio_snippets/`, extracts their embeddings using the CLAP model (`laion/larger_clap_music_and_speech` by default), and saves these embeddings as PyTorch tensor files (`.pt`) in `art-from-audio-dataset/audio_embeddings/`.

```bash
python clap_feature_extractor.py
```
*   Configuration (model ID, directories) is at the top of the script.
*   This step is **required** before training.

### 2. Train the Diffusion Model with LoRA (`train_lora_diffusion.py`)

This script fine-tunes the Stable Diffusion UNet using the extracted audio embeddings and their corresponding cover images.

```bash
python train_lora_diffusion.py
```
Key configurable parameters are located at the top of `train_lora_diffusion.py`:
*   `MODEL_NAME`: Base Stable Diffusion model (default: `runwayml/stable-diffusion-v1-5`).
*   `AUDIO_EMBEDDINGS_DIR`, `IMAGE_DIR`, `OUTPUT_DIR`.
*   `LEARNING_RATE`, `LORA_RANK`, `BATCH_SIZE`, `NUM_TRAIN_EPOCHS`.
*   `IMAGE_RESOLUTION`, `MIXED_PRECISION` (e.g., "bf16", "fp16", "no").
*   `SAVE_MODEL_EPOCHS`: Frequency for saving checkpoints.

The script saves the trained UNet LoRA weights and the audio projection layer weights into a single `.safetensors` file in the `lora_output/` directory (e.g., `final_lora_and_projection_weights.safetensors` and epoch-based checkpoints).

### 3. Generate Artwork - Inference (`generate_artwork.py`)

This script uses a trained checkpoint (containing the UNet LoRA and audio projection weights) to generate an image conditioned on a new input audio file.

**Example Usage:**
```bash
python generate_artwork.py \
    --audio_file_path "path/to/your/input_audio.mp3" \
    --checkpoint_path "lora_output/final_lora_and_projection_weights.safetensors" \
    --output_image_path "generated_artwork.png" \
    --lora_rank 4  # Must match the LORA_RANK used during training
```

**Key Command-Line Arguments:**
*   `--audio_file_path` (required): Path to the input audio file.
*   `--checkpoint_path`: Path to the `.safetensors` checkpoint file.
*   `--output_image_path`: Path to save the generated image.
*   `--lora_rank`: The LoRA rank used during training (important for correctly configuring the LoRA layers before loading weights).
*   `--base_model_name`: Base Stable Diffusion model (if different from default).
*   `--clap_model_name`: CLAP model for embedding (if different from default).
*   `--num_inference_steps`: Number of diffusion steps (default: 50).
*   `--guidance_scale`: Classifier-Free Guidance scale (default: 7.5). Values `_<= 1.0_` effectively disable CFG or reduce its impact.
*   `--seed`: Optional random seed for reproducible image generation.
*   `--mixed_precision_inference`: (Optional, e.g., "fp16", "bf16") Enable mixed precision for inference models on CUDA for potential speedup.

## Pre-trained Models from Hugging Face

The scripts automatically download the necessary pre-trained models from the Hugging Face Hub:
*   **CLAP Model**: Default is `laion/larger_clap_music_and_speech`.
*   **Stable Diffusion Model**: Default is `runwayml/stable-diffusion-v1-5` (including VAE, UNet, text encoder, tokenizer, scheduler).

An active internet connection is required the first time each of these models is downloaded by the scripts. They will be cached locally by the `transformers` and `diffusers` libraries for subsequent runs.

---

*This README provides a guide to understanding, setting up, and running the project. You may need to adjust paths, parameters, or data collection methods based on your specific requirements and computational resources.* 
