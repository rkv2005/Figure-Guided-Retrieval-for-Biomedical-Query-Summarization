import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import open_clip
from transformers import AutoTokenizer
from tqdm.auto import tqdm

STORE_ROOT = Path("/content/drive/MyDrive/ffhrag_store")
FIGURES_JSONL = STORE_ROOT / "figures.jsonl"

MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

print("Loading BiomedCLIP via open_clip...")
try:
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✅ Loaded on {device}")
    print(f"   Model architecture: {type(model).__name__}")
except Exception as e:
    print(f"❌ Error: {e}")
    raise


def load_images(image_paths):
    """Load PIL Images from paths."""
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {e}")
    return images, valid_paths


def encode_images_only(image_paths):
    """Encode images using BiomedCLIP."""
    images, valid_paths = load_images(image_paths)

    if not images:
        return np.array([])

    # Preprocess images (resize, normalize, etc.)
    image_tensors = torch.stack([preprocess(img) for img in images]).to(device)

    with torch.no_grad():
        img_embs = model.encode_image(image_tensors)

    # L2 normalize
    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
    return img_embs.cpu().numpy()


def encode_text_only(texts):
    """Encode text using BiomedCLIP."""
    # Tokenize text
    text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_embs = model.encode_text(text_inputs['input_ids'])

    # L2 normalize
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    return text_embs.cpu().numpy()
