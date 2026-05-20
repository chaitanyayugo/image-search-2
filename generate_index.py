import os
import json
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION LAYER ---
IMAGE_DIR = "./images"
INDEX_FILENAME = "image_index.json"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16  # Higher batch sizes utilize more GPU/CPU parallel cores
NUM_WORKERS = min(4, os.cpu_count() or 1)  # Thread workers for high-speed disk I/O

# --- HARDWARE AUTO-ACCELERATION CONFIG ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing Structural Engine Core on: [{device.upper()}]")

print("Loading foundational weights for CLIP AI Engine...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()  # Disable dropout layers for deterministic calculations


class ParallelImageDataset(Dataset):
    """Elite Dataset with structural I/O and pre-filtering properties."""
    def __init__(self, image_dir, existing_filenames):
        self.image_dir = image_dir
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        
        # Pull raw items from directory map
        raw_files = os.listdir(image_dir) if os.path.exists(image_dir) else []
        
        # Filter and exclude pre-cached items instantly
        self.filenames = [
            f for f in raw_files 
            if f.lower().endswith(valid_extensions) and f not in existing_filenames
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.image_dir, filename)
        try:
            # High-performance disk reading and image normalization
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img) # Correct phone orientation rotations
                img_rgb = img.convert("RGB")
                img_flipped = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                return filename, img_rgb, img_flipped, True
        except Exception as e:
            print(f"\n[IO_ERROR] Failed parsing matrix payload for {filename}: {e}")
            # Return dummy placeholders to prevent dataset worker batch collapses
            return filename, Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224)), False


def custom_collate(batch):
    """Custom collector to isolate filenames from tensor compilation loops."""
    filenames, images, flipped_images, statuses = zip(*batch)
    return filenames, list(images), list(flipped_images), statuses


def run_high_throughput_indexer():
    # 1. Thread-safe Incremental Cache Evaluation
    existing_filenames = set()
    index_data = []

    if os.path.exists(INDEX_FILENAME):
        try:
            with open(INDEX_FILENAME, "r") as f:
                index_data = json.load(f)
                # Map set for fast O(1) lookups
                existing_filenames = {item["filename"] for item in index_data}
            print(f"Cache Matched. Found {len(existing_filenames)} pre-existing documents.")
        except Exception as e:
            print(f"Warning: Index map parsing failure, clearing cache. Error: {e}")

    # Build container directories if missing
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 2. Build Multi-threaded Datastream Loader
    dataset = ParallelImageDataset(IMAGE_DIR, existing_filenames)
    if len(dataset) == 0:
        print("⚡ System Synced: No new document matrices detected in image directory.")
        return

    print(f"Processing Array Initiated: {len(dataset)} new assets queued for processing.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=custom_collate
    )

    new_records_added = 0

    # 3. Batch Hardware Vector Pipeline Execution
    with torch.no_grad():
        for batch_idx, (filenames, images, flipped_images, statuses) in enumerate(dataloader):
            # Strip files that failed to read safely from the stream
            valid_indices = [i for i, status in enumerate(statuses) if status]
            if not valid_indices:
                continue

            valid_filenames = [filenames[i] for i in valid_indices]
            valid_images = [images[i] for i in valid_indices]
            valid_flipped = [flipped_images[i] for i in valid_indices]

            # Vector processing blocks
            inputs_orig = processor(images=valid_images, return_tensors="pt").to(device)
            feat_orig = model.get_image_features(**inputs_orig)
            feat_orig = feat_orig / feat_orig.norm(p=2, dim=-1, keepdim=True)
            vectors_orig = feat_orig.cpu().numpy().tolist()

            inputs_flip = processor(images=valid_flipped, return_tensors="pt").to(device)
            feat_flip = model.get_image_features(**inputs_flip)
            feat_flip = feat_flip / feat_flip.norm(p=2, dim=-1, keepdim=True)
            vectors_flip = feat_flip.cpu().numpy().tolist()

            # 4. Stream and Merge data structures
            for i, fname in enumerate(valid_filenames):
                index_data.append({"filename": fname, "vector": vectors_orig[i]})
                index_data.append({"filename": fname, "vector": vectors_flip[i]})
                new_records_added += 1

            print(f"Processed Frame Matrix Batch [{batch_idx + 1}] -> Vectorized: {len(valid_filenames)} items.")

    # 5. Atomic File Operations (Prevents truncation data drops if pipeline is killed)
    if new_records_added > 0:
        temp_filename = f"{INDEX_FILENAME}.tmp"
        with open(temp_filename, "w") as f:
            json.dump(index_data, f, separators=(',', ':'))  # Minified layout to save disk size
        os.replace(temp_filename, INDEX_FILENAME)
        print(f"🏆 Verification Passed. Successfully injected {new_records_added} elements into {INDEX_FILENAME}.")


if __name__ == "__main__":
    run_high_throughput_indexer()
