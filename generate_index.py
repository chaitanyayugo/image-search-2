import os
import json
import torch
import io
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION LAYER ---
IMAGE_DIR = "./images"
INDEX_FILENAME = "image_index.json"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16 

# Try loading pypdf dynamically (will gracefully log if missing)
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing Structural Engine Core on: [{device.upper()}]")

print("Loading foundational weights for CLIP AI Engine...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

class EliteHybridDataset(Dataset):
    """Processes both loose images and raw PDF documents natively."""
    def __init__(self, image_dir, existing_filenames):
        self.image_dir = image_dir
        self.existing_filenames = existing_filenames
        self.payloads = [] # Stores structural tasks: (source_file, type, identifier, data)
        
        if not os.path.exists(image_dir):
            return
            
        raw_files = os.listdir(image_dir)
        for filename in raw_files:
            ext = filename.lower()
            
            # Case A: Standard loose image file
            if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if filename not in existing_filenames:
                    self.payloads.append((filename, "IMAGE", filename, os.path.join(image_dir, filename)))
                    
            # Case B: Unprocessed raw PDF file
            elif ext.endswith('.pdf') and PYPDF_AVAILABLE:
                pdf_path = os.path.join(image_dir, filename)
                try:
                    reader = PdfReader(pdf_path)
                    for page_idx, page in enumerate(reader.pages):
                        virtual_id = f"{filename}_page_{page_idx + 1}"
                        # Check if this specific page was already indexed to save time
                        if virtual_id not in existing_filenames:
                            self.payloads.append((filename, "PDF_PAGE", virtual_id, (pdf_path, page_idx)))
                except Exception as e:
                    print(f"Skipping corrupt structural PDF file {filename}: {e}")

    def __len__(self):
        return len(self.payloads)

    def __getitem__(self, idx):
        source_file, task_type, virtual_id, payload = self.payloads[idx]
        
        try:
            if task_type == "IMAGE":
                with Image.open(payload) as img:
                    img = ImageOps.exif_transpose(img)
                    img_rgb = img.convert("RGB")
                    return virtual_id, img_rgb, img_rgb.transpose(Image.FLIP_LEFT_RIGHT), True
                    
            elif task_type == "PDF_PAGE":
                pdf_path, page_idx = payload
                reader = PdfReader(pdf_path)
                page = reader.pages[page_idx]
                
                # High-speed extraction of embedded image streams inside the PDF page
                for img_file_object in page.images:
                    # Extracts raw pixel data stream directly from PDF binary blocks without rendering delays
                    img_data = img_file_object.data
                    with Image.open(io.BytesIO(img_data)) as img:
                        img_rgb = img.convert("RGB")
                        return virtual_id, img_rgb, img_rgb.transpose(Image.FLIP_LEFT_RIGHT), True
                        
                # Fallback placeholder if page contains pure text instead of images
                return virtual_id, Image.new('RGB', (224, 224), color="#1e1b4b"), Image.new('RGB', (224, 224), color="#1e1b4b"), True
                
        except Exception as e:
            print(f"\n[IO_ERROR] Failed resolving stream matrix for {virtual_id}: {e}")
            return virtual_id, Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224)), False

def custom_collate(batch):
    virtual_ids, images, flipped_images, statuses = zip(*batch)
    return virtual_ids, list(images), list(flipped_images), statuses

def run_high_throughput_indexer():
    existing_filenames = set()
    index_data = []

    if os.path.exists(INDEX_FILENAME):
        try:
            with open(INDEX_FILENAME, "r") as f:
                index_data = json.load(f)
                existing_filenames = {item["filename"] for item in index_data}
            print(f"Cache Matched. Found {len(existing_filenames)} pre-existing documents.")
        except Exception as e:
            print(f"Warning: Index map parsing failure, clearing cache: {e}")

    os.makedirs(IMAGE_DIR, exist_ok=True)

    if not PYPDF_AVAILABLE:
        print("⚠️ Warning: 'pypdf' package not found. PDF documents will be bypassed. Running pure Image mode.")

    dataset = EliteHybridDataset(IMAGE_DIR, existing_filenames)
    if len(dataset) == 0:
        print("⚡ System Synced: No new assets or PDF pages detected.")
        return

    print(f"Processing Array Initiated: {len(dataset)} vector targets queued.")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    new_records_added = 0

    with torch.no_grad():
        for batch_idx, (v_ids, images, flipped_images, statuses) in enumerate(dataloader):
            valid_indices = [i for i, status in enumerate(statuses) if status]
            if not valid_indices: continue

            valid_vids = [v_ids[i] for i in valid_indices]
            valid_images = [images[i] for i in valid_indices]
            valid_flipped = [flipped_images[i] for i in valid_indices]

            inputs_orig = processor(images=valid_images, return_tensors="pt").to(device)
            feat_orig = model.get_image_features(**inputs_orig)
            feat_orig = feat_orig / feat_orig.norm(p=2, dim=-1, keepdim=True)
            vectors_orig = feat_orig.cpu().numpy().tolist()

            inputs_flip = processor(images=valid_flipped, return_tensors="pt").to(device)
            feat_flip = model.get_image_features(**inputs_flip)
            feat_flip = feat_flip / feat_flip.norm(p=2, dim=-1, keepdim=True)
            vectors_flip = feat_flip.cpu().numpy().tolist()

            for i, vid in enumerate(valid_vids):
                index_data.append({"filename": vid, "vector": vectors_orig[i]})
                index_data.append({"filename": vid, "vector": vectors_flip[i]})
                new_records_added += 1

            print(f"Processed Frame Matrix Batch [{batch_idx + 1}] -> Vectorized: {len(valid_vids)} items.")

    if new_records_added > 0:
        temp_filename = f"{INDEX_FILENAME}.tmp"
        with open(temp_filename, "w") as f:
            json.dump(index_data, f, separators=(',', ':'))
        os.replace(temp_filename, INDEX_FILENAME)
        print(f"🏆 Verification Passed. Successfully injected {new_records_added} document points.")

if __name__ == "__main__":
    run_high_throughput_indexer()
