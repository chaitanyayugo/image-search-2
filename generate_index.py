import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the lightweight pre-trained AI embedding engine
print("Initializing CLIP AI Engine...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

image_dir = "./images"
index_filename = "image_index.json"
index_data = []
existing_filenames = set()

# 2. Incremental Cache Logic: Load existing index data if it exists
if os.path.exists(index_filename):
    try:
        with open(index_filename, "r") as f:
            index_data = json.load(f)
            # Track images that have already been converted to vectors
            existing_filenames = {item["filename"] for item in index_data}
        print(f"Loaded existing index cache. Found {len(existing_filenames)} pre-indexed images.")
    except Exception as e:
        print(f"Warning: Failed to parse existing index, building fresh. Error: {e}")
        index_data = []

# Ensure the image folder exists before looping
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Created missing directory: {image_dir}. Please place target images here.")

new_images_count = 0

# 3. Process ONLY new assets found in the repository
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        # Skip if this image has already been calculated in a previous workflow run
        if filename in existing_filenames:
            continue
            
        img_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            
            # --- EXTRACT VARIATION 1: Original Image Vector ---
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                vector_orig = feat.numpy().flatten().tolist()
            
            # --- EXTRACT VARIATION 2: Horizontal Mirror Vector (Fixes flipped products) ---
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            inputs_flipped = processor(images=flipped_image, return_tensors="pt")
            with torch.no_grad():
                feat_flipped = model.get_image_features(**inputs_flipped)
                feat_flipped = feat_flipped / feat_flipped.norm(p=2, dim=-1, keepdim=True)
                vector_flipped = feat_flipped.numpy().flatten().tolist()

            # Append both vector options to the index array pointing to the same file
            index_data.append({"filename": filename, "vector": vector_orig})
            index_data.append({"filename": filename, "vector": vector_flipped})
            
            new_images_count += 1
            print(f"Successfully indexed new asset variants for: {filename}")
            
        except Exception as e:
            print(f"Skipping corrupt or unreadable image file {filename}. Error: {e}")

# 4. Save and commit updates only if new images were added
if new_images_count > 0:
    with open(index_filename, "w") as f:
        json.dump(index_data, f)
    print(f"Index updated! Added {new_images_count} new images to {index_filename}.")
else:
    print("No new image additions detected. Index file remains identical.")
