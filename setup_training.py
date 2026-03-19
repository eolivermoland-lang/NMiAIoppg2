import json
import os
import shutil
from pathlib import Path

# Paths
root = Path(r'C:\Users\edu3650218\norgesgruppen')
coco_json = root / 'NM_NGD_coco_dataset' / 'train' / 'annotations.json'
images_src = root / 'NM_NGD_coco_dataset' / 'train' / 'images'
output_dir = root / 'yolo_training_ready'

# 1. Create directory structure
labels_dir = output_dir / 'labels'
images_dest = output_dir / 'images'
labels_dir.mkdir(parents=True, exist_ok=True)
images_dest.mkdir(parents=True, exist_ok=True)

# 2. Load COCO Data
with open(coco_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Map image IDs to filenames and sizes
image_map = {img['id']: img for img in data['images']}

# 3. Convert to YOLO Format
print("Konverterer annoteringer til YOLO-format...")
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_map:
        continue
    
    img = image_map[img_id]
    w, h = img['width'], img['height']
    
    # YOLO: class_id x_center y_center width height (normalized)
    bx, by, bw, bh = ann['bbox']
    x_center = (bx + bw / 2) / w
    y_center = (by + bh / 2) / h
    wn = bw / w
    hn = bh / h
    
    cat_id = ann['category_id']
    
    label_file = labels_dir / f"{Path(img['file_name']).stem}.txt"
    with open(label_file, 'a') as f:
        f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {wn:.6f} {hn:.6f}\n")

# 4. Copy Images (optional, but good for local check - for Colab we zip them)
# print("Kopierer bilder...")
# for img_file in images_src.iterdir():
#     shutil.copy(img_file, images_dest / img_file.name)

# 5. Create data.yaml
print("Lager data.yaml...")
names = {cat['id']: cat['name'] for cat in data['categories']}
yaml_lines = [
    "path: /content/dataset", # Standard path for Colab
    "train: images",
    "val: images",
    "",
    "names:"
]
for i in range(357): # Ensure all IDs are included
    name = names.get(i, f"Product_{i}")
    yaml_lines.append(f"  {i}: {name}")

with open(output_dir / 'data.yaml', 'w', encoding='utf-8') as f:
    f.write('\n'.join(yaml_lines))

print(f"\nSuksess! Alt er klart i: {output_dir}")
print("Neste steg: Zip 'images', 'labels' og 'data.yaml' og send til Colab.")
