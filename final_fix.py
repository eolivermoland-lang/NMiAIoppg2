import shutil
import os
from pathlib import Path

# Ny nøyaktig sti
base_dir = Path(r'C:\Users\edu3650218\norgesgruppen\NMiAI2')
src_images = base_dir / 'NM_NGD_coco_dataset' / 'train' / 'images'
src_json = base_dir / 'NM_NGD_coco_dataset' / 'train' / 'annotations.json'

# Opprett den ferdige mappen (Kit)
kit_dir = base_dir / 'Colab_Training_Kit'
(kit_dir / 'images').mkdir(parents=True, exist_ok=True)
(kit_dir / 'labels').mkdir(parents=True, exist_ok=True)

print(f"Starter klargjøring i: {kit_dir}")

# 1. Last COCO for å lage labler (igjen for å være helt sikker på IDer)
import json
with open(src_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

image_map = {img['id']: img for img in data['images']}

print("Lager YOLO-labler...")
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_map: continue
    img = image_map[img_id]
    w, h = img['width'], img['height']
    bx, by, bw, bh = ann['bbox']
    label_file = kit_dir / 'labels' / f"{Path(img['file_name']).stem}.txt"
    with open(label_file, 'a') as f:
        f.write(f"{ann['category_id']} {(bx + bw/2)/w:.6f} {(by + bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")

# 2. Kopier bilder
print("Kopierer bilder...")
for img_file in src_images.iterdir():
    if img_file.is_file():
        shutil.copy(img_file, kit_dir / 'images' / img_file.name)

# 3. Lag data.yaml (med alle navnene)
names = {cat['id']: cat['name'] for cat in data['categories']}
yaml_content = "path: /content/dataset\ntrain: images\nval: images\n\nnames:\n"
for i in range(357):
    yaml_content += f"  {i}: {names.get(i, f'Product_{i}')}\n"

with open(kit_dir / 'data.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print("\n--- FERDIG! ---")
print(f"Alt ligger i: {kit_dir}")
print(f"1. Zip mappen 'Colab_Training_Kit' og send til kompisen din.")
print(f"2. Send også 'run.py' som ligger i {base_dir / 'run.py'}")
