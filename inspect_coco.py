import json

path = r'C:\Users\edu3650218\norgesgruppen\NM_NGD_coco_dataset\train\annotations.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Images: {len(data['images'])}")
print(f"Annotations: {len(data['annotations'])}")
print(f"Categories: {len(data['categories'])}")
print("\nFirst 10 Categories:")
for cat in data['categories'][:10]:
    print(f"  ID: {cat['id']}, Name: {cat['name']}")

# Check some image paths
print("\nFirst 5 Images:")
for img in data['images'][:5]:
    print(f"  ID: {img['id']}, File: {img['file_name']}")
