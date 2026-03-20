import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Mappe med testbilder")
    parser.add_argument("--output", required=True, help="Sti til predictions.json")
    args = parser.parse_args()

    # 1. Last modellen (leter etter best.pt først, så fallback til x)
    model_path = Path("best.pt")
    if not model_path.exists():
        model_path = Path("yolov8x.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    
    predictions = []

    input_path = Path(args.input)
    if not input_path.exists():
        return

    # 2. Kjør prediksjon med MAKSIMAL nøyaktighet
    for img_file in sorted(input_path.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        
        try:
            # Henter image_id fra filnavn (img_00042.jpg -> 42)
            image_id = int(img_file.stem.split("_")[-1])
        except (ValueError, IndexError):
            continue

        # Inference med High Resolution og Half-precision
        # Vi bruker imgsz=1280 for å matche treningen og finne de minste produktene
        results = model.predict(
            str(img_file), 
            device=device, 
            imgsz=1280, 
            verbose=False, 
            conf=0.15,      # Lav threshold for å finne ALLE bokser (viktig for score)
            iou=0.45,       # Standard NMS threshold
            half=True,      # Sparer minne på L4 GPU
            augment=True    # TTA: Sjekker bildet flere ganger (flippet/skalert) for 100% presisjon!
        )

        for r in results:
            if r.boxes is None:
                continue
            
            # Behandle på CPU for å unngå minnelekasje på GPU
            boxes = r.boxes.cpu()
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                
                score = float(box.conf[0])
                category_id = int(box.cls[0])

                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                    "score": round(score, 3)
                })

    # 3. Lagre resultatene
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
