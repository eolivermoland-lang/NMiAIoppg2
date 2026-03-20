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

    # 1. Last modellen
    model_path = Path("best.pt")
    if not model_path.exists():
        # Fallback til Large eller X hvis best.pt ikke finnes
        model_path = Path("yolov8l.pt")
        if not model_path.exists():
            model_path = Path("yolov8x.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    
    predictions = []

    input_path = Path(args.input)
    if not input_path.exists():
        return

    # 2. Kjør prediksjon
    # Vi bruker imgsz=1024 for å balansere mellom nøyaktighet og 8GB minne-limit.
    for img_file in sorted(input_path.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        
        try:
            # Hent image_id (f.eks img_00042.jpg -> 42)
            image_id = int(img_file.stem.split("_")[-1])
        except (ValueError, IndexError):
            continue

        # Inference med Half-precision (viktig for L4 GPU og minne)
        results = model.predict(
            str(img_file), 
            device=device, 
            imgsz=1024, 
            verbose=False, 
            conf=0.2,       # Litt lavere threshold for å fange opp flere produkter (score=0.7 på detection)
            iou=0.45, 
            half=True
        )

        for r in results:
            if r.boxes is None:
                continue
            
            # Flytt til CPU for behandling
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
