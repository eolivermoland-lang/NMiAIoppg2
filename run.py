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

    # 1. Last modellen (denne må hete 'best.pt' og ligge i samme mappe)
    model_path = Path("best.pt")
    if not model_path.exists():
        # Fallback for testing hvis best.pt ikke er klar
        model_path = Path("yolov8x.pt") 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Map_location håndteres automatisk av Ultralytics
    model = YOLO(str(model_path))
    
    predictions = []

    # 2. Gå gjennom alle bildene i input-mappen
    input_path = Path(args.input)
    if not input_path.exists():
        return

    for img_file in sorted(input_path.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        
        # Hent image_id fra filnavnet (f.eks. img_00042.jpg -> 42)
        try:
            # Splitter på _ og tar siste del før .jpg
            parts = img_file.stem.split("_")
            image_id = int(parts[-1])
        except (ValueError, IndexError):
            continue

        # 3. Kjør rekognosering (Inference)
        # Bruker imgsz=1280 for best mulig resultat på små produkter
        results = model(str(img_file), device=device, imgsz=1280, verbose=False, conf=0.05)

        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                # xyxy er [x1, y1, x2, y2] i piksler
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

    # 4. Lagre resultatene til predictions.json
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
