import torch
from ultralytics import YOLO
from pathlib import Path

# Sjekk GPU
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Trener på: {device}")

# Finn data.yaml (den relative stien fra der vi er)
data_yaml = Path(r'C:\Users\edu3650218\norgesgruppen\NMiAI2\yolo_training_ready\data.yaml')

# Start trening (YOLOv8l - Large)
# Vi bruker imgsz=800 for å unngå Out of Memory på 4GB VRAM
# Vi bruker batch=4 for stabilitet
model = YOLO('yolov8l.pt')

model.train(
    data=str(data_yaml),
    epochs=50,
    imgsz=800,
    batch=4,
    device=device,
    optimizer='AdamW',
    lr0=0.01,
    patience=10,
    project='NM_Local_Train',
    name='v8l_grocery'
)

print("TRENING FERDIG! Finn 'best.pt' i NM_Local_Train/v8l_grocery/weights/")
