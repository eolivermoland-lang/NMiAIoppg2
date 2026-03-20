from ultralytics import YOLO

def start_training():
    # 1. Last YOLOv8 Large - Bedre balanse mellom kraft og overtilpasning enn Extra Large
    model = YOLO('yolov8l.pt')

    # 2. Kjør trening med forbedrede innstillinger
    model.train(
        data='data.yaml',      
        epochs=100,            
        imgsz=1024,            # 1024 er ofte "sweet spot" for disse hyllebildene (1280 kan gi minneproblemer på 8GB RAM senere)
        batch=-1,              
        device=0,              
        patience=25,           # Litt høyere tålmodighet
        save=True,             
        optimizer='AdamW',     
        lr0=0.01,              # Standard start-læringsrate
        lrf=0.01,              # Slutt-læringsrate (1% av start)
        augment=True,          
        mosaic=1.0,            
        mixup=0.2,             # Økt mixup for å hjelpe på sjeldne produkter
        copy_paste=0.1,        # Legger til copy-paste for å trene på flere objekter
        scale=0.5,             # Hjelper med å detektere produkter i ulik avstand
        project='NM_AI_Project',
        name='v8l_optimized',
        close_mosaic=20        # Skru av mosaic de siste 20 epokene for å "finpusse" boksene
    )

if __name__ == "__main__":
    start_training()
