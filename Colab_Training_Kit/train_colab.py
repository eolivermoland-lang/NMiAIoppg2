from ultralytics import YOLO

def start_training():
    # 1. Last den kraftigste modellen (Extra Large)
    # Dette er utgangspunktet som skal finjusteres
    model = YOLO('yolov8x.pt')

    # 2. Kjør trening med innstillinger for MAKS POENG
    # Alle stier her er relative til der data.yaml ligger
    model.train(
        data='data.yaml',      
        epochs=100,            
        imgsz=1280,            # Kritisk for å detektere små varer på store hyllebilder
        batch=-1,              # Auto-batch: Utnytter alt tilgjengelig VRAM (opptil 24GB i sandkassen)
        device=0,              
        patience=20,           
        save=True,             
        optimizer='AdamW',     
        augment=True,          
        mosaic=1.0,            
        mixup=0.1,             
        project='NM_AI_Project',
        name='v8x_high_res'
    )

if __name__ == "__main__":
    start_training()
