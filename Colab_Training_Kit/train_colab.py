from ultralytics import YOLO

def start_training():
    # 1. Last YOLOv8 Extra Large - Den kraftigste modellen for 100% poeng
    # Denne er treig å trene, men den ser flest detaljer
    model = YOLO('yolov8x.pt')

    # 2. Kjør trening med MAKSIMALE innstillinger for 100% nøyaktighet
    model.train(
        data='data.yaml',      
        epochs=150,            # Flere runder for å være helt sikker på at den lærer alt
        imgsz=1280,            # Høyeste oppløsning for å se bittesmå strekkoder/tekst
        batch=-1,              # Auto-batch: Bruker alt VRAM den finner
        device=0,              
        patience=30,           # Stopper ikke før den er helt ferdigutlært
        save=True,             
        optimizer='AdamW',     
        lr0=0.01,              
        lrf=0.01,              
        augment=True,          
        mosaic=1.0,            
        mixup=0.3,             # Hjelper modellen å generalisere på få bilder
        copy_paste=0.2,        # Genialt for å trene på objekter som overlapper
        scale=0.5,             
        degrees=15.0,          # Takler skjeve bilder i hyllene
        project='NM_AI_Project',
        name='v8x_prestige_100',
        close_mosaic=15        # De siste 15 rundene brukes til å finjustere boksene
    )

if __name__ == "__main__":
    start_training()
