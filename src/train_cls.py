
from ultralytics import YOLO
import torch, sys
from pathlib import Path

BASE = Path(r"C:\Users\MerveOzdemir\Desktop\VCTS")
DATA_ROOT = BASE / "splits" / "v1" / "cls"
PROJECT   = BASE / "models" / "v1"
NAME      = "oralcls_v1_m384"

# Varsayılan GPU ayarları
MODEL_WEIGHTS = "yolov8m-cls.pt"
IMGSZ    = 384
EPOCHS   = 30
BATCH    = -1
PATIENCE = 7
WORKERS  = 2
DEVICE   = 0  

def run(device, imgsz, epochs, batch, name):
    print(f"\n[RUN] device={device} imgsz={imgsz} epochs={epochs} batch={batch} name={name}")
    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=str(DATA_ROOT),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=PATIENCE,
        workers=WORKERS,
        device=device,
        project=str(PROJECT),
        name=name,
    )
    print("\nEğitim bitti. Sonuç klasörü:", results.save_dir)
    print("Ağırlıklar:", results.save_dir / "weights" / "best.pt")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

try:
    run(device=DEVICE, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, name=NAME)
except Exception as e:
    print("\n[UYARI] GPU eğitiminde hata yakalandı -> CPU'ya düşüyorum.")
    print("Hata:", e)
    
    MODEL_WEIGHTS = "yolov8s-cls.pt"
    IMGSZ = 320
    EPOCHS = 12
    BATCH = 16
    NAME = "oralcls_v1_cpu_s320"
    run(device="cpu", imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, name=NAME)
