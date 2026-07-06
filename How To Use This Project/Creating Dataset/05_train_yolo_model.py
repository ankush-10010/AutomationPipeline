from ultralytics import YOLO
import torch
from pathlib import Path

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "YOLO_Final_Dataset" / "data.yaml"
FINAL_MODEL_DIR = PROJECT_ROOT / "final_model"

def main():
    print(f"{'='*60}\nInitializing Advanced YOLOv8 Training Pipeline\n{'='*60}")
    
    # 1. Check GPU Hardware
    has_gpu = torch.cuda.is_available()
    print(f"CUDA GPU Available: {has_gpu}")
    if has_gpu:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("WARNING: Training on CPU! This will take a very long time.")
        device = 'cpu'
        
    if not DATA_YAML.exists():
        print(f"Error: {DATA_YAML} not found!")
        return

    # 2. Choose the YOLOv8 Small Model for 30-Minute Training
    # The 'Small' (yolov8s.pt) model trains 3x faster than Medium while maintaining excellent accuracy.
    print("\nLoading YOLOv8 Small (yolov8s.pt) architecture for speed...")
    model = YOLO("yolov8s.pt")
    
    # 3. Train the Model with Optimized Hyperparameters
    print(f"\nStarting Training Phase...")
    print(f"Dataset: {DATA_YAML}")
    print(f"Output Directory: {FINAL_MODEL_DIR}/ben10_detector")
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,             # 50 Epochs is the sweet spot for fast convergence on cartoon datasets
        imgsz=640,             
        batch=16,              # Increased batch to 16 because Small model uses less VRAM
        patience=15,           
        project=str(FINAL_MODEL_DIR), 
        name="ben10_detector", 
        device=device,
        workers=8,             # Maximize CPU data loading threads
        cache=True,            # CRITICAL FOR SPEED: Loads all images into RAM to eliminate disk bottlenecks!
        optimizer='auto'       
    )
    
    print(f"\n{'='*60}")
    print("TRAINING SUCCESSFULLY COMPLETED!")
    print(f"The BEST model weights are saved at:")
    print(f"-> {FINAL_MODEL_DIR}/ben10_detector/weights/best.pt")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
