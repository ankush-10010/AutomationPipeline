import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO Image Classification (No Bounding Boxes).")
    parser.add_argument("--data", type=str, required=True, help="Path to your formatted dataset root (must contain train/ and val/ folders)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train (default: 20)")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Pretrained classification base model")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    # Load a pre-trained YOLOv8 CLASSIFICATION model (-cls suffix)
    print(f"Loading classification base model: {args.model}")
    model = YOLO(args.model)

    # Train the model on the image folders
    print(f"Starting classification training on {args.data} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=224,  # Classification standard size
        device="cuda" 
    )

    print("\n✅ Classification Training complete!")
    print("Your fine-tuned weights are saved in: runs/classify/train/weights/best.pt")

if __name__ == "__main__":
    main()
