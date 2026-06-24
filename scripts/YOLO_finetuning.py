import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on custom character dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to your dataset's data.yaml file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train (default: 50)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained base model (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    # Load a pre-trained YOLOv8 model
    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Train the model on your custom dataset
    print(f"Starting training on {args.data} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device="cuda"  # It will automatically fallback to CPU if CUDA isn't available, but cuda is preferred
    )

    print("\n✅ Training complete!")
    print("Your fine-tuned weights are saved in: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
