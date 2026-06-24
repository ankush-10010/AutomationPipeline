import argparse
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned YOLO model on an image or video clip.")
    parser.add_argument("--weights", type=str, required=True, help="Path to your trained model weights (e.g., runs/detect/train/weights/best.pt)")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an image file to test")
    group.add_argument("--video", type=str, help="Path to a video clip (.mp4) to test")
    
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    # Load the fine-tuned model
    print(f"Loading custom weights from {args.weights}...")
    model = YOLO(args.weights)

    if args.image:
        print(f"\n🖼️ Running inference on IMAGE: {args.image}")
        results = model(args.image, conf=args.conf)
        
        for result in results:
            print("\nDetected Characters:")
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f" - {class_name} ({confidence:.2f} conf)")
                
            # Display the image with bounding boxes
            result.show()

    elif args.video:
        print(f"\n🎬 Running inference on VIDEO: {args.video}")
        # Process the video stream and show bounding boxes in real-time
        # save=True will also save the annotated video to runs/detect/predict/
        results = model.predict(source=args.video, conf=args.conf, show=True, save=True)
        print(f"\n✅ Video inference complete. Processed video saved to: runs/detect/predict/")

if __name__ == "__main__":
    main()
