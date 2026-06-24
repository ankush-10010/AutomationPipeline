import argparse
import json
from collections import defaultdict
from ultralytics import YOLO

# ── Aggregation Thresholds ──────────────────────────────────────────────
# Rule 1: Character appeared very clearly in at least one frame
CLEAR_APPEARANCE_THRESHOLD = 0.85

# Rule 2: Character appeared consistently across many frames
MIN_FRAME_RATIO = 0.15       # Must appear in at least 15% of frames
MIN_MAX_CONF_FOR_RATIO = 0.20  # With at least 0.20 peak confidence

# Minimum confidence to count a frame as "this character appeared"
FRAME_PRESENCE_THRESHOLD = 0.10


def classify_clip(model, video_path, conf_threshold=0.01):
    """Run classification on every frame and aggregate results."""

    results = model.predict(source=video_path, stream=True, verbose=False)

    # Collect per-character stats across all frames
    char_confidences = defaultdict(list)  # char_name -> [conf_per_frame]
    total_frames = 0

    for result in results:
        total_frames += 1
        names_dict = result.names
        probs = result.probs.data.tolist()

        for class_id, conf in enumerate(probs):
            char_name = names_dict[class_id]
            if char_name.lower() == "test":  # Skip the accidental "test" class
                continue
            char_confidences[char_name].append(conf)

    if total_frames == 0:
        return [], {}

    # Calculate aggregated stats for each character
    stats = {}
    present_characters = []

    for char_name, confs in char_confidences.items():
        max_conf = max(confs)
        avg_conf = sum(confs) / len(confs)
        frames_above_threshold = sum(1 for c in confs if c > FRAME_PRESENCE_THRESHOLD)
        frame_ratio = frames_above_threshold / total_frames

        stats[char_name] = {
            "max_confidence": round(max_conf, 4),
            "avg_confidence": round(avg_conf, 4),
            "frames_detected": frames_above_threshold,
            "total_frames": total_frames,
            "frame_ratio": round(frame_ratio, 4),
        }

        # ── Decision Logic ──
        # Rule 1: Very clear appearance in at least one frame
        if max_conf >= CLEAR_APPEARANCE_THRESHOLD:
            present_characters.append(char_name)
        # Rule 2: Consistent presence across many frames
        elif frame_ratio >= MIN_FRAME_RATIO and max_conf >= MIN_MAX_CONF_FOR_RATIO:
            present_characters.append(char_name)

    return present_characters, stats


def classify_image(model, image_path):
    """Run classification on a single image."""
    results = model(image_path)

    for result in results:
        names_dict = result.names
        probs = result.probs.data.tolist()

        print("\nClassification Results:")
        for i, prob in sorted(enumerate(probs), key=lambda x: x[1], reverse=True):
            name = names_dict[i]
            if name.lower() == "test":
                continue
            if prob > 0.05:
                print(f"  - {name}: {prob*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Classification Inference with Smart Frame Aggregation"
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model (e.g., best.pt)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an image file")
    group.add_argument("--video", type=str, help="Path to a video clip")

    parser.add_argument("--debug", action="store_true",
                        help="Show detailed per-character stats")

    args = parser.parse_args()

    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)

    if args.image:
        print(f"\nRunning classification on IMAGE: {args.image}")
        classify_image(model, args.image)

    elif args.video:
        print(f"\nRunning classification on VIDEO: {args.video}")
        characters, stats = classify_clip(model, args.video)

        print("\n" + "=" * 50)
        print("CHARACTERS DETECTED IN THIS CLIP:")
        if characters:
            for c in characters:
                s = stats[c]
                print(f"  [YES] {c}  (max: {s['max_confidence']:.2f}, "
                      f"in {s['frames_detected']}/{s['total_frames']} frames)")
        else:
            print("  None detected.")
        print("=" * 50)

        if args.debug:
            print("\nDETAILED STATS (all characters):")
            for char_name, s in sorted(stats.items(), key=lambda x: x[1]["max_confidence"], reverse=True):
                marker = "[YES]" if char_name in characters else "[NO] "
                print(f"  {marker} {char_name:10s}  max={s['max_confidence']:.4f}  "
                      f"avg={s['avg_confidence']:.4f}  "
                      f"frames={s['frames_detected']}/{s['total_frames']}  "
                      f"ratio={s['frame_ratio']:.2%}")


if __name__ == "__main__":
    main()
