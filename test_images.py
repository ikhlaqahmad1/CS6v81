from ultralytics import YOLO
import cv2
import os
import argparse

def run_inference(weights, source, out_dir="results", conf=0.35, display=True):
    # Load trained model
    model = YOLO(weights)

    # Create output dir
    os.makedirs(out_dir, exist_ok=True)

    # Process all images in source
    for img_name in os.listdir(source):
        img_path = os.path.join(source, img_name)
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        results = model(img_path, conf=conf)[0]   # run prediction on one image

        # Get annotated image (numpy array with boxes drawn)
        annotated = results.plot()

        # Save
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, annotated)

        # Print detections (class, conf, bbox)
        for box in results.boxes:
            cls = int(box.cls[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"{img_name} → class:{results.names[cls]}, conf:{conf_val:.2f}, box:{xyxy}")

        # Display window if enabled
        if display:
            cv2.imshow("Prediction", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):  # press 'q' to quit early
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to trained model")
    parser.add_argument("--source", default="dataset_split/test/images", help="Folder with images to test")
    parser.add_argument("--out", default="results", help="Folder to save annotated results")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--no_display", action="store_true", help="Disable display window")
    args = parser.parse_args()

    run_inference(args.weights, args.source, args.out, args.conf, display=not args.no_display)

