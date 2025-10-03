from ultralytics import YOLO
import cv2
import argparse

def run_webcam(weights, camera_index=0, conf=0.35):
    # Load trained YOLO model
    model = YOLO(weights)

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_index}")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=conf)[0]

        # Annotated frame (boxes + labels drawn automatically)
        annotated = results.plot()

        # Show detections
        cv2.imshow("YOLO Webcam Detection", annotated)

        # Print detections to terminal
        # Print detections to terminal
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"class:{results.names[cls]}, conf:{conf_val:.2f}, box:{xyxy}")
        else:
            print("No detections")

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Change the default value for the webcam 0, 1, 2 etc.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/exp/weights/best.pt", help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    run_webcam(args.weights, args.camera, args.conf)
