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

        # Run YOLO detection
        results = model(frame, conf=conf)[0]

        # Draw boxes and labels automatically
        annotated = results.plot()

        # Loop through detections
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()

                # Get box coordinates
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Print detection info
                print(f"class: {results.names[cls]}, conf: {conf_val:.2f}, box: {xyxy}, center: ({cx:.1f}, {cy:.1f})")

                # Draw center point
                cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        else:
            print("No detections")

        # Show the frame
        cv2.imshow("YOLO Webcam Detection", annotated)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/exp/weights/best.pt", help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    run_webcam(args.weights, args.camera, args.conf)
