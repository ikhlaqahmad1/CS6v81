from ultralytics import YOLO
import cv2
import argparse
import serial
import time

def run_webcam(weights, camera_index=0, conf=0.35, serial_port=None, baud=9600):
    # Load YOLO model
    model = YOLO(weights)

    # Setup serial if port is provided
    ser = None
    if serial_port:
        try:
            ser = serial.Serial(serial_port, baud, timeout=1)
            time.sleep(2)  # wait for Arduino to reset
            print(f"✅ Serial connected on {serial_port} at {baud} baud")
        except Exception as e:
            print(f"⚠️ Could not open serial port: {e}")
            ser = None

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam {camera_index}")
        return

    print("📸 Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=conf)[0]
        annotated = results.plot()

        # Process detections
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Calculate center coordinates
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Draw center on annotated image
                cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                label = f"{results.names[cls]} ({int(cx)},{int(cy)})"
                cv2.putText(annotated, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Print info for debugging
                print(f"class: {results.names[cls]}, conf: {conf_val:.2f}, center: ({cx:.1f}, {cy:.1f})")

                # Send coordinates to Arduino if connected
                if ser:
                    message = f"{int(cx)},{int(cy)}\n"
                    try:
                        ser.write(message.encode('utf-8'))
                        # Optional: Read back Arduino response
                        # response = ser.readline().decode().strip()
                        # print(f"Arduino: {response}")
                    except Exception as e:
                        print(f"⚠️ Serial write error: {e}")
        else:
            print("No detections")

        # Show the annotated frame
        cv2.imshow("YOLO Webcam Detection", annotated)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
        print("🔌 Serial connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/exp/weights/best.pt", help="Path to trained YOLO model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--serial_port", default=None, help="Serial port (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate (default 9600)")
    args = parser.parse_args()

    run_webcam(args.weights, args.camera, args.conf, args.serial_port, args.baud)
