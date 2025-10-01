# detect_and_send.py
import cv2
import argparse
import time
import serial
import sys
import numpy as np
from ultralytics import YOLO

def estimate_distance(known_width_m, focal_length, pixel_width):
    # simple focal-based distance: distance = (known_width * focal) / pixel_width
    # known_width_m: real object width in meters
    # focal_length: in pixel units (calibrate)
    if pixel_width <= 0:
        return None
    return (known_width_m * focal_length) / pixel_width

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/train/exp/weights/best.pt', help='path to trained weights')
    p.add_argument('--camera', type=int, default=0, help='webcam index (0 default)')
    p.add_argument('--serial_port', default=None, help='Serial port to Arduino (e.g. COM3 or /dev/ttyACM0)')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--show', action='store_true', help='Show visual output window')
    p.add_argument('--focal', type=float, default=800.0, help='Focal length in pixels (for distance estimation)')
    p.add_argument('--known_width', type=float, default=0.05, help='Known object width in meters (for distance estimation)')
    return p.parse_args()

def format_message(class_id, conf, norm_x, norm_y, norm_w, norm_h, distance=None):
    # Simple CSV-like message: class,conf,x,y,w,h,dist\n
    if distance is None:
        dist_str = 'nan'
    else:
        dist_str = f"{distance:.3f}"
    return f"{class_id},{conf:.3f},{norm_x:.4f},{norm_y:.4f},{norm_w:.4f},{norm_h:.4f},{dist_str}\n"

def main():
    args = parse_args()
    model = YOLO(args.weights)  # load model
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open camera", args.camera)
        sys.exit(1)

    ser = None
    if args.serial_port:
        try:
            ser = serial.Serial(args.serial_port, args.baud, timeout=1)
            time.sleep(2)  # wait for arduino reset
            print("Serial connected:", args.serial_port)
        except Exception as e:
            print("Warning: cannot open serial port:", e)
            ser = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]

            # Run inference (Ultralytics returns a list of results)
            results = model(frame, imgsz=640, conf=0.35, iou=0.45)[0]

            # results.boxes contains boxes. Each box has xyxy, conf, cls
            # iterate boxes
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                box_w = x2 - x1
                box_h = y2 - y1
                cx = x1 + box_w / 2.0
                cy = y1 + box_h / 2.0

                # normalized coordinates (0..1)
                norm_x = cx / w
                norm_y = cy / h
                norm_w = box_w / w
                norm_h = box_h / h

                # simple angle relative to image center (radians)
                center_x = w / 2.0
                dx = cx - center_x
                angle_rad = np.arctan2(dx, args.focal)  # small-angle approx using focal (pixels)
                angle_deg = np.degrees(angle_rad)

                # distance estimate using known width (if desired)
                distance_m = estimate_distance(args.known_width, args.focal, box_w)

                msg = format_message(cls, conf, norm_x, norm_y, norm_w, norm_h, distance_m)
                print("Detected:", msg.strip())

                # send to serial
                if ser and ser.is_open:
                    try:
                        ser.write(msg.encode('utf-8'))
                    except Exception as e:
                        print("Serial write error:", e)

                # optional visualization
                if args.show:
                    label = f"{results.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)
                    # display angle and distance
                    cv2.putText(frame, f"a:{angle_deg:.1f} deg d:{distance_m:.2f}m", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            if args.show:
                cv2.imshow("YOLO Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
