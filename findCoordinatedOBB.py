# detect_and_send_obb.py
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
    p.add_argument('--weights', default='runs/train/exp/weights/best.pt', help='path to trained OBB weights')
    p.add_argument('--camera', type=int, default=0, help='webcam index (0 default)')
    p.add_argument('--serial_port', default=None, help='Serial port to Arduino (e.g. COM3 or /dev/ttyACM0)')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--show', action='store_true', help='Show visual output window')
    p.add_argument('--focal', type=float, default=800.0, help='Focal length in pixels (for distance estimation)')
    p.add_argument('--known_width', type=float, default=0.05, help='Known object width in meters (for distance estimation)')
    return p.parse_args()

def format_message_obb(class_id, conf, norm_cx, norm_cy, norm_w, norm_h, angle_rad, 
                        norm_x1, norm_y1, norm_x2, norm_y2, norm_x3, norm_y3, norm_x4, norm_y4, 
                        distance=None):
    # Extended CSV message with OBB data: class,conf,cx,cy,w,h,angle,x1,y1,x2,y2,x3,y3,x4,y4,dist\n
    if distance is None:
        dist_str = 'nan'
    else:
        dist_str = f"{distance:.3f}"
    return (f"{class_id},{conf:.3f},{norm_cx:.4f},{norm_cy:.4f},{norm_w:.4f},{norm_h:.4f},"
            f"{angle_rad:.4f},{norm_x1:.4f},{norm_y1:.4f},{norm_x2:.4f},{norm_y2:.4f},"
            f"{norm_x3:.4f},{norm_y3:.4f},{norm_x4:.4f},{norm_y4:.4f},{dist_str}\n")

def main():
    args = parse_args()
    model = YOLO(args.weights)  # load OBB model
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

            # Run inference for OBB
            results = model(frame, imgsz=640, conf=0.35, iou=0.45)

            # Iterate through results
            for result in results:
                if result.obb is None or len(result.obb) == 0:
                    continue
                
                # Extract OBB data
                xywhr = result.obb.xywhr.cpu().numpy()  # [N, 5] center-x, center-y, width, height, angle (radians)
                xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()  # [N, 4, 2] polygon format with 4 points
                confs = result.obb.conf.cpu().numpy()  # [N] confidence scores
                classes = result.obb.cls.int().cpu().numpy()  # [N] class indices
                
                # Iterate through each detected OBB
                for i in range(len(xywhr)):
                    cx, cy, box_w, box_h, angle_rad = xywhr[i]
                    conf = float(confs[i])
                    cls = int(classes[i])
                    
                    # Extract 4 corner points from polygon
                    x1, y1 = xyxyxyxy[i][0]
                    x2, y2 = xyxyxyxy[i][1]
                    x3, y3 = xyxyxyxy[i][2]
                    x4, y4 = xyxyxyxy[i][3]
                    
                    # Normalized coordinates (0..1)
                    norm_cx = cx / w
                    norm_cy = cy / h
                    norm_w = box_w / w
                    norm_h = box_h / h
                    
                    norm_x1 = x1 / w
                    norm_y1 = y1 / h
                    norm_x2 = x2 / w
                    norm_y2 = y2 / h
                    norm_x3 = x3 / w
                    norm_y3 = y3 / h
                    norm_x4 = x4 / w
                    norm_y4 = y4 / h
                    
                    # Simple angle relative to image center (degrees)
                    center_x = w / 2.0
                    dx = cx - center_x
                    view_angle_rad = np.arctan2(dx, args.focal)
                    view_angle_deg = np.degrees(view_angle_rad)
                    
                    # Distance estimate using known width
                    distance_m = estimate_distance(args.known_width, args.focal, box_w)
                    
                    # Format and send message
                    msg = format_message_obb(cls, conf, norm_cx, norm_cy, norm_w, norm_h, angle_rad,
                                            norm_x1, norm_y1, norm_x2, norm_y2, 
                                            norm_x3, norm_y3, norm_x4, norm_y4, distance_m)
                    print("Detected OBB:", msg.strip())
                    
                    # Send to serial
                    if ser and ser.is_open:
                        try:
                            ser.write(msg.encode('utf-8'))
                        except Exception as e:
                            print("Serial write error:", e)
                    
                    # Optional visualization
                    if args.show:
                        label = f"{result.names[cls]} {conf:.2f}"
                        
                        # Draw oriented bounding box (4 corners)
                        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                        
                        # Draw center point
                        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                        
                        # Display label
                        cv2.putText(frame, label, (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Display angle and distance info
                        angle_deg = np.degrees(angle_rad)
                        info_text = f"rot:{angle_deg:.1f}° view:{view_angle_deg:.1f}° d:{distance_m:.2f}m"
                        cv2.putText(frame, info_text, (int(x1), int(max(y1, y2, y3, y4))+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if args.show:
                cv2.imshow("YOLO OBB Webcam", frame)
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