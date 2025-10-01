# train_yolo.py
from ultralytics import YOLO
import argparse
import os

def main(data_yaml, model='yolov8n.pt', epochs=10, imgsz=640, batch=16,
         save_dir='train'):
    # Create model from a pretrained checkpoint (yolov8n small)
    model = YOLO(model)
    # Train
    model.train(data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=os.path.dirname(save_dir) or 'runs/train',
                name=os.path.basename(save_dir),
                exist_ok=True)
    print("Training finished. Best model saved in project folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.yaml', help='data.yaml')
    parser.add_argument('--model', default='yolov8n.pt', help='base model checkpoint (or custom)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--save_dir', default='runs/train/exp', help='where to save results')
    args = parser.parse_args()
    main(args.data, args.model, args.epochs, args.imgsz, args.batch, args.save_dir)
