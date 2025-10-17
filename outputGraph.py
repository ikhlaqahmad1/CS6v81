import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# === Load your trained YOLOv11 OBB model ===
# Example: model = YOLO("runs/obb/train/weights/best.pt")
model = YOLO("runs/train/exp/weights/best.pt")  # change path as needed

# === Evaluate model on validation set ===
results = model.val(split='val', plots=False)  # disable plots for speed

# === Extract per-class mAPs (mAP@0.5 and mAP@0.5:0.95) ===
# results.maps is an array: one row per class
maps = results.maps  # shape (num_classes, 2) or similar depending on metrics
names = results.names  # dict: {class_index: class_name}

# If using OBB, ensure it’s available
if maps is None:
    raise ValueError("No mAP results found. Ensure you have validation data and correct metrics computed.")

# === Choose which metric to plot ===
# By default, take mAP@0.5:0.95
mAP_per_class = maps[:, 1] if maps.ndim > 1 else maps

# === Plot histogram ===
plt.figure(figsize=(10, 6))
plt.bar(names.values(), mAP_per_class, color='skyblue')
plt.title('Per-Class mAP (YOLOv11 OBB)')
plt.xlabel('Class Label')
plt.ylabel('mAP@0.5:0.95')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Print table summary ===
print("\nPer-class mAP@0.5:0.95:")
for i, name in names.items():
    print(f"{name:<15}: {mAP_per_class[i]:.4f}")