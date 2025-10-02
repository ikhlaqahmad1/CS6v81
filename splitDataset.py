import os
import shutil
from glob import glob

# Paths
base_dir = "train"   # where original images/labels folders exist
output_dir = "dataset_split"

prefixes = ["Conforming_gauze2", "HBS_Robot_Table"]
splits = {"train": 0.8, "test": 0.1, "val": 0.1}

# Create output folders
for split in splits:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

# Process each prefix separately
for prefix in prefixes:
    print(f"\nProcessing prefix: {prefix}")

    # Collect all images/labels with this prefix
    images = sorted(glob(os.path.join(base_dir, "images", f"{prefix}*")))
    labels = sorted(glob(os.path.join(base_dir, "labels", f"{prefix}*")))

    # Ensure matching counts
    assert len(images) == len(labels), f"Mismatch for prefix {prefix}"

    total = len(images)
    n_train = int(total * splits["train"])
    n_test = int(total * splits["test"])
    n_val = total - n_train - n_test  # remaining

    # Sequential split
    train_imgs, test_imgs, val_imgs = images[:n_train], images[n_train:n_train+n_test], images[n_train+n_test:]
    train_lbls, test_lbls, val_lbls = labels[:n_train], labels[n_train:n_train+n_test], labels[n_train+n_test:]

    # Helper to copy files
    def copy_files(file_list, dest_subfolder):
        for f in file_list:
            shutil.copy(f, os.path.join(output_dir, dest_subfolder, os.path.basename(f)))

    # Copy into split folders
    copy_files(train_imgs, "train/images")
    copy_files(test_imgs, "test/images")
    copy_files(val_imgs, "val/images")

    copy_files(train_lbls, "train/labels")
    copy_files(test_lbls, "test/labels")
    copy_files(val_lbls, "val/labels")

    print(f" -> {len(train_imgs)} train, {len(test_imgs)} test, {len(val_imgs)} val")

print("\n✅ Dataset split completed.")