import os
import re

folders = ["train/images",
           "train/labels"]

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".txt"):  # jpg for images, txt for labels
            old_path = os.path.join(folder, filename)

            # Remove '_png' and everything after it, but keep the .jpg or .txt
            new_filename = re.sub(r'_png.*?(\.[a-z]+)$', r'\1', filename)

            new_path = os.path.join(folder, new_filename)

            if old_path != new_path:  # avoid unnecessary rename
                if not os.path.exists(new_path):  # prevent accidental overwrite
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                else:
                    print(f"Skipped (already exists): {new_filename}")