#%%
import os
from pathlib import Path

# IMPORTANT: Change this to the path where your 'train' folder is located
BASE_DIR = Path('D:\YV\DSGP-IIT\disease and pest detection\All Collected Dataset')

# Get a list of all class folders (e.g., Healthy, Red_Rust, etc.)
classes = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]

print(f"--- Dataset Structure Check ({BASE_DIR.name} Directory) ---")
total_images = 0
class_counts = {}

for class_name in classes:
    class_path = BASE_DIR / class_name
    count = len(list(class_path.glob('*')))
    class_counts[class_name] = count
    total_images += count
    print(f"  {class_name}: {count} images")

print(f"\nTotal Images in Training Set: {total_images}")