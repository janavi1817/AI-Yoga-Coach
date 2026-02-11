import os
import shutil
import random

# Configuration
SOURCE_DIR = 'source_dataset'
BASE_DIR = 'yoga_poses'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
SPLIT_RATIO = 0.85

def import_data():
    # Verify source exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: The directory '{SOURCE_DIR}' does not exist.")
        print(f"Please create '{SOURCE_DIR}' inside 'classification model' and put your pose folders there.")
        return

    classes = os.listdir(SOURCE_DIR)
    print(f"Found {len(classes)} potential classes in {SOURCE_DIR}...")

    for class_name in classes:
        class_source_path = os.path.join(SOURCE_DIR, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_source_path):
            continue

        print(f"Processing class: {class_name}")

        # Create target directories for this class
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

        # Get all images
        images = [f for f in os.listdir(class_source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"  No images found in {class_name}, skipping.")
            continue

        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        # Copy files
        for img in train_imgs:
            src = os.path.join(class_source_path, img)
            dst = os.path.join(TRAIN_DIR, class_name, img)
            shutil.copy2(src, dst)
            
        for img in test_imgs:
            src = os.path.join(class_source_path, img)
            dst = os.path.join(TEST_DIR, class_name, img)
            shutil.copy2(src, dst)
            
        print(f"  -> Imported {len(train_imgs)} training and {len(test_imgs)} testing images.")

    print("\nImport completed successfully.")
    print("Next Step: Run 'python proprocessing.py' to generate the CSV datasets.")

if __name__ == "__main__":
    import_data()
