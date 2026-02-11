import pandas as pd
import numpy as np
import os

# This script generates thousands of synthetic training samples from a single pose
# to make the model robust to rotation, scaling, jitter, and translation.

df = pd.read_csv('train_data.csv')
augmented_rows = []

# Target poses that need aggressive augmentation
target_poses = ['bakasana', 'dhanurasana', 'gomukhasana', 'padmasana', 'ustrasana']

for index, sample in df.iterrows():
    if sample['class_name'] in target_poses:
        # Generate 1000 variations for each sample of target poses
        for _ in range(1000):
            kp_data = sample.drop(['filename', 'class_name', 'class_no']).values.astype(float).reshape(17, 3)
            
            # Extreme jitter
            noise = np.random.normal(0, 0.02, (17, 2)) 
            kp_data[:, :2] += noise
            
            # Extreme scaling
            scale = np.random.uniform(0.7, 1.3)
            kp_data[:, :2] *= scale
            
            # Random rotation
            angle = np.random.uniform(-0.5, 0.5) # ~28 degrees
            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c, -s), (s, c)))
            pivot = kp_data[11:13, :2].mean(axis=0) # Pivot around hips
            kp_data[:, :2] = (kp_data[:, :2] - pivot) @ R.T + pivot

            # Random translation
            kp_data[:, :2] += np.random.uniform(-0.1, 0.1, (1, 2))
            
            # Mirroring (Randomly flip x coordinates)
            if np.random.random() > 0.5:
                # Based on preprocessing.py: [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score] ...]
                # So x is at index 0. We assume x is normalized 0-1.
                kp_data[:, 0] = 1.0 - kp_data[:, 0]

            # Perspective-ish shear
            shear = np.random.uniform(-0.15, 0.15, (2, 2))
            kp_data[:, :2] = kp_data[:, :2] @ (np.eye(2) + shear)

            new_row = sample.copy()
            new_row.loc[df.columns.drop(['filename', 'class_name', 'class_no'])] = kp_data.flatten()
            augmented_rows.append(new_row)
            
if augmented_rows:
    aug_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    aug_df.to_csv('train_data_super_augmented.csv', index=False)
    print(f"Super-Augmentation complete: {len(aug_df)} total samples.")
else:
    print("No target poses found for augmentation.")
