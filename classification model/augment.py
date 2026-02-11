import pandas as pd
import numpy as np

def augment_data(csv_path, target_samples=200):
    df = pd.read_csv(csv_path)
    augmented_rows = []
    
    # Identify classes with low sample counts
    counts = df['class_name'].value_counts()
    low_classes = counts[counts < target_samples].index.tolist()
    
    for class_name in low_classes:
        class_df = df[df['class_name'] == class_name]
        current_count = len(class_df)
        needed = target_samples - current_count
        
        print(f"Augmenting {class_name}: {current_count} -> {target_samples}")
        
        for _ in range(needed):
            # Pick a random sample from this class
            sample = class_df.sample(1).iloc[0].copy()
            
            # Extract keypoints (excluding filename, class_name, class_no)
            # Coordinates are in format y0, x0, s0, y1, x1, s2...
            kp_data = sample.drop(['filename', 'class_name', 'class_no']).values.reshape(17, 3)
            
            # Apply random noise/perturbation
            noise = np.random.normal(0, 0.005, (17, 2)) 
            kp_data[:, :2] += noise
            
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            kp_data[:, :2] *= scale
            
            # Random rotation
            angle = np.random.uniform(-0.15, 0.15) # ~8 degrees
            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c, -s), (s, c)))
            pivot = kp_data[11:13, :2].mean(axis=0) # Pivot around hips
            kp_data[:, :2] = (kp_data[:, :2] - pivot) @ R.T + pivot

            # Random translation
            kp_data[:, :2] += np.random.uniform(-0.05, 0.05, (1, 2))
            
            # Reconstruct the row
            new_row = sample.copy()
            new_row.loc[df.columns.drop(['filename', 'class_name', 'class_no'])] = kp_data.flatten()
            augmented_rows.append(new_row)
            
    if augmented_rows:
        aug_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
        aug_df.to_csv('train_data_augmented.csv', index=False)
        print(f"Saved {len(aug_df)} rows to train_data_augmented.csv")
    else:
        print("No augmentation needed.")

if __name__ == "__main__":
    augment_data('train_data.csv')
