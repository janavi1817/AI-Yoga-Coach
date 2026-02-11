import pandas as pd
import os
import sys
sys.path.append('classification model')
from data import BodyPart

class Preprocessor:
    def __init__(self):
        self._csvs_out_folder_per_class = 'classification model/csv_per_pose'
        all_classes = sorted([f.replace('.csv', '') for f in os.listdir(self._csvs_out_folder_per_class) if f.endswith('.csv')])
        self._pose_class_names = [c for c in all_classes if c in ['chakrasana', 'bakasana', 'chair']] # Added chair for comparison
        print(f"Found classes: {self._pose_class_names}")

    def all_landmarks_as_dataframe(self):
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(self._csvs_out_folder_per_class, class_name + '.csv')
            try:
                per_class_df = pd.read_csv(csv_out_path, header=None)
                print(f"Loaded {class_name}: {len(per_class_df)} rows")
                
                # Add labels
                per_class_df['class_no'] = [class_index]*len(per_class_df)
                per_class_df['class_name'] = [class_name]*len(per_class_df)
                
                # Filename fix
                per_class_df[per_class_df.columns[0]] = class_name + '/' +  per_class_df[per_class_df.columns[0]]

                if total_df is None:
                    total_df = per_class_df
                else:
                    total_df = pd.concat([total_df, per_class_df], axis=0)
            except Exception as e:
                print(f"Error loading {class_name}: {e}")

        # Rename columns logic (simplified)
        # ...
        return total_df

p = Preprocessor()
df = p.all_landmarks_as_dataframe()
if df is not None:
    print(f"Final DF shape: {df.shape}")
    print("Classes in final DF:")
    print(df['class_name'].unique())
