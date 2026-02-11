import pandas as pd

try:
    df = pd.read_csv('classification model/train_data.csv')
    if 'class_name' in df.columns and 'class_no' in df.columns:
        mapping = df[['class_name', 'class_no']].drop_duplicates().sort_values('class_no')
        print("Class mapping found in train_data.csv:")
        print(mapping)
    else:
        print("Required columns not found.")
except Exception as e:
    print(f"Error: {e}")
