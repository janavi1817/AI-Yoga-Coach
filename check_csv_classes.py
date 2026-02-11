import pandas as pd

try:
    df = pd.read_csv('classification model/train_data.csv')
    if 'class_name' in df.columns:
        print("Unique classes in train_data.csv:")
        print(df['class_name'].unique())
        print(f"Total classes: {len(df['class_name'].unique())}")
    else:
        print("Column 'class_name' not found in train_data.csv")
        print("Columns found:", df.columns)
except Exception as e:
    print(f"Error reading CSV: {e}")
