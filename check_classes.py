import pandas as pd
try:
    df = pd.read_csv("classification model/train_data.csv")
    print(f"Total rows: {len(df)}")
    print("Classes found:")
    print(df['class_name'].unique())
except Exception as e:
    print(e)
