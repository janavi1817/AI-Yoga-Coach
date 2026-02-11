import pandas as pd
try:
    df = pd.read_csv("classification model/train_data.csv", header=None)
    # The last column is class_name. Usually column index -1.
    class_col_idx = len(df.columns) - 1
    chakrasana_df = df[df[class_col_idx] == 'chakrasana']
    print(f"Chakrasana rows: {len(chakrasana_df)}")
except Exception as e:
    print(e)
