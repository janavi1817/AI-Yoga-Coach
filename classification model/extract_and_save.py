import h5py
import numpy as np
import os

h5_path = 'model_14.h5'
bin_path = 'group1-shard1of1.bin'

with h5py.File(h5_path, 'r') as f:
    # Feature extraction based on inspection
    # dense/dense/kernel:0, bias:0
    # dense_1/dense_1/kernel:0, bias:0
    # dense_2/dense_2/kernel:0, bias:0
    
    weights_to_save = []
    
    # Layer 0: dense
    weights_to_save.append(f['model_weights']['dense']['dense']['kernel'][:])
    weights_to_save.append(f['model_weights']['dense']['dense']['bias'][:])
    
    # Layer 1: dense_1
    weights_to_save.append(f['model_weights']['dense_1']['dense_1']['kernel'][:])
    weights_to_save.append(f['model_weights']['dense_1']['dense_1']['bias'][:])
    
    # Layer 2: dense_2
    weights_to_save.append(f['model_weights']['dense_2']['dense_2']['kernel'][:])
    weights_to_save.append(f['model_weights']['dense_2']['dense_2']['bias'][:])

    # Join and flatten
    with open(bin_path, 'wb') as out_f:
        for w in weights_to_save:
            print(f"Saving weight shape: {w.shape}")
            out_f.write(w.tobytes())

print(f"Success! Saved weights to {bin_path}")
