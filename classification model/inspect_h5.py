import h5py
import numpy as np

with h5py.File('model_14.h5', 'r') as f:
    # Look for the relevant layers
    # In Keras functional model, weights are often under 'model_weights' or similar
    def get_weights(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}")

    f.visititems(get_weights)
