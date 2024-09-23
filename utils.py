import numpy as np

def check_empty_data(data):
    """Check if the data is empty or contains only zeros."""
    return len(data) == 0 or np.all(data == 0)

def calculate_rms(data, window_size):
    squared = data**2
    windows = np.lib.stride_tricks.sliding_window_view(squared, window_size)
    return np.sqrt(np.mean(windows, axis=1))