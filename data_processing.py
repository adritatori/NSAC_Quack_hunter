import numpy as np
from obspy import read
from utils import calculate_rms
def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)
def load_data(file_path):
    """Load seismic data from MiniSEED file."""
    st = read(file_path)
    return st[0].data, st[0].stats.sampling_rate, st[0].stats.starttime

def ste_segmentation(data, frame_size, hop_size, threshold_factor):
    # Normalize the input data
    normalized_data = normalize_data(data)

    num_frames = int(np.ceil((len(normalized_data) - frame_size) / hop_size)) + 1
    padding_length = (num_frames - 1) * hop_size + frame_size - len(normalized_data)
    padded_data = np.pad(normalized_data, (0, padding_length), mode='constant')

    ste = []
    for i in range(num_frames):
        frame = padded_data[i * hop_size : i * hop_size + frame_size]
        ste.append(np.sum(frame**2) / frame_size)

    threshold = threshold_factor * np.median(ste)

    segments = []
    in_segment = False
    start_index = 0
    for i, energy in enumerate(ste):
        if energy > threshold and not in_segment:
            in_segment = True
            start_index = i * hop_size
        elif energy <= threshold and in_segment:
            in_segment = False
            end_index = i * hop_size
            segments.append((start_index, end_index))

    if in_segment:
        end_index = len(data)
        segments.append((start_index, end_index))

    return segments

def calculate_rms(data, window_size):
    # Normalize the input data
    normalized_data = normalize_data(data)
    squared = normalized_data**2
    windows = np.lib.stride_tricks.sliding_window_view(squared, int(window_size))
    return np.sqrt(np.mean(windows, axis=1))


def adaptive_segmentation(data, window_size, threshold_factor):
    rms = calculate_rms(data, window_size)
    threshold = threshold_factor * np.mean(rms)
    mask = np.zeros(len(data), dtype=bool)
    mask[window_size-1:] = rms < threshold
    return mask
def combined_ste_rms_reduction(data, sampling_rate, ste_frame_size, ste_hop_size, ste_threshold, rms_window_size, global_rms_threshold, segment_threshold_factor):
    # Normalize the input data
    normalized_data = normalize_data(data)

    ste_segments = ste_segmentation(normalized_data, ste_frame_size, ste_hop_size, ste_threshold)
    global_rms_mask = adaptive_segmentation(normalized_data, rms_window_size, global_rms_threshold)

    final_mask = global_rms_mask.copy()

    for start, end in ste_segments:
        segment = normalized_data[start:end]
        segment_mask = adaptive_segmentation(segment, int(rms_window_size/2), segment_threshold_factor)
        final_mask[start:end] = final_mask[start:end] | segment_mask

    return final_mask, ste_segments, calculate_rms(normalized_data, rms_window_size)