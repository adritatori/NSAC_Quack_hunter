import numpy as np
from obspy import read
from utils import calculate_rms

def load_data(file_path):
    """Load seismic data from MiniSEED file."""
    st = read(file_path)
    return st[0].data, st[0].stats.sampling_rate, st[0].stats.starttime

def ste_segmentation(data, frame_size, hop_size, threshold_factor):
    num_frames = int(np.ceil((len(data) - frame_size) / hop_size)) + 1
    padding_length = (num_frames - 1) * hop_size + frame_size - len(data)
    padded_data = np.pad(data, (0, padding_length), mode='constant')

    ste = []
    for i in range(num_frames):
        frame = padded_data[i * hop_size : i * hop_size + frame_size]
        ste.append(np.sum(frame**2) / frame_size)

    threshold = threshold_factor * np.mean(ste)

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

def adaptive_segmentation(data, window_size, threshold_factor):
    rms = calculate_rms(data, window_size)
    threshold = threshold_factor * np.mean(rms)
    mask = np.zeros(len(data), dtype=bool)
    mask[window_size-1:] = rms < threshold
    return mask

def combined_ste_rms_reduction(data, sampling_rate, ste_frame_size, ste_hop_size, ste_threshold, rms_window_size, global_rms_threshold, segment_threshold_factor=0.8):
    # Perform STE segmentation
    ste_segments = ste_segmentation(data, ste_frame_size, ste_hop_size, ste_threshold)

    # Perform global RMS adaptive segmentation
    global_rms_mask = adaptive_segmentation(data, rms_window_size, global_rms_threshold)
    # Create a mask for the final result
    final_mask = global_rms_mask.copy()

    for start, end in ste_segments:
        # Extract the segment
        segment = data[start:end]

        # Calculate RMS for this segment
        segment_rms = calculate_rms(segment, 50)

        # Calculate adaptive threshold for this segment
        segment_threshold = segment_threshold_factor * np.mean(segment_rms)

        # Create a mask for this segment based on adaptive threshold
        segment_mask = adaptive_segmentation(segment, 5, segment_threshold_factor)
        segment_mask[50-1:] = segment_rms > segment_threshold

        # Combine the segment mask with the global mask
        final_mask[start:end] = final_mask[start:end] | segment_mask

    return final_mask, ste_segments, calculate_rms(data, rms_window_size)