import altair as alt
import pandas as pd
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import numpy as np
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
from obspy import read
from scipy import signal, stats
from scipy.signal import find_peaks
import pywt
from pywt import cwt
from scipy.spatial.distance import cdist
import kaggle


KAGGLE_DATASET = "aiofrivia/NasaSPaceAppsChallanegeDataset"
DATASET_PATH = "NasaSPaceAppsChallanegeDataset"

def download_kaggle_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path='.', unzip=True)
    st.success(f"Dataset downloaded to {DATASET_PATH}")


st.set_page_config(
    page_title="Quack Hunter", page_icon="⬇", layout="centered"
)


def load_data(file_path):
    """Load seismic data from MiniSEED file."""
    st = read(file_path)
    return st[0].data, st[0].stats.sampling_rate, st[0].stats.starttime

def check_empty_data(data):
    """Check if the data is empty or contains only zeros."""
    return len(data) == 0 or np.all(data == 0)

def autocorrelation(data, max_lag):
    """Compute autocorrelation of the data."""
    result = np.correlate(data, data, mode='full')
    return result[result.size // 2:result.size // 2 + max_lag]


def detect_events(data, sampling_rate, sta_len=5, lta_len=50, thr_on=3.0, thr_off=1, min_dur=2.0):
    sta_samples = int(sta_len * sampling_rate)
    lta_samples = int(lta_len * sampling_rate)
    cft = classic_sta_lta(data, sta_samples, lta_samples)

    triggers = trigger_onset(cft, thr_on, thr_off)

    refined_triggers = []
    for start, end in triggers:
        duration = (end - start) / sampling_rate
        if duration >= min_dur:
            event_amp = np.max(np.abs(data[start:end]))
            background_amp = np.median(np.abs(data))
            if event_amp > 3.0 * background_amp:
                refined_triggers.append((start, end))

    return refined_triggers

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

def calculate_rms(data, window_size):
    squared = data**2
    windows = np.lib.stride_tricks.sliding_window_view(squared, window_size)
    return np.sqrt(np.mean(windows, axis=1))

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
def evaluate_detections(detections, catalog, sampling_rate, time_tolerance=5.0):
    det_times = np.array([(start + end) / 2 / sampling_rate for start, end in detections])
    cat_times = catalog['time_rel_sec'].values

    dist_matrix = cdist(det_times.reshape(-1, 1), cat_times.reshape(-1, 1))
    matches = dist_matrix < time_tolerance

    true_positives = np.sum(matches.any(axis=0))
    false_positives = len(detections) - true_positives
    false_negatives = len(catalog) - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def plot_results(raw_data, processed_data, raw_time, processed_time, sampling_rate, detections, catalog, file_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Plot raw data
    ax1.plot(raw_time, raw_data, label='Raw Data', color='blue', alpha=0.7)
    ax1.set_title(f"Raw Data - {file_name}", fontsize=16)
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=10)

    # Plot processed data
    ax2.plot(raw_time, raw_data, color='lightgray', alpha=0.5, label='Raw Data')
    ax2.plot(processed_time, processed_data, color='blue', alpha=0.7, label='Processed Data')

    # Plot detections
    for start, end in detections:
        start_time = processed_time[start]
        end_time = processed_time[end]
        ax2.axvspan(start_time, end_time, color='red', alpha=0.3)

    # Plot catalog events if available
    if catalog is not None and len(catalog) > 0:
        for _, event in catalog.iterrows():
            event_time = event['time_rel_sec']
            ax2.axvline(x=event_time, color='green', linestyle='--', linewidth=2)
        print(f"Plotted {len(catalog)} catalog events")
        ax2.legend(['Raw Signal', 'Processed Signal', 'Detected Events', 'Catalog Events'], fontsize=10, loc='upper right')
    else:
        ax2.legend(['Raw Signal', 'Processed Signal', 'Detected Events'], fontsize=10, loc='upper right')

    ax2.set_title(f"Processed Data with Detections - {file_name}", fontsize=16)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()

def process_dataset(data_dir, global_rms_threshold, ste_frame_size, ste_hop_size, ste_threshold, rms_window_size,  catalog_file=None, is_training=True):
    results = []

    if is_training and catalog_file:
        catalog = pd.read_csv(catalog_file)
        print(f"Loaded catalog with {len(catalog)} events")
    else:
        catalog = None

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mseed')]

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)

        raw_data, sampling_rate, start_time = load_data(file_path)

        if check_empty_data(raw_data):
            print(f"Warning: Empty or zero data in {file_path}")
            continue

        raw_time = np.arange(len(raw_data)) / sampling_rate

        final_mask, ste_segments, rms_values = combined_ste_rms_reduction(
            raw_data, sampling_rate, ste_frame_size, ste_hop_size, ste_threshold,
            rms_window_size, global_rms_threshold, segment_threshold_factor=1.7
        )

        processed_data = raw_data[final_mask]
        processed_time = raw_time[final_mask]
        print(f"Data reduction: {100 * (1 - len(processed_data) / len(raw_data)):.2f}%")

        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)

        # Implement a warm-up period
        warm_up_samples = int(30 * sampling_rate)  # 30 seconds warm-up
        detections = detect_events(processed_data[warm_up_samples:], sampling_rate)
        detections = [(start + warm_up_samples, end + warm_up_samples) for start, end in detections]

        if is_training and catalog is not None:
            # Find matching catalog events
            file_catalog = catalog[catalog['filename'] == file_name.split('.mseed')[0]]

            if len(file_catalog) > 0:
                print(f"Found {len(file_catalog)} catalog events for {file_name}")
                file_catalog['time_rel_sec'] = file_catalog['time_rel(sec)']
            else:
                print(f"No catalog events found for {file_name}")

        plot_results(raw_data, processed_data, raw_time, processed_time, sampling_rate, detections, file_catalog if is_training else None, file_name)
        plot_results(raw_data, processed_data, raw_time, processed_time, sampling_rate, detections, file_catalog if is_training else None, file_name)
        results.append({
            'filename': file_name,
            'detections': detections,
            'sampling_rate': sampling_rate,
            'data_reduction': 100 * (1 - len(processed_data) / len(raw_data))
        })

    return results


def main():
    st.title("Seismic Data Visualization")

    kaggle_json = st.file_uploader("Upload your kaggle.json file", type="json")
    if kaggle_json:
        # Save the uploaded kaggle.json file
        with open("kaggle.json", "wb") as f:
            f.write(kaggle_json.getvalue())
        os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
        st.success("Kaggle credentials uploaded successfully!")

        # Download dataset button
        if st.button("Download Kaggle Dataset"):
            download_kaggle_dataset() 

    ste_frame_size = 1000 # 1 second frame
    ste_hop_size = 500  # 0.5 second hop
    ste_threshold = 0.5 # STE threshold factor
    rms_window_size = 100  # RMS window size
    rms_threshold = 1
    global_rms_threshold = 0.8
    # File selection
    data_dir = st.text_input("Enter the path to the data directory:", "data/lunar/training/data/S12_GradeA")
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mseed')]
    selected_file = st.selectbox("Select a file:", file_list)

    if selected_file:
        file_path = os.path.join(data_dir, selected_file)
        
        # Load and process data
        raw_data, sampling_rate, start_time = load_data(file_path)
        raw_time = np.arange(len(raw_data)) / sampling_rate

        # Process data
        final_mask, ste_segments, rms_values = combined_ste_rms_reduction(
            raw_data, sampling_rate, ste_frame_size, ste_hop_size, ste_threshold,
            rms_window_size, global_rms_threshold, segment_threshold_factor=1.7
        )

        processed_data = raw_data[final_mask]
        processed_time = raw_time[final_mask]

        # Detect events
        warm_up_samples = int(30 * sampling_rate)
        detections = detect_events(processed_data[warm_up_samples:], sampling_rate)
        detections = [(start + warm_up_samples, end + warm_up_samples) for start, end in detections]

        # Display plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # Raw data plot
        ax1.plot(raw_time, raw_data, label='Raw Data', color='blue', alpha=0.7)
        ax1.set_title("Raw Data", fontsize=16)
        ax1.set_ylabel("Amplitude", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.tick_params(labelsize=10)

        # Processed data plot
        ax2.plot(raw_time, raw_data, color='lightgray', alpha=0.5, label='Raw Data')
        ax2.plot(processed_time, processed_data, color='blue', alpha=0.7, label='Processed Data')

        # Plot detections
        for start, end in detections:
            start_time = processed_time[start]
            end_time = processed_time[end]
            ax2.axvspan(start_time, end_time, color='red', alpha=0.3)

        ax2.set_title("Processed Data with Detections", fontsize=16)
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Amplitude", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.tick_params(labelsize=10)

        plt.tight_layout()
        st.pyplot(fig)

        # Display statistics
        st.subheader("Statistics")
        st.write(f"Data reduction: {100 * (1 - len(processed_data) / len(raw_data)):.2f}%")
        st.write(f"Number of detected events: {len(detections)}")

if __name__ == "__main__":
    main()