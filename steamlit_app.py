import altair as alt
import pandas as pd
import streamlit as st
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
from sklearn.metrics import precision_recall_fscore_support
from data_processing import load_data, combined_ste_rms_reduction
from event_detection import detect_events
from visualization import plot_results
from utils import check_empty_data


def calculate_snr(data, sampling_rate, freq_bands=None, time_windows=None):
    freqs, psd = signal.welch(data, sampling_rate)
    signal_power = np.sum(psd)
    noise_power = signal_power - np.max(psd)
    snr_overall = 10 * np.log10(signal_power / noise_power)
    
    results = {"Overall SNR": f"{snr_overall:.2f} dB"}
    
    if freq_bands:
        for f_min, f_max in freq_bands:
            band_mask = (freqs >= f_min) & (freqs <= f_max)
            if np.any(band_mask):
                signal_power_band = np.sum(psd[band_mask])
                noise_power_band = signal_power_band - np.max(psd[band_mask])
                snr_band = 10 * np.log10(signal_power_band / noise_power_band)
                results[f"SNR in band {f_min}-{f_max} Hz"] = f"{snr_band:.2f} dB"
    
    if time_windows:
        time = np.arange(len(data)) / sampling_rate
        for t_min, t_max in time_windows:
            window_mask = (time >= t_min) & (time <= t_max)
            if np.any(window_mask):
                _, psd_window = signal.welch(data[window_mask], sampling_rate)
                signal_power_window = np.sum(psd_window)
                noise_power_window = signal_power_window - np.max(psd_window)
                snr_window = 10 * np.log10(signal_power_window / noise_power_window)
                results[f"SNR in time window {t_min}-{t_max} s"] = f"{snr_window:.2f} dB"
    
    return results

def create_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(raw_time, raw_data, color='blue', alpha=0.7, label='Raw Data')
    ax1.set_title("Raw Seismic Data")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    
    ax2.plot(raw_time, raw_data, color='lightgray', alpha=0.5, label='Raw Data')
    ax2.plot(processed_time, processed_data, color='green', alpha=0.7, label='Processed Data')
    
    for start, end in detections:
        ax2.axvspan(processed_time[start], processed_time[end], color='red', alpha=0.3)
    
    ax2.set_title("Processed Seismic Data with Detections")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_frequency_analysis(data, sampling_rate):
    f, Pxx = signal.periodogram(data, sampling_rate)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(f, Pxx)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.set_title('Frequency Analysis')
    ax.grid()
    return fig

def plot_time_frequency_analysis(data, sampling_rate):
    scales = np.arange(1, 128)
    wavelet = 'morl'
    coeffs, freqs = pywt.cwt(data, scales, wavelet, 1/sampling_rate)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(np.abs(coeffs), extent=[0, len(data)/sampling_rate, freqs[-1], freqs[0]], 
                   aspect='auto', cmap='jet')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Time-Frequency Analysis (Continuous Wavelet Transform)')
    fig.colorbar(im, ax=ax, label='Magnitude')
    return fig

# Streamlit app
def main():
    st.set_page_config(page_title="Seismic Data Analyzer", page_icon="🌋", layout="wide")
    
    # Custom CSS to make the app more attractive
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1e3a8a;
    }
    h2 {
        color: #2563eb;
    }
    p,h3, div{
          color: #1e3a8a;      }      
                      
    .stSelectbox label, .stSlider label {
        color: #1e3a8a;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌋 Seismic Data Analyzer")

    # Sidebar for controls
    st.sidebar.header("Settings")
    data_dir = st.sidebar.text_input("Data Directory", "data/lunar/training/data/S12_GradeA")
    
    # Parameters
    ste_frame_size = st.sidebar.slider("STE Frame Size", 100, 2000, 1000)
    ste_hop_size = st.sidebar.slider("STE Hop Size", 100, 1000, 500)
    ste_threshold = st.sidebar.slider("STE Threshold", 0.1, 1.0, 0.5)
    rms_window_size = st.sidebar.slider("RMS Window Size", 10, 500, 100)
    global_rms_threshold = st.sidebar.slider("Global RMS Threshold", 0.1, 2.0, 0.8)

    # File selection
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mseed')]
    selected_file = st.selectbox("Select a seismic data file:", file_list)

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

        # Create Matplotlib figure
        fig = create_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections)

        # Display the plot
        st.pyplot(fig)

        # Add interactivity with Streamlit
        st.header("🔍 Interactive Data Exploration")
        start_time = st.slider("Start Time (s)", min_value=0.0, max_value=raw_time[-1], value=0.0, step=1.0)
        duration = st.slider("Duration (s)", min_value=1.0, max_value=60.0, value=10.0, step=1.0)
        
        # Update plot based on selected time range
        mask = (raw_time >= start_time) & (raw_time < start_time + duration)
        fig_zoomed = create_seismic_plot(
            raw_time[mask], raw_data[mask],
            processed_time[(processed_time >= start_time) & (processed_time < start_time + duration)],
            processed_data[(processed_time >= start_time) & (processed_time < start_time + duration)],
            [(start, end) for start, end in detections if start_time <= processed_time[start] < start_time + duration]
        )
        st.pyplot(fig_zoomed)

        # Display statistics
        st.header("📊 Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Reduction", f"{100 * (1 - len(processed_data) / len(raw_data)):.2f}%")
        with col2:
            st.metric("Detected Events", len(detections))
        with col3:
            st.metric("Duration", f"{raw_time[-1]:.2f} s")

        # SNR Analysis
        st.subheader("Signal-to-Noise Ratio (SNR) Analysis")
        snr_raw = calculate_snr(raw_data, sampling_rate)
        snr_processed = calculate_snr(processed_data, sampling_rate)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Raw Data SNR:")
            for key, value in snr_raw.items():
                st.write(f"{key}: {value}")
        with col2:
            st.write("Processed Data SNR:")
            for key, value in snr_processed.items():
                st.write(f"{key}: {value}")

        # Frequency Analysis
        st.subheader("Frequency Analysis")
        freq_fig = plot_frequency_analysis(processed_data, sampling_rate)
        st.pyplot(freq_fig)

        # Time-Frequency Analysis
        st.subheader("Time-Frequency Analysis")
        tf_fig = plot_time_frequency_analysis(processed_data, sampling_rate)
        st.pyplot(tf_fig)

        # Event details
        if detections:
            st.subheader("🔍 Detected Events")
            event_df = pd.DataFrame({
                'Start Time (s)': [processed_time[start] for start, _ in detections],
                'End Time (s)': [processed_time[end] for _, end in detections],
                'Duration (s)': [processed_time[end] - processed_time[start] for start, end in detections]
            })
            st.dataframe(event_df)

if __name__ == "__main__":
    main()