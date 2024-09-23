import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from obspy import read
from scipy import signal
import pywt
import base64
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_processing import load_data, combined_ste_rms_reduction
from event_detection import detect_events, ensemble_detection, sliding_window_detection
from visualization import plot_results
from utils import check_empty_data

# Load metadata
@st.cache_data
def load_metadata(metadata_file):
    with open(metadata_file, 'r') as f:
        return json.load(f)


def calculate_snr(data, sampling_rate):
    freqs, psd = signal.welch(data, sampling_rate)
    signal_power = np.sum(psd)
    noise_power = signal_power - np.max(psd)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    return snr

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
def create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Raw Seismic Data", "Processed Seismic Data with Detections"))
    
    fig.add_trace(go.Scatter(x=raw_time, y=raw_data, mode='lines', name='Raw Data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=processed_time, y=processed_data, mode='lines', name='Processed Data'), row=2, col=1)
    
    for start, end in detections:
        fig.add_vrect(x0=processed_time[start], x1=processed_time[end],
                      fillcolor="red", opacity=0.2, layer="below", line_width=0, row=2, col=1)
    
    fig.update_layout(height=800, title_text="Seismic Data Analysis")
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    
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

def main():
    st.set_page_config(page_title="Seismic Data Analyzer", page_icon="🌋", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {background-color: #f0f2f6;}
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3, p, div {color: #1e3a8a;}
    .stSelectbox label, .stSlider label {color: #1e3a8a; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🌋 Seismic Data Analyzer")

    # Load metadata
    metadata = load_metadata('metadata.json')
    df_metadata = pd.DataFrame(metadata)

    # Sidebar for dataset overview and filtering
    st.sidebar.header("Dataset Overview")
    st.sidebar.write(f"Total files: {len(df_metadata)}")
    st.sidebar.write(f"Total duration: {df_metadata['num_samples'].sum() / df_metadata['sampling_rate'].mean():.2f} seconds")
    st.sidebar.write(f"Total data size: {df_metadata['file_size'].sum():.2f} MB")

    # Filters
    st.sidebar.header("Filters")
    selected_station = st.sidebar.selectbox("Select Station", ['All'] + list(df_metadata['station'].unique()))
    if selected_station != 'All':
        df_metadata = df_metadata[df_metadata['station'] == selected_station]
     
    # Sidebar for processing parameters
    st.sidebar.header("Processing Parameters")
    ste_frame_size = st.sidebar.slider("STE Frame Size", 100, 2000, 1000)
    ste_hop_size = st.sidebar.slider("STE Hop Size", 100, 1000, 500)
    ste_threshold = st.sidebar.slider("STE Threshold", 0.1, 1.0, 0.5)
    rms_window_size = st.sidebar.slider("RMS Window Size", 10, 500, 100)
    global_rms_threshold = st.sidebar.slider("Global RMS Threshold", 0.1, 2.0, 0.8)
    segment_threshold_factor = st.sidebar.slider("Segment Threshold Factor", 0.5, 2.0, 1.7)


    # Sidebar for detection parameters
    st.sidebar.header("Detection Parameters")
    detection_method = st.sidebar.selectbox("Detection Method", ["STA/LTA", "Sliding Window", "Ensemble"])

    if detection_method == "STA/LTA":
        sta_len = st.sidebar.slider("STA Length (seconds)", 1, 10, 5)
        lta_len = st.sidebar.slider("LTA Length (seconds)", 10, 100, 50)
        thr_on = st.sidebar.slider("Trigger On Threshold", 1.0, 5.0, 3.0)
        thr_off = st.sidebar.slider("Trigger Off Threshold", 0.5, 2.0, 1.0)
        min_dur = st.sidebar.slider("Minimum Duration (seconds)", 0.5, 5.0, 2.0)
    elif detection_method == "Sliding Window":
        window_size = st.sidebar.slider("Window Size", 500, 2000, 1500)
        factor = st.sidebar.slider("Factor", 1.0, 5.0, 2.0)
        min_change = st.sidebar.slider("Minimum Change", 0.01, 0.2, 0.07)



    # Main content
    st.header("File Selection")
    selected_file = st.selectbox("Select a file to analyze:", df_metadata['filename'])

    if selected_file:
        file_metadata = df_metadata[df_metadata['filename'] == selected_file].iloc[0]
        
        # Display file metadata
        st.subheader("File Metadata")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Start Time: {file_metadata['start_time']}")
            st.write(f"Sampling Rate: {file_metadata['sampling_rate']} Hz")
            st.write(f"Channel: {file_metadata['channel']}")
        with col2:
            st.write(f"End Time: {file_metadata['end_time']}")
            st.write(f"Number of Samples: {file_metadata['num_samples']}")
            st.write(f"Station: {file_metadata['station']}")
        with col3:
            st.write(f"Duration: {file_metadata['num_samples'] / file_metadata['sampling_rate']:.2f} s")
            st.write(f"File Size: {file_metadata['file_size']:.2f} MB")
            st.write(f"Network: {file_metadata['network']}")

        # Display thumbnail
        st.image(BytesIO(base64.b64decode(file_metadata['thumbnail'])), caption="Signal Overview")

        # Display pre-computed metrics
        st.subheader("Signal Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Statistics:")
            for key, value in file_metadata['statistics'].items():
                st.write(f"{key.capitalize()}: {value:.2f}")
        with col2:
            st.write("Spectral Information:")
            for key, value in file_metadata['spectral_info'].items():
                st.write(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")

        st.write("Quality Metrics:")
        st.write(f"SNR: {file_metadata['quality_metrics']['snr']:.2f} dB")
        st.write(f"Data Completeness: {file_metadata['quality_metrics']['data_completeness']:.2%}")
        
        # Option to load full data for detailed analysis
        if st.button("Perform Detailed Analysis"):
            file_path = os.path.join(file_metadata['directory'], file_metadata['relative_path'])
            raw_data, sampling_rate, _ = load_data(file_path)
            raw_time = np.arange(len(raw_data)) / sampling_rate

            # Calculate SNR before processing
            snr_before = calculate_snr(raw_data,sampling_rate)

            # Process data
            final_mask, _, _ = combined_ste_rms_reduction(
                raw_data, sampling_rate, ste_frame_size, ste_hop_size, ste_threshold,
                rms_window_size, global_rms_threshold, segment_threshold_factor
            )
            processed_data = raw_data[final_mask]
            processed_time = raw_time[final_mask]

            # Calculate SNR after processing
            snr_after = calculate_snr(processed_data,sampling_rate)

            # Detect events based on selected method
            if detection_method == "STA/LTA":
                detections = detect_events(processed_data, sampling_rate, sta_len, lta_len, thr_on, thr_off, min_dur)
            elif detection_method == "Sliding Window":
                detections = sliding_window_detection(processed_data, window_size, factor, min_change)
            else:  # Ensemble
                detections = ensemble_detection(processed_data, sampling_rate)

            # Display analytics
            st.subheader("📊 Analysis Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("SNR Before", f"{snr_before:.2f} dB")
            with col2:
                st.metric("SNR After", f"{snr_after:.2f} dB")
            with col3:
                st.metric("Detected Events", len(detections))
            with col4:
                data_reduction = (1 - len(processed_data) / len(raw_data)) * 100
                st.metric("Data Reduction", f"{data_reduction:.2f}%")

            # Create and display interactive plot
            fig = create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections)
            st.plotly_chart(fig, use_container_width=True)  
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