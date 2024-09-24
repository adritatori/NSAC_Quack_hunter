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
from visualization import create_3d_seismic_trace, create_audio_file, create_interactive_seismic_plot, create_interactive_spectrogram, plot_time_frequency_analysis
from utils import check_empty_data
import librosa
import soundfile as sf

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

def display_audio(audio_buffer, label):
    if audio_buffer is not None:
        st.audio(audio_buffer, format='audio/wav')
        st.download_button(
            label=f"Download {label} Audio",
            data=audio_buffer,
            file_name=f"{label.lower().replace(' ', '_')}_seismic.wav",
            mime="audio/wav"
        )
    else:
        st.error(f"Failed to create {label} audio file.")
def main():
    st.set_page_config(page_title="Quack Hunter", page_icon="🌋", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-image: url('https://i.ibb.co.com/svz0WsV/quack-hunter.jpg');
        background-attachment: fixed;
        background-size: cover;
    }
    .main .block-container {
        background-color: rgba(240, 242, 246, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    h1, h2, h3 {color: #8B0000;}
    p, div {color: #1e3a8a;}
    .stSelectbox label, .stSlider label {color: #8B0000; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🌋 Quack Finder")

    # Load metadata
    metadata = load_metadata('metadata.json')
    df_metadata = pd.DataFrame(metadata)

    # Sidebar for dataset overview and filtering
    with st.sidebar:
        st.header("Dataset Overview")
        st.write(f"Total files: {len(df_metadata)}")
        st.write(f"Total duration: {df_metadata['num_samples'].sum() / df_metadata['sampling_rate'].mean():.2f} seconds")
        st.write(f"Total data size: {df_metadata['file_size'].sum():.2f} MB")

        # Filters
        st.header("Filters")
        selected_station = st.selectbox("Select Station", ['All'] + list(df_metadata['station'].unique()))

        if selected_station != 'All':
            df_metadata = df_metadata[df_metadata['station'] == selected_station]
     
        # Processing parameters
        st.header("Processing Parameters")
        ste_frame_size = st.slider("STE Frame Size", 100, 2000, 1000)
        ste_hop_size = st.slider("STE Hop Size", 100, 1000, 500)
        ste_threshold = st.slider("STE Threshold", 0.1, 1.0, 0.5)
        rms_window_size = st.slider("RMS Window Size", 10, 500, 100)
        global_rms_threshold = st.slider("Global RMS Threshold", 0.1, 2.0, 0.8)
        segment_threshold_factor = st.slider("Segment Threshold Factor", 0.5, 2.0, 1.7)

        # Detection parameters
        st.header("Detection Parameters")
        detection_method = st.selectbox("Detection Method", ["STA/LTA", "Sliding Window", "Ensemble"])

        if detection_method == "STA/LTA":
            sta_len = st.slider("STA Length (seconds)", 1, 1000, 5)
            lta_len = st.slider("LTA Length (seconds)", 10, 1000, 50)
            thr_on = st.slider("Trigger On Threshold", 1.0, 5.0, 3.0)
            thr_off = st.slider("Trigger Off Threshold", 0.5, 2.0, 1.0)
            min_dur = st.slider("Minimum Duration (seconds)", 0.5, 3.0, 10.0)
        elif detection_method == "Sliding Window":
            window_size = st.slider("Window Size", 500, 2000, 1500)
            factor = st.slider("Factor", 1.0, 5.0, 2.0)
            min_change = st.slider("Minimum Change", 0.01, 0.2, 0.07)

    # Main content
    st.header("File Selection")
    selected_file = st.selectbox("Select a file to analyze:", df_metadata['filename'])

    if selected_file:
        file_metadata = df_metadata[df_metadata['filename'] == selected_file].iloc[0]
        
        # Display file metadata
        with st.expander("File Metadata"):
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
        with st.expander("Signal Metrics"):
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
            with st.spinner("Processing data..."):
                file_path = os.path.join(file_metadata['directory'], file_metadata['relative_path'])
                raw_data, sampling_rate, _ = load_data(file_path)
                raw_time = np.arange(len(raw_data)) / sampling_rate

                # Calculate SNR before processing
                snr_before = calculate_snr(raw_data, sampling_rate)

                # Process data
                final_mask, _, _ = combined_ste_rms_reduction(
                    raw_data, sampling_rate, ste_frame_size, ste_hop_size, ste_threshold,
                    rms_window_size, global_rms_threshold, segment_threshold_factor
                )
                processed_data = raw_data[final_mask]
                processed_time = raw_time[final_mask]

                # Calculate SNR after processing
                snr_after = calculate_snr(processed_data, sampling_rate)

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

            # Interactive seismic plot
            st.subheader("Interactive Seismic Plot")
            fig = create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections, downsample_factor=10)
            st.plotly_chart(fig, use_container_width=True)

            # Audio comparison
            

            # 3D seismic trace visualization
            
            
            # Time-Frequency Analysis
            if(selected_station == 'S12'):
                st.subheader("Seismic Audio Comparison")
                raw_audio = create_audio_file(raw_data, sampling_rate, octave= 1)
                processed_audio = create_audio_file(processed_data, sampling_rate, octave = 1)

                if raw_audio is not None and processed_audio is not None:
                    st.subheader("Seismic Audio Representation")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        display_audio(raw_audio, "Raw Seismic Data")
                    with col2:
                        display_audio(processed_audio, "Processed Seismic Data")

                st.subheader("3D Seismic Trace Visualization")
                fig_3d = create_3d_seismic_trace(processed_data, sampling_rate, detections, downsample_factor=50)
                st.plotly_chart(fig_3d, use_container_width=True)

                st.subheader("Time-Frequency Analysis")
                tf_fig = plot_time_frequency_analysis(processed_data, sampling_rate, downsample_factor=20)
                st.pyplot(tf_fig)
            else:
                st.subheader("Seismic Audio Comparison")
                raw_audio = create_audio_file(raw_data, sampling_rate, octave= 4)
                processed_audio = create_audio_file(processed_data, sampling_rate, octave = 4)

                if raw_audio is not None and processed_audio is not None:
                    st.subheader("Seismic Audio Representation")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        display_audio(raw_audio, "Raw Seismic Data")
                    with col2:
                        display_audio(processed_audio, "Processed Seismic Data")

                st.subheader("3D Seismic Trace Visualization")
                fig_3d = create_3d_seismic_trace(processed_data, sampling_rate, detections, downsample_factor=10)
                st.plotly_chart(fig_3d, use_container_width=True)

                st.subheader("Time-Frequency Analysis")
                tf_fig = plot_time_frequency_analysis(processed_data, sampling_rate, downsample_factor=1)
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