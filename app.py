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
from data_processing import combined_ste_rms_reduction
from event_detection import multi_scale_event_detection
from visualization import create_3d_seismic_trace, create_audio_file, create_comprehensive_plot, create_interactive_seismic_plot, validate_data
from utils import check_empty_data
import librosa
import soundfile as sf

def calculate_snr(data, sampling_rate):
    if len(data) == 0:
        return float('-inf')

    signal_power = np.mean(data**2)
    noise_power = np.var(data)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
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
    
    # Custom CSS (unchanged)
    st.markdown("""
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1530508777238-14544088c3ed?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-attachment: fixed;
        background-size: cover;
    }
    .main .block-container {
        background-color: rgba(32, 33, 33, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    .stSelectbox label, .stSlider label {font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🌋 Quack Finder")

    # Sidebar for processing parameters (unchanged)
    with st.sidebar:
        st.header("Processing Parameters")
        ste_frame_size = st.slider("STE Frame Size", 100, 2000, 1000)
        ste_hop_size = st.slider("STE Hop Size", 100, 1000, 500)
        ste_threshold = st.slider("STE Threshold", 0.1, 1.0, 1.5)
        rms_window_size = st.slider("RMS Window Size", 10, 500, 1000)
        global_rms_threshold = st.slider("Global RMS Threshold", 0.1, 10.0, 0.8)
        segment_threshold_factor = st.slider("Segment Threshold Factor", 0.5, 2.0, 1.1)
        st.sidebar.header("STA/LTA Parameters")
        num_detectors = st.sidebar.number_input("Number of STA/LTA detectors", min_value=1, max_value=5, value=1)

    # File upload
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload your MAR/MOONS seismic data file", type=["mseed", "sac"])

    if uploaded_file is not None:
        # Read the uploaded file
        with st.spinner("Processing uploaded data..."):
            try:
                # Save the uploaded file temporarily
                with open("temp_seismic_file", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load the data using ObsPy
                st_obj = read("temp_seismic_file")
                raw_data = st_obj[0].data
                sampling_rate = st_obj[0].stats.sampling_rate
                raw_time = np.arange(len(raw_data)) / sampling_rate
                raw_data = 2 * (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data)) - 1
                raw_time = np.arange(len(raw_data)) / sampling_rate
                #adjustment = len(raw_data) % ste_hop_size    
                # Validate data
                raw_data = validate_data(raw_data)

                # Display file metadata
                with st.expander("File Metadata"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"Start Time: {st_obj[0].stats.starttime}")
                        st.write(f"Sampling Rate: {sampling_rate} Hz")
                        st.write(f"Channel: {st_obj[0].stats.channel}")
                    with col2:
                        st.write(f"End Time: {st_obj[0].stats.endtime}")
                        st.write(f"Number of Samples: {len(raw_data)}")
                        st.write(f"Station: {st_obj[0].stats.station}")
                    with col3:
                        st.write(f"Duration: {len(raw_data) / sampling_rate:.2f} s")
                        st.write(f"Network: {st_obj[0].stats.network}")

                # Calculate SNR before processing
                snr_before = calculate_snr(raw_data, sampling_rate)

                # Process data
                with st.spinner("Processing data..."):
                    final_mask, _, _ = combined_ste_rms_reduction(
                        raw_data, sampling_rate, ste_frame_size, ste_hop_size, ste_threshold,
                        rms_window_size, global_rms_threshold, segment_threshold_factor
                    )
                    processed_data = raw_data[final_mask]
                    processed_time = raw_time[final_mask]

                # Validate processed data
                processed_data = validate_data(processed_data)

                # Calculate SNR after processing
                snr_after = calculate_snr(processed_data, sampling_rate)

                # Event detection
                with st.spinner("Detecting events..."):
                    detector_params = []
                    for i in range(num_detectors):
                        detector_params.append({
                            "sta": st.sidebar.slider(f"STA (seconds) - Detector {i+1}", 0.1, 10.0, 3.0, 0.1),
                            "lta": st.sidebar.slider(f"LTA (seconds) - Detector {i+1}", 10.0, 1000.0, 600.0, 10.0),
                            "thr_on": st.sidebar.slider(f"Trigger On Threshold - Detector {i+1}", 1.0, 5.0, 2.5, 0.1),
                            "thr_off": st.sidebar.slider(f"Trigger Off Threshold - Detector {i+1}", 0.5, 3.0, 1.8, 0.1),
                            "min_dur": st.sidebar.slider(f"Minimum Duration (seconds) - Detector {i+1}", 0.1, 5.0, 0.9, 0.1)
                        })
                    detections = multi_scale_event_detection(processed_data, sampling_rate, detector_params)

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
                try:
                    fig = create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections, downsample_factor=10)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating interactive plot: {str(e)}")
                    st.warning("The interactive plot could not be created due to data size limitations. Consider further downsampling or using a smaller dataset.")

                # Audio comparison
                st.subheader("Seismic Audio Comparison")
                try:
                    raw_audio = create_audio_file(raw_data, sampling_rate, octave=4)
                    processed_audio = create_audio_file(processed_data, sampling_rate, octave=4)

                    if raw_audio is not None and processed_audio is not None:
                        st.subheader("Seismic Audio Representation")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.audio(raw_audio, format='audio/wav')
                            st.download_button("Download Raw Audio", raw_audio, "raw_seismic.wav", "audio/wav")
                        with col2:
                            st.audio(processed_audio, format='audio/wav')
                            st.download_button("Download Processed Audio", processed_audio, "processed_seismic.wav", "audio/wav")
                    else:
                        st.warning("Audio files could not be created. The dataset might be too large for audio conversion.")
                except Exception as e:
                    st.error(f"Error creating audio files: {str(e)}")

                # Comprehensive plot
                st.subheader("Comprehensive Seismic Analysis")
                try:
                    with st.spinner("Generating comprehensive plot..."):
                        comp_fig = create_comprehensive_plot(raw_data, processed_data, raw_time, processed_time, sampling_rate, detections, uploaded_file.name)
                        st.pyplot(comp_fig)
                except Exception as e:
                    st.error(f"Error creating comprehensive plot: {str(e)}")
                    st.warning("The comprehensive plot could not be created. This might be due to memory limitations with large datasets.")

                # 3D seismic trace visualization
                # 3D seismic trace visualization
                st.subheader("3D Seismic Trace Visualization")
                try:
                    with st.spinner("Generating 3D visualization..."):
                        fig_3d = create_3d_seismic_trace(processed_data, sampling_rate, detections, downsample_factor=20)
                        st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating 3D seismic trace: {str(e)}")
                    st.warning("The 3D seismic trace could not be created. This might be due to memory limitations with large datasets.")

                # Event details
                if detections:
                    st.subheader("🔍 Detected Events")
                    event_df = pd.DataFrame(detections)
                    event_df['Start Time (s)'] = event_df['start'] / sampling_rate
                    event_df['End Time (s)'] = event_df['end'] / sampling_rate
                    event_df['Duration (s)'] = event_df['End Time (s)'] - event_df['Start Time (s)']
                    st.dataframe(event_df[['Start Time (s)', 'End Time (s)', 'Duration (s)', 'type', 'energy_ratio', 'max_cft', 'detector']])
                else:
                    st.info("No events were detected in the processed data.")

            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
                st.error("Please try uploading a different file or contact support if the issue persists.")

            finally:
                # Clean up the temporary file
                if os.path.exists("temp_seismic_file"):
                    os.remove("temp_seismic_file")

if __name__ == "__main__":
    main()