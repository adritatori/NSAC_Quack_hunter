import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_results(raw_data, processed_data, sampling_rate, detections, file_name):
    # Create time arrays
    raw_time = np.arange(len(raw_data)) / sampling_rate
    processed_time = np.arange(len(processed_data)) / sampling_rate

    # Create the plot
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

    ax2.set_title(f"Processed Data with Detections - {file_name}", fontsize=16)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=10)

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Clear the plot to free up memory
    plt.close(fig)