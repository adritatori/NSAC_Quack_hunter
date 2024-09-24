import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal

def create_3d_seismic_trace(data, sampling_rate, detections, num_traces=50):
    time = np.arange(len(data)) / sampling_rate
    
    # Handle the case where num_traces might be a list
    if isinstance(num_traces, list):
        num_traces = len(num_traces)
    else:
        num_traces = int(num_traces)
    
    traces = np.arange(num_traces)
    
    # Calculate frequency content
    f, t, Sxx = signal.spectrogram(data, fs=sampling_rate, nperseg=256, noverlap=128)
    dominant_freq = f[np.argmax(Sxx, axis=0)]
    
    # Normalize frequency for coloring
    norm_freq = (dominant_freq - dominant_freq.min()) / (dominant_freq.max() - dominant_freq.min())
    
    fig = go.Figure()
    
    for i in range(num_traces):
        start = i * len(data) // num_traces
        end = (i + 1) * len(data) // num_traces
        
        # Create color array for this trace
        colors = plt.cm.viridis(norm_freq[start:end])
        
        fig.add_trace(go.Scatter3d(
            x=time[start:end],
            y=np.full_like(time[start:end], i),
            z=data[start:end],
            mode='lines',
            line=dict(color=colors, width=2),
            opacity=0.6
        ))
    
    # Add markers for detected events
    for start, end in detections:
        fig.add_trace(go.Scatter3d(
            x=[time[start], time[end]],
            y=[num_traces/2, num_traces/2],  # Place markers in the middle
            z=[data[start], data[end]],
            mode='markers',
            marker=dict(size=5, color='red', symbol='diamond'),
            name='Detected Event'
        ))
    
    # Create frames for animation, focusing on detected events
    frames = []
    for start, end in detections:
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=time[:end],
                    y=np.full_like(time[:end], i),
                    z=data[:end],
                    mode='lines',
                    line=dict(color=plt.cm.viridis(norm_freq[:end]), width=2),
                    opacity=0.6
                ) for i in range(num_traces)
            ] + [
                go.Scatter3d(
                    x=[time[start], time[end]],
                    y=[num_traces/2, num_traces/2],
                    z=[data[start], data[end]],
                    mode='markers',
                    marker=dict(size=5, color='red', symbol='diamond'),
                    name='Detected Event'
                )
            ],
            name=f'event_{start}'
        )
        frames.append(frame)
    
    fig.frames = frames
    
    fig.update_layout(
        title='3D Seismic Trace Visualization with Detected Events',
        scene=dict(
            xaxis_title='Time [s]',
            yaxis_title='Trace Number',
            zaxis_title='Amplitude',
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5))
        ),
        height=700,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}]
            }]
        }]
    )
    
    return fig
def create_interactive_spectrogram(data, sampling_rate):
    f, t, Sxx = signal.spectrogram(data, fs=sampling_rate)
    
    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx),
        x=t,
        y=f,
        colorscale='Jet',
        zmin=-100,  # Adjust as needed
        zmax=0,     # Adjust as needed
    ))
    
    fig.update_layout(
        xaxis_title='Time [s]',
        yaxis_title='Frequency [Hz]',
        yaxis_type='log',
        height=600,
    )
    
    return fig


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