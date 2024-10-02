import librosa
import numpy as np
import plotly.graph_objects as go
from scipy import signal
import matplotlib.pyplot as plt
import pywt
from matplotlib import cm
import plotly.graph_objects as go
import scipy.io.wavfile as wav
import base64
from io import BytesIO
import soundfile as sf
from scipy.io import wavfile
from scipy.stats import zscore

def validate_data(data):
    if len(data) == 0:
        raise ValueError("Input data is empty")
    return data

# 2. Implement downsampling for large datasets
def adaptive_downsample(data, max_points=100000):
    if len(data) > max_points:
        downsample_factor = len(data) // max_points
        return data[::downsample_factor]
    return data

def downsample(data, factor):
    """Downsample the data by the given factor."""
    return data[::factor]
def create_audio_file(data, sampling_rate, octave):
    
    try:
        # Ensure data is a numpy array
        data = np.array(data)
        S = librosa.stft(data)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        noise_floor_db = np.mean(S_db) - 1.5 * np.std(S_db)
        mask = S_db > noise_floor_db
        S_clean = S * mask
        enhanced = librosa.istft(S_clean)
        enhanced = librosa.util.normalize(enhanced)
        
        # Resample the audio to a standard rate for better playback
        resampled_data = librosa.resample(enhanced, orig_sr=sampling_rate, target_sr=sampling_rate*octave)
        
        # Create a BytesIO buffer
        buffer = BytesIO()
        
        # Write the audio data to the buffer
        sf.write(buffer, resampled_data, 22050, format='wav')
        
        # Reset the buffer position
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        print(f"Error in create_audio_file: {e}")
        print(f"Data type: {type(data)}, shape: {data.shape}, dtype: {data.dtype}")
        print(f"Sampling rate: {sampling_rate}, type: {type(sampling_rate)}")
        return None

def create_audio_player(data, sampling_rate, label):
    # Normalize audio data
    normalized_data = np.int16(data / np.max(np.abs(data)) * 32767)
    
    # Create a BytesIO object to store the WAV file
    buf = BytesIO()
    wav.write(buf, sampling_rate, normalized_data)
    buf.seek(0)
    
    # Create a base64 encoded string of the WAV file
    b64 = base64.b64encode(buf.read()).decode()
    
    # Create HTML audio element
    audio_html = f'<audio controls><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    
    return f"**{label} Audio:**\n\n{audio_html}"


def create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections, downsample_factor=5):
    # Validate and downsample data
    raw_data = validate_data(raw_data)
    processed_data = validate_data(processed_data)
    
    raw_data_ds = adaptive_downsample(raw_data)
    processed_data_ds = adaptive_downsample(processed_data)
    raw_time_ds = raw_time[:len(raw_data_ds)]
    processed_time_ds = processed_time[:len(processed_data_ds)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=raw_time_ds, y=raw_data_ds, mode='lines', name='Raw Data'))
    fig.add_trace(go.Scatter(x=processed_time_ds, y=processed_data_ds, mode='lines', name='Processed Data'))
    
    
    # Color mapping for event types
    color_map = {
        "Large Event": "red",
        "Moderate Event": "orange",
        "Small Event": "yellow",
        "Micro Event": "green"
    }
    
    for event in detections:
        start, end = event['start'], event['end']
        event_type = event['type']
        energy_ratio = event['energy_ratio']
        max_cft = event['max_cft']
        
        fig.add_vrect(
            x0=processed_time[start], x1=processed_time[end],
            fillcolor=color_map[event_type], opacity=0.2, layer="below", line_width=0
        )
        
        fig.add_annotation(
            x=(processed_time[start] + processed_time[end]) / 2,
            y=max(processed_data_ds[start//downsample_factor:end//downsample_factor]),
            text=f"{event_type}<br>ER: {energy_ratio:.2f}<br>CFT: {max_cft:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color_map[event_type],
            font=dict(size=10),
            bgcolor="white",
            opacity=0.8
        )
    
    fig.update_layout(
        title='Seismic Data Analysis with Multi-scale Event Detection',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
  
    return fig



def create_3d_seismic_trace(data, sampling_rate, detections, downsample_factor=10):
    # Downsample the data
    downsampled_data = data[::downsample_factor]
    time = np.arange(len(downsampled_data)) * downsample_factor / sampling_rate
    
    # Calculate frequency content (on downsampled data)
    f, t, Sxx = signal.spectrogram(downsampled_data, fs=sampling_rate/downsample_factor, nperseg=256, noverlap=128)
    
    # Normalize the spectrogram
    Sxx_norm = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
    
    fig = go.Figure()
    
    # Add spectrogram as a surface at the bottom of the 3D plot
    spectrogram_z = np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
    fig.add_trace(go.Surface(
        x=t, y=f, z=spectrogram_z,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Spectrogram Power (dB)", x=1.1, y=0.15, len=0.3),
        opacity=0.8
    ))
    
    # Calculate z-score of the data for better visualization
    z_scored_data = zscore(downsampled_data)
    
    # Create a custom colorscale for amplitude
    colorscale = [
        [0, "blue"],
        [0.25, "cyan"],
        [0.5, "yellow"],
        [0.75, "orange"],
        [1, "red"]
    ]
    
    # Add 3D scatter plot for seismic trace
    fig.add_trace(go.Scatter3d(
        x=time, y=np.zeros_like(time), z=z_scored_data,
        mode='lines',
        line=dict(color=z_scored_data, colorscale='RdBu', width=5),
        name='Seismic Trace'
    ))
    
    # Add a separate trace for the colorbar
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            colorscale='RdBu',
            showscale=True,
            cmin=np.min(z_scored_data),
            cmax=np.max(z_scored_data),
            colorbar=dict(title="Amplitude (Z-score)", x=1.1, y=0.5, len=0.3)
        ),
        hoverinfo='none'
    ))
    
    # Color mapping for event types
    color_map = {
        "Large Event": "red",
        "Moderate Event": "orange",
        "Small Event": "yellow",
        "Micro Event": "green"
    }
    
    # Add markers for detected events
    for event in detections:
        start, end = event['start'] // downsample_factor, event['end'] // downsample_factor
        event_type = event['type']
        energy_ratio = event['energy_ratio']
        max_cft = event['max_cft']
        
        # Calculate mean frequency and amplitude for the event
        event_freq = np.mean(f[np.argmax(Sxx_norm[:, start:end], axis=0)])
        event_amp = np.mean(z_scored_data[start:end])
        
        fig.add_trace(go.Scatter3d(
            x=[time[start], time[end]],
            y=[event_freq, event_freq],
            z=[event_amp, event_amp],
            mode='markers',
            marker=dict(size=12, color=color_map[event_type], symbol='diamond', line=dict(color='black', width=2)),
            name=f'{event_type} (ER: {energy_ratio:.2f}, CFT: {max_cft:.2f})'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Enhanced 3D Seismic Trace Visualization with Multi-scale Event Detection',
            font=dict(size=24)
        ),
        scene=dict(
            xaxis_title=dict(text='Time (s)', font=dict(size=16)),
            yaxis_title=dict(text='Frequency (Hz)', font=dict(size=16)),
            zaxis_title=dict(text='Amplitude (Z-score)', font=dict(size=16)),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5))
        ),
        height=800,
        margin=dict(r=20, l=10, b=10, t=40),
        legend=dict(
            font=dict(size=14),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.7)"
        )
    )

    # Add clearer annotations
    annotations = [
        dict(x=0.02, y=1, text="Seismic Trace: 3D line shows amplitude over time", font=dict(size=14)),
        dict(x=0.02, y=0.96, text="Bottom Surface: Spectrogram (frequency content over time)", font=dict(size=14)),
        dict(x=0.02, y=0.92, text="Diamond Markers: Detected events (time, frequency, and type)", font=dict(size=14))
    ]
    for annot in annotations:
        fig.add_annotation(
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="white",
            opacity=0.8,
            **annot
        )

    return fig
def create_comprehensive_plot(raw_data, processed_data, raw_time, processed_time, sampling_rate, classified_events, file_name):
    # Define color scheme for event types
    event_colors = {
        'Large Event': '#FF1493',  # Deep Pink
        'Moderate Event': '#FFD700',  # Gold
        'Small Event': '#32CD32',  # Lime Green
        'Micro Event': '#00BFFF',  # Deep Sky Blue
    }

    # Create figure with four subplots
    fig, (ax_raw, ax_processed, ax_spec_raw, ax_spec_proc) = plt.subplots(4, 1, figsize=(15, 20))

    # Plot time series data with events
    plot_time_series_with_events(ax_raw, raw_time, raw_data, "Raw Data", file_name, classified_events, event_colors, is_raw_data=True)
    plot_time_series_with_events(ax_processed, processed_time, processed_data, "Processed Data", file_name, classified_events, event_colors, is_raw_data=False)

    # Create spectrograms with events
    plot_spectrogram_with_events(ax_spec_raw, raw_data, sampling_rate, raw_time, "Raw Data Spectrogram")
    plot_spectrogram_with_events(ax_spec_proc, processed_data, sampling_rate, processed_time, "Processed Data Spectrogram")

    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_time_series_with_events(ax, time, data, title, file_name, classified_events, event_colors, is_raw_data):
    if is_raw_data:
        ax.plot(time, data, label='Raw Data', color='blue', alpha=0.7)
        ax.set_title(f"Raw Data - {file_name}", fontsize=16)
    else:
        ax.plot(time, data, label=title, color='blue', alpha=0.7)
        ax.set_title(f"{title} - {file_name}", fontsize=16)
    
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.tick_params(labelsize=10)
    
    for event in classified_events:
        start_time = time[event['start']]
        end_time = time[event['end']]
        ax.axvspan(start_time, end_time, color=event_colors.get(event['type'], '#808080'), alpha=0.4, label=event['type'])
        
        if 'detector' in event:
            mid_time = (start_time + end_time) / 2
            ax.text(mid_time, ax.get_ylim()[1], event['detector'], rotation=90,
                    verticalalignment='bottom', fontsize=8, alpha=0.7)
    
    # Create legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper right')

def plot_spectrogram_with_events(ax, data, sampling_rate, time, title):
    f, t, sxx = signal.spectrogram(data, sampling_rate)
    im = ax.pcolormesh(t, f, 10 * np.log10(sxx), cmap=cm.jet, shading='auto')
    ax.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)', fontweight='bold')
    
    # Create legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper right')

def create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections, downsample_factor=5):
    # Downsample the data
    raw_time_ds = downsample(raw_time, downsample_factor)
    raw_data_ds = downsample(raw_data, downsample_factor)
    processed_time_ds = downsample(processed_time, downsample_factor)
    processed_data_ds = downsample(processed_data, downsample_factor)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=raw_time_ds, y=raw_data_ds, mode='lines', name='Raw Data'))
    fig.add_trace(go.Scatter(x=processed_time_ds, y=processed_data_ds, mode='lines', name='Processed Data'))
    
    # Color mapping for event types
    color_map = {
        "Large Event": "#FF1493",  # Deep Pink
        "Moderate Event": "#FFD700",  # Gold
        "Small Event": "#32CD32",  # Lime Green
        "Micro Event": "#00BFFF",  # Deep Sky Blue
    }
    
    for event in detections:
        start, end = event['start'], event['end']
        event_type = event['type']
        energy_ratio = event['energy_ratio']
        max_cft = event['max_cft']
        
        fig.add_vrect(
            x0=processed_time[start], x1=processed_time[end],
            fillcolor=color_map[event_type], opacity=0.2, layer="below", line_width=0
        )
        
        fig.add_annotation(
            x=(processed_time[start] + processed_time[end]) / 2,
            y=max(processed_data_ds[start//downsample_factor:end//downsample_factor]),
            text=f"{event_type}<br>ER: {energy_ratio:.2f}<br>CFT: {max_cft:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color_map[event_type],
            font=dict(color="black",size=12),
            bgcolor="white",
            opacity=0.8
        )
    
    fig.update_layout(
        title='Seismic Data Analysis with Multi-scale Event Detection',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig