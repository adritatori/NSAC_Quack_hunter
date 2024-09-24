import librosa
import numpy as np
import plotly.graph_objects as go
from scipy import signal
import matplotlib.pyplot as plt
import pywt

import numpy as np
import plotly.graph_objects as go
import scipy.io.wavfile as wav
import base64
from io import BytesIO
import soundfile as sf
from scipy.io import wavfile

def downsample(data, factor):
    """Downsample the data by the given factor."""
    return data[::factor]


def create_3d_mars_visualization(raw_data, processed_data, detections, sampling_rate, downsample_factor=10):
    # Downsample the data
    downsampled_raw = downsample(raw_data, downsample_factor)
    downsampled_processed = downsample(processed_data, downsample_factor)
    time = np.arange(len(downsampled_raw)) * downsample_factor / sampling_rate
    
    fig = go.Figure()
    
    # Add Mars sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_surface(x=x, y=y, z=z, colorscale=[[0, 'rgb(200, 100, 50)'], [1, 'rgb(150, 50, 0)']])
    
    # Add raw data wave
    fig.add_trace(go.Scatter3d(x=time, y=downsampled_raw, z=np.ones_like(downsampled_raw)*15,
                               mode='lines', line=dict(color='white', width=2),
                               name='Raw Data'))
    
    # Add processed data wave
    fig.add_trace(go.Scatter3d(x=time, y=downsampled_processed, z=np.ones_like(downsampled_processed)*(-15),
                               mode='lines', line=dict(color='cyan', width=2),
                               name='Processed Data'))
    
    # Add detection markers
    for start, end in detections:
        fig.add_trace(go.Scatter3d(x=[time[start//downsample_factor], time[end//downsample_factor]], 
                                   y=[downsampled_processed[start//downsample_factor], downsampled_processed[end//downsample_factor]], 
                                   z=[-15, -15],
                                   mode='markers',
                                   marker=dict(size=5, color='yellow', symbol='diamond'),
                                   name='Detected Event'))
    
    fig.update_layout(scene=dict(xaxis_title='Time [s]',
                                 yaxis_title='Amplitude',
                                 zaxis_title=''),
                      width=800, height=800, title='3D Mars Seismic Visualization')
    
    return fig

    # Noise reduction using spectral gating
    S = librosa.stft(data)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    noise_floor_db = np.mean(S_db) - 1.5 * np.std(S_db)
    mask = S_db > noise_floor_db
    S_clean = S * mask
    enhanced = librosa.istft(S_clean)
    
    # Amplitude normalization
    enhanced = librosa.util.normalize(enhanced)
    
    # Apply a high-pass filter to remove low-frequency noise
    sos = signal.butter(10, 20, 'hp', fs=sampling_rate, output='sos')
    enhanced = signal.sosfilt(sos, enhanced)
    
    # Apply a gentle boost to mid-high frequencies
    sos_eq = signal.butter(10, [100, 2000], 'bandpass', fs=sampling_rate, output='sos')
    enhanced_eq = signal.sosfilt(sos_eq, enhanced)
    enhanced = enhanced + 0.3 * enhanced_eq  # Blend original and equalized
    
    # Final normalization
    enhanced = librosa.util.normalize(enhanced)
    
    return enhanced
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

# def create_audio_file(data, sampling_rate):
#     try:
#         # Ensure data is a numpy array
#         data = np.array(data)
        
#         # Normalize the data
#         normalized_data = np.int16(data / np.max(np.abs(data)) * 32767)
        
#         # Ensure sampling_rate is an integer
#         sampling_rate = int(sampling_rate)
        
#         # Create a BytesIO buffer
#         buffer = BytesIO()
        
#         # Write the audio data to the buffer
#         sf.write(buffer, normalized_data, sampling_rate, format='wav')
        
#         # Reset the buffer position
#         buffer.seek(0)
        
#         return buffer
#     except Exception as e:
#         print(f"Error in create_audio_file: {e}")
#         print(f"Data type: {type(data)}, shape: {data.shape}, dtype: {data.dtype}")
#         print(f"Sampling rate: {sampling_rate}, type: {type(sampling_rate)}")
#         return None

# def create_audio_file(data, sampling_rate):
#     try:
#         # Ensure data is a numpy array
#         data = np.array(data)
        
#         # Normalize the data
#         normalized_data = np.int16(data / np.max(np.abs(data)) * 32767)
        
#         # Ensure sampling_rate is an integer
#         sampling_rate = int(sampling_rate)
        
#         # Create a BytesIO buffer
#         buffer = BytesIO()
        
#         # Write the audio data to the buffer
#         sf.write(buffer, normalized_data, sampling_rate, format='wav')
        
#         # Reset the buffer position
#         buffer.seek(0)
        
#         return buffer
#     except Exception as e:
#         print(f"Error in create_audio_file: {e}")
#         print(f"Data type: {type(data)}, shape: {data.shape}, dtype: {data.dtype}")
#         print(f"Sampling rate: {sampling_rate}, type: {type(sampling_rate)}")
#         return None

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

def create_event_highlight_plot(raw_time, raw_data, processed_time, processed_data, detections, selected_event_index, downsample_factor=5):
    # Create the base interactive seismic plot
    fig = create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections, downsample_factor)
    
    # Highlight the selected event
    start, end = detections[selected_event_index]
    fig.add_vrect(x0=processed_time[start], x1=processed_time[end],
                  fillcolor="yellow", opacity=0.3, layer="above", line_width=0)
    
    fig.update_layout(title=f'Seismic Data Analysis - Event {selected_event_index + 1} Highlighted')
    
    return fig

def create_3d_seismic_trace(data, sampling_rate, detections, num_traces=50, downsample_factor=10):
    # Downsample the data
    downsampled_data = downsample(data, downsample_factor)
    time = np.arange(len(downsampled_data)) * downsample_factor / sampling_rate
    
    if isinstance(num_traces, list):
        num_traces = len(num_traces)
    else:
        num_traces = int(num_traces)
    
    traces = np.arange(num_traces)
    
    # Calculate frequency content (on downsampled data)
    f, t, Sxx = signal.spectrogram(downsampled_data, fs=sampling_rate/downsample_factor, nperseg=256, noverlap=128)
    dominant_freq = f[np.argmax(Sxx, axis=0)]
    norm_freq = (dominant_freq - dominant_freq.min()) / (dominant_freq.max() - dominant_freq.min())
    
    fig = go.Figure()
    
    # Add spectrogram as a surface at the bottom of the 3D plot
    spectrogram_z = np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
    fig.add_trace(go.Surface(
        x=t,
        y=f,
        z=spectrogram_z,
        colorscale='Viridis',
        showscale=False,
        opacity=0.8
    ))
    
    for i in range(num_traces):
        start = i * len(downsampled_data) // num_traces
        end = (i + 1) * len(downsampled_data) // num_traces
        
        colors = plt.cm.viridis(norm_freq[start:end])
        
        fig.add_trace(go.Scatter3d(
            x=time[start:end],
            y=np.full_like(time[start:end], i),
            z=downsampled_data[start:end] + i*0.1,  # Add offset for better visibility
            mode='lines',
            line=dict(color=colors, width=2),
            opacity=0.6
        ))
    
    # Add markers for detected events (adjust indices for downsampled data)
    for start, end in detections:
        fig.add_trace(go.Scatter3d(
            x=[time[start//downsample_factor], time[end//downsample_factor]],
            y=[num_traces/2, num_traces/2],
            z=[downsampled_data[start//downsample_factor] + num_traces*0.05, 
               downsampled_data[end//downsample_factor] + num_traces*0.05],
            mode='markers',
            marker=dict(size=5, color='red', symbol='diamond'),
            name='Detected Event'
        ))
    
    # Create frames for animation (on downsampled data)
    frames = []
    for start, end in detections:
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=time[:end//downsample_factor],
                    y=np.full_like(time[:end//downsample_factor], i),
                    z=downsampled_data[:end//downsample_factor] + i*0.1,
                    mode='lines',
                    line=dict(color=plt.cm.viridis(norm_freq[:end//downsample_factor]), width=2),
                    opacity=0.6
                ) for i in range(num_traces)
            ] + [
                go.Scatter3d(
                    x=[time[start//downsample_factor], time[end//downsample_factor]],
                    y=[num_traces/2, num_traces/2],
                    z=[downsampled_data[start//downsample_factor] + num_traces*0.05, 
                       downsampled_data[end//downsample_factor] + num_traces*0.05],
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
        title='3D Seismic Trace Visualization with Spectrogram and Detected Events',
        scene=dict(
            xaxis_title='Time [s]',
            yaxis_title='Frequency [Hz]',
            zaxis_title='Amplitude / Log Power',
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

def create_interactive_spectrogram(data, sampling_rate, downsample_factor=5):
    # Downsample the data
    downsampled_data = downsample(data, downsample_factor)
    f, t, Sxx = signal.spectrogram(downsampled_data, fs=sampling_rate/downsample_factor)
    
    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx),
        x=t,
        y=f,
        colorscale='Jet',
        zmin=-100,  # Adjust as needed
        zmax=0,     # Adjust as needed
    ))
    
    fig.update_layout(
        title='Interactive Spectrogram (Downsampled)',
        xaxis_title='Time [s]',
        yaxis_title='Frequency [Hz]',
        yaxis_type='log',
        height=600,
    )
    
    return fig

def create_interactive_seismic_plot(raw_time, raw_data, processed_time, processed_data, detections, downsample_factor=5):
    # Downsample the data
    raw_time_ds = downsample(raw_time, downsample_factor)
    raw_data_ds = downsample(raw_data, downsample_factor)
    processed_time_ds = downsample(processed_time, downsample_factor)
    processed_data_ds = downsample(processed_data, downsample_factor)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=raw_time_ds, y=raw_data_ds, mode='lines', name='Raw Data'))
    fig.add_trace(go.Scatter(x=processed_time_ds, y=processed_data_ds, mode='lines', name='Processed Data'))
    
    for start, end in detections:
        fig.add_vrect(
            x0=processed_time[start], x1=processed_time[end],
            fillcolor="red", opacity=0.2, layer="below", line_width=0
        )
    
    fig.update_layout(
        title='Seismic Data Analysis (Downsampled)',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_time_frequency_analysis(data, sampling_rate, downsample_factor=5):
    # Downsample the data
    downsampled_data = downsample(data, downsample_factor)
    downsampled_sampling_rate = sampling_rate / downsample_factor
    
    scales = np.arange(1, 128)
    wavelet = 'morl'
    coeffs, freqs = pywt.cwt(downsampled_data, scales, wavelet, 1/downsampled_sampling_rate)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(np.abs(coeffs), extent=[0, len(downsampled_data)/downsampled_sampling_rate, freqs[-1], freqs[0]], 
                   aspect='auto', cmap='jet')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Time-Frequency Analysis (Continuous Wavelet Transform) - Downsampled')
    fig.colorbar(im, ax=ax, label='Magnitude')
    return fig