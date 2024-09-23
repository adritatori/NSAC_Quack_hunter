import os
import json
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy import signal

def generate_thumbnail(data):
    fig, ax = plt.subplots(figsize=(3, 1))
    ax.plot(data)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=50, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_metadata(mseed_directories, output_file='metadata.json'):
    metadata = []
    for mseed_directory in mseed_directories:
        for root, dirs, files in os.walk(mseed_directory):
            for file in files:
                if file.endswith('.mseed'):
                    file_path = os.path.join(root, file)
                    try:
                        st = read(file_path)
                        for trace in st:
                            data = trace.data
                            sampling_rate = trace.stats.sampling_rate
                            
                            # Basic file info
                            file_info = {
                                'filename': file,
                                'directory': mseed_directory,
                                'start_time': str(trace.stats.starttime),
                                'end_time': str(trace.stats.endtime),
                                'sampling_rate': sampling_rate,
                                'num_samples': trace.stats.npts,
                                'channel': trace.stats.channel,
                                'station': trace.stats.station,
                                'network': trace.stats.network,
                                'file_size': os.path.getsize(file_path) / (1024 * 1024),  # Size in MB
                                'relative_path': os.path.relpath(file_path, mseed_directory)
                            }
                            
                            # Basic statistics
                            stats = {
                                'mean': np.mean(data),
                                'median': np.median(data),
                                'std': np.std(data),
                                'max': np.max(data),
                                'min': np.min(data),
                                'rms': np.sqrt(np.mean(np.square(data)))
                            }
                            
                            # Spectral information
                            freqs, psd = signal.welch(data, sampling_rate)
                            spectral_info = {
                                'dominant_frequency': freqs[np.argmax(psd)],
                                'spectral_centroid': np.sum(freqs * psd) / np.sum(psd)
                            }
                            
                            # Event detection summary

                            # Signal quality metrics
                            signal_power = np.sum(psd)
                            noise_power = signal_power - np.max(psd)
                            quality_metrics = {
                                'snr': 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf'),
                                'data_completeness': 1 - (np.count_nonzero(np.isnan(data)) + np.count_nonzero(data == 0)) / len(data)
                            }
                            

                            
                            # Thumbnail
                            thumbnail = generate_thumbnail(data)
                            
                            metadata.append({
                                **file_info,
                                'statistics': stats,
                                'spectral_info': spectral_info,
                                'quality_metrics': quality_metrics,
                                'thumbnail': thumbnail
                            })
                            
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}")
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {output_file}")

# Example usage
if __name__ == "__main__":
    mseed_directories = [
        "data/mars/training/data/",
        "data/mars/test/data/",
        "data/lunar/training/data/S12_GradeA",
    ]
    generate_metadata(mseed_directories)