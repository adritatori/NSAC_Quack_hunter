import numpy as np
from obspy.signal.trigger import classic_sta_lta, trigger_onset

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

def merge_triggers(triggers, min_samples):
    if not triggers:
        return []

    merged = [triggers[0]]
    for current in triggers[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], current[1]))
        elif current[0] - merged[-1][1] < min_samples:
            merged[-1] = (merged[-1][0], current[1])
        else:
            merged.append(current)

    return [trigger for trigger in merged if trigger[1] - trigger[0] >= min_samples]


def sliding_window_detection(data, window_size, factor, min_change, feature_extraction='fft'):
    """
    Sliding window anomaly detection with optional feature extraction.

    Args:
        data: The input data.
        window_size: The size of the sliding window.
        factor: The factor for the local threshold.
        min_change: The minimum change for a significant event.
        feature_extraction: The type of feature extraction to use ('fft' or None).

    Returns:
        A list of detected events.
    """

    events = []

    for i in range(len(data) - window_size):
        window = data[i:i + window_size]

        # Apply feature extraction if specified
        if feature_extraction == 'fft':
            features = np.abs(np.fft.fft(window))
        else:
            features = window

        # Calculate local standard deviation and threshold
        local_std = np.std(features)
        local_threshold = factor * local_std

        # Calculate changes and identify significant changes
        changes = np.abs(np.diff(features))
        significant_changes = (changes > min_change) & (np.abs(features[1:]) > local_threshold)

        # If significant changes found, append event
        if np.any(significant_changes):
            start = i + np.argmax(significant_changes)
            end = i + window_size - 1 - np.argmax(significant_changes[::-1])
            events.append((start, end))

    return merge_triggers(events, window_size // 2)


def ensemble_detection(data, sampling_rate):
    detections2 = sliding_window_detection(data, window_size=1500, factor=2, min_change=0.07)

    all_detections = detections2
    return merge_triggers(all_detections, int(2.0 * sampling_rate))