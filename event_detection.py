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