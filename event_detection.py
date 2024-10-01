import numpy as np
from obspy.signal.trigger import recursive_sta_lta
import logging

logging.basicConfig(level=logging.INFO)

def detect_events(data, sampling_rate, params):
    try:
        sta_samples = int(params["sta"] * sampling_rate)
        lta_samples = int(params["lta"] * sampling_rate)

        cft = recursive_sta_lta(data, sta_samples, lta_samples)

        thr_on_adaptive = np.mean(cft) + params["thr_on"] * np.std(cft)
        thr_off_adaptive = np.mean(cft) + params["thr_off"] * np.std(cft)

        triggers = np.where((cft > thr_on_adaptive) & (np.roll(cft, 1) <= thr_on_adaptive))[0]
        detriggers = np.where((cft < thr_off_adaptive) & (np.roll(cft, 1) >= thr_off_adaptive))[0]

        events = []
        for i, start in enumerate(triggers):
            end_candidates = detriggers[detriggers > start]
            if len(end_candidates) > 0:
                end = end_candidates[0]
                duration = (end - start) / sampling_rate
                if duration >= params["min_dur"]:
                    max_cft = np.max(cft[start:end])
                    events.append((start, end, max_cft))

        logging.info(f"Detected {len(events)} events with parameters: {params}")
        return events
    except Exception as e:
        logging.error(f"Error in detect_events: {str(e)}")
        return []

def multi_scale_event_detection(data, sampling_rate, detector_params):
    all_events = []
    for params in detector_params:
        events = detect_events(data, sampling_rate, params)
        all_events.extend(events)

    if not all_events:
        logging.warning("No events detected. Using fallback method.")
        # Fallback method: simple threshold-based detection
        threshold = np.mean(data) + 2 * np.std(data)
        above_threshold = np.where(data > threshold)[0]
        events = np.split(above_threshold, np.where(np.diff(above_threshold) != 1)[0] + 1)
        all_events = [(e[0], e[-1], np.max(data[e[0]:e[-1]])) for e in events if len(e) > sampling_rate * 0.5]

    energy = np.square(data)
    background_energy = np.median(energy)

    classified_events = []
    for start, end, max_cft in all_events:
        event_energy = np.mean(energy[start:end])
        energy_ratio = event_energy / background_energy

        if energy_ratio > 5.0 and max_cft > 3.0:
            event_type = "Large Event"
        elif energy_ratio > 2.5 and max_cft > 2.0:
            event_type = "Moderate Event"
        elif energy_ratio > 1.5 and max_cft > 1.5:
            event_type = "Small Event"
        else:
            event_type = "Micro Event"

        classified_events.append({
            "start": start,
            "end": end,
            "type": event_type,
            "energy_ratio": energy_ratio,
            "max_cft": max_cft,
            "detector": f"STA:{params['sta']},LTA:{params['lta']}"
        })

    # Sort events by start time and merge overlapping events
    classified_events.sort(key=lambda x: x['start'])
    merged_events = []
    for event in classified_events:
        if not merged_events or event['start'] > merged_events[-1]['end']:
            merged_events.append(event)
        else:
            # Merge overlapping events, keeping the "larger" classification
            merged_events[-1]['end'] = max(merged_events[-1]['end'], event['end'])
            merged_events[-1]['type'] = max(merged_events[-1]['type'], event['type'],
                                            key=lambda x: ["Micro Event", "Small Event", "Moderate Event", "Large Event"].index(x))

    logging.info(f"Final number of events after merging: {len(merged_events)}")
    return merged_events