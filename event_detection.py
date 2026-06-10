from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_id",
    "start_s",
    "end_s",
    "duration_ms",
    "predicted_class",
    "frame_count",
    "peak_confidence",
    "mean_confidence",
]


@dataclass
class _Event:
    start_s: float
    end_s: float
    predicted_class: str
    confidences: list[float]

    @property
    def frame_count(self) -> int:
        return len(self.confidences)

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end_s - self.start_s) * 1000.0)

    @property
    def peak_confidence(self) -> float:
        return float(np.max(self.confidences)) if self.confidences else 0.0

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _normalize_label(label: str) -> str:
    return str(label).strip().casefold()


def _frame_step_seconds(frame_times: np.ndarray) -> float:
    if len(frame_times) >= 2:
        diffs = np.diff(frame_times.astype(float))
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs):
            return float(np.median(positive_diffs))
    return 0.01


def _class_confidence(
    frame_index: int,
    predicted_class: str,
    probabilities: np.ndarray,
    class_index: dict[str, int],
) -> float:
    index = class_index.get(_normalize_label(predicted_class))
    if index is None:
        return float(np.max(probabilities[frame_index]))
    return float(probabilities[frame_index, index])


def _merge_events(events: list[_Event], merge_gap_ms: int) -> list[_Event]:
    if not events:
        return []

    merged = [events[0]]
    max_gap_s = merge_gap_ms / 1000.0
    for event in events[1:]:
        previous = merged[-1]
        gap_s = event.start_s - previous.end_s
        if event.predicted_class == previous.predicted_class and gap_s <= max_gap_s:
            previous.end_s = max(previous.end_s, event.end_s)
            previous.confidences.extend(event.confidences)
        else:
            merged.append(event)
    return merged


def detect_events(
    frame_times,
    predicted_classes,
    probabilities,
    class_names,
    normal_class="Normal",
    confidence_threshold=0.70,
    min_duration_ms=20,
    merge_gap_ms=20,
) -> pd.DataFrame:
    frame_times = np.asarray(frame_times, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted_classes = list(predicted_classes)
    class_names = list(class_names)

    if len(frame_times) == 0 or not predicted_classes or probabilities.size == 0:
        return _empty_events()
    if len(frame_times) != len(predicted_classes) or probabilities.shape[0] != len(predicted_classes):
        raise ValueError("frame_times, predicted_classes, and probabilities must have the same frame count.")

    normal_label = _normalize_label(normal_class)
    class_index = {_normalize_label(name): index for index, name in enumerate(class_names)}
    frame_step_s = _frame_step_seconds(frame_times)

    raw_events: list[_Event] = []
    active_event: _Event | None = None

    for frame_index, predicted_class in enumerate(predicted_classes):
        confidence = _class_confidence(frame_index, predicted_class, probabilities, class_index)
        is_artifact = _normalize_label(predicted_class) != normal_label and confidence >= confidence_threshold

        if not is_artifact:
            if active_event is not None:
                raw_events.append(active_event)
                active_event = None
            continue

        start_s = float(frame_times[frame_index])
        end_s = start_s + frame_step_s
        if active_event is None or active_event.predicted_class != predicted_class:
            if active_event is not None:
                raw_events.append(active_event)
            active_event = _Event(start_s=start_s, end_s=end_s, predicted_class=predicted_class, confidences=[confidence])
        else:
            active_event.end_s = end_s
            active_event.confidences.append(confidence)

    if active_event is not None:
        raw_events.append(active_event)

    merged_events = _merge_events(raw_events, int(merge_gap_ms))
    filtered_events = [
        event for event in merged_events if event.duration_ms + 1e-9 >= float(min_duration_ms)
    ]

    rows = []
    for event_id, event in enumerate(filtered_events, start=1):
        rows.append(
            {
                "event_id": event_id,
                "start_s": round(event.start_s, 4),
                "end_s": round(event.end_s, 4),
                "duration_ms": round(event.duration_ms, 2),
                "predicted_class": event.predicted_class,
                "frame_count": int(event.frame_count),
                "peak_confidence": round(event.peak_confidence, 4),
                "mean_confidence": round(event.mean_confidence, 4),
            }
        )

    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
