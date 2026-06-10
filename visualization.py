from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import NORMAL_CLASS, display_class_name


CLASS_COLORS = {
    "normal": "#7A869A",
    "packet_loss_plc": "#0E8F8F",
    "packet_loss_no_plc": "#D97706",
    "dropout_plc": "#00A3A3",
    "burst_plc": "#2F80ED",
    "dropout_no_plc": "#F2994A",
    "burst_no_plc": "#EB5757",
    "repeated_packet": "#9B51E0",
    "encryption_noise": "#27AE60",
    "corrupted_packet": "#D946EF",
}
FALLBACK_COLORS = ("#00A3A3", "#2F80ED", "#F2994A", "#EB5757", "#9B51E0", "#27AE60")


def _class_color(class_name: str) -> str:
    key = str(class_name).casefold()
    if key in CLASS_COLORS:
        return CLASS_COLORS[key]
    return FALLBACK_COLORS[abs(hash(key)) % len(FALLBACK_COLORS)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _downsample_waveform(waveform: np.ndarray, sample_rate: int, max_points: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    audio = np.asarray(waveform, dtype=float).squeeze()
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return np.array([0.0]), np.array([0.0])

    step = max(1, int(np.ceil(audio.size / max_points)))
    indices = np.arange(0, audio.size, step)
    times = indices / float(sample_rate)
    return times, audio[indices]


def plot_waveform_with_events(waveform, sample_rate, events_df):
    times, values = _downsample_waveform(np.asarray(waveform), int(sample_rate))
    y_min = float(np.min(values)) if len(values) else -1.0
    y_max = float(np.max(values)) if len(values) else 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min, y_max = -1.0, 1.0
    padding = 0.08 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    fig = go.Figure()
    shown_classes: set[str] = set()

    if isinstance(events_df, pd.DataFrame) and not events_df.empty:
        for _, event in events_df.iterrows():
            class_name = str(event["predicted_class"])
            start_s = float(event["start_s"])
            end_s = float(event["end_s"])
            color = _class_color(class_name)
            hover = (
                f"{display_class_name(class_name)}<br>"
                f"Start: {start_s:.3f}s<br>"
                f"End: {end_s:.3f}s<br>"
                f"Duration: {float(event['duration_ms']):.0f} ms<br>"
                f"Peak confidence: {float(event['peak_confidence']):.1%}"
            )
            fig.add_trace(
                go.Scatter(
                    x=[start_s, start_s, end_s, end_s, start_s],
                    y=[y_min, y_max, y_max, y_min, y_min],
                    fill="toself",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=_hex_to_rgba(color, 0.22),
                    name=display_class_name(class_name),
                    showlegend=class_name not in shown_classes,
                    hoveron="fills",
                    hovertemplate=hover + "<extra></extra>",
                )
            )
            shown_classes.add(class_name)

    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines",
            line=dict(color="#1F2937", width=1.1),
            name="Waveform",
            hovertemplate="Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=28, r=18, t=42, b=32),
        title="Waveform with detected artifact regions",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="closest",
    )
    fig.update_yaxes(range=[y_min, y_max], zeroline=True, zerolinecolor="rgba(31,41,55,0.18)")
    return fig


def plot_confidence_over_time(frame_times, probabilities, class_names, selected_classes=None):
    frame_times = np.asarray(frame_times, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    class_names = list(class_names)

    if selected_classes is None:
        artifact_classes = [name for name in class_names if name.casefold() != NORMAL_CLASS.casefold()]
        if len(artifact_classes) <= 7:
            selected_classes = artifact_classes
        else:
            scores = [
                (name, float(np.max(probabilities[:, class_names.index(name)])))
                for name in artifact_classes
            ]
            selected_classes = [name for name, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:3]]

    fig = go.Figure()
    for class_name in selected_classes:
        if class_name not in class_names:
            continue
        class_index = class_names.index(class_name)
        fig.add_trace(
            go.Scatter(
                x=frame_times,
                y=probabilities[:, class_index],
                mode="lines",
                line=dict(color=_class_color(class_name), width=2),
                name=display_class_name(class_name),
                hovertemplate="Time: %{x:.3f}s<br>Confidence: %{y:.1%}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=28, r=18, t=42, b=32),
        title="Frame-level confidence over time",
        xaxis_title="Time (s)",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    return fig


def plot_class_distribution(predicted_classes):
    counts = Counter(predicted_classes)
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    labels = [display_class_name(name) for name, _ in ordered]
    values = [count for _, count in ordered]
    colors = [_class_color(name) for name, _ in ordered]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            hovertemplate="Class: %{x}<br>Frame count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=28, r=18, t=42, b=80),
        title="Frame count distribution, not file-level majority vote",
        xaxis_title="Predicted class",
        yaxis_title="Frame count",
    )
    fig.update_xaxes(tickangle=-25)
    return fig
