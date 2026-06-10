from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    AUDIO_SEARCH_ROOTS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MERGE_GAP_MS,
    DEFAULT_MIN_DURATION_MS,
    DEFAULT_MODEL_DIR,
    NORMAL_CLASS,
    SUPPORTED_AUDIO_EXTENSIONS,
    TEMP_UPLOAD_DIR,
    display_class_name,
    is_model_artifact_dir,
)
from event_detection import EVENT_COLUMNS, detect_events
from inference import run_inference
from operational_targets import aggregate_result_to_operational
from visualization import (
    plot_class_distribution,
    plot_confidence_over_time,
    plot_waveform_with_events,
)


st.set_page_config(page_title="RTX Audio Artifact Inference", layout="wide")


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(31, 41, 55, 0.10);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(31, 41, 55, 0.06);
        }
        [data-testid="stMetricLabel"] p {
            color: #4B5563;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #111827;
        }
        .rtx-subtitle {
            color: #4B5563;
            font-size: 1.05rem;
            margin-top: -0.6rem;
            margin-bottom: 1.1rem;
        }
        .small-muted {
            color: #6B7280;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_name(filename: str) -> str:
    stem = Path(filename).stem or "audio"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "audio"
    suffix = Path(filename).suffix.lower()
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def _save_uploaded_file(uploaded_file) -> Path:
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = TEMP_UPLOAD_DIR / _safe_name(uploaded_file.name)
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def _audio_mime(path_or_name: str | Path) -> str:
    suffix = Path(path_or_name).suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".flac":
        return "audio/flac"
    return "audio/wav"


def _demo_clip_cache_key() -> str:
    manifest = Path(__file__).resolve().parent / "demo_clips" / "demo_clips_manifest.csv"
    if not manifest.exists():
        return "no-demo-manifest"
    return str(manifest.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def _available_audio_files(search_roots: tuple[str, ...], cache_key: str) -> list[str]:
    files: list[Path] = []
    for raw_root in search_roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        for suffix in SUPPORTED_AUDIO_EXTENSIONS:
            files.extend(sorted(root.rglob(f"*{suffix}")))
        if len(files) >= 300:
            break
    return [str(path) for path in files[:300]]


def _display_audio_label(path: str) -> str:
    audio_path = Path(path)
    for root in AUDIO_SEARCH_ROOTS:
        try:
            return str(audio_path.relative_to(root.parent))
        except ValueError:
            continue
    return audio_path.name


def _selected_audio_path(uploaded_file, selected_file: str | None, run_clicked: bool) -> Path | None:
    if uploaded_file is not None:
        upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("uploaded_file_key") != upload_key:
            st.session_state["uploaded_file_key"] = upload_key
            st.session_state.pop("active_audio_path", None)
            st.session_state["active_audio_name"] = uploaded_file.name
        if run_clicked:
            path = _save_uploaded_file(uploaded_file)
            st.session_state["active_audio_path"] = str(path)
            st.session_state["active_audio_name"] = uploaded_file.name
        return Path(st.session_state.get("active_audio_path", "")) if st.session_state.get("active_audio_path") else None
    st.session_state.pop("uploaded_file_key", None)
    if selected_file:
        st.session_state["active_audio_path"] = selected_file
        st.session_state["active_audio_name"] = Path(selected_file).name
        return Path(selected_file)
    return None


def _file_duration_seconds(result: dict) -> float:
    waveform = result["waveform"]
    sample_rate = int(result["sample_rate"])
    return float(len(waveform) / sample_rate) if sample_rate > 0 else 0.0


def _diagnosis(events_df: pd.DataFrame, result: dict) -> dict:
    duration_s = _file_duration_seconds(result)
    if events_df.empty:
        return {
            "status": "No artifact event detected",
            "dominant_artifact": "Normal / no confident artifact",
            "event_count": 0,
            "affected_duration_s": 0.0,
            "coverage_pct": 0.0,
            "highest_confidence": None,
            "file_duration_s": duration_s,
        }

    duration_by_class = events_df.groupby("predicted_class")["duration_ms"].sum().sort_values(ascending=False)
    dominant_artifact = str(duration_by_class.index[0])
    affected_duration_s = float(events_df["duration_ms"].sum() / 1000.0)
    coverage_pct = 100.0 * affected_duration_s / duration_s if duration_s > 0 else 0.0
    return {
        "status": "Artifact detected",
        "dominant_artifact": display_class_name(dominant_artifact),
        "event_count": int(len(events_df)),
        "affected_duration_s": affected_duration_s,
        "coverage_pct": coverage_pct,
        "highest_confidence": float(events_df["peak_confidence"].max()),
        "file_duration_s": duration_s,
    }


def _format_event_table(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    table = events_df.copy()
    table["predicted_class"] = table["predicted_class"].map(display_class_name)
    table["start_s"] = table["start_s"].map(lambda value: round(float(value), 3))
    table["end_s"] = table["end_s"].map(lambda value: round(float(value), 3))
    table["duration_ms"] = table["duration_ms"].map(lambda value: round(float(value), 1))
    table["peak_confidence"] = table["peak_confidence"].map(lambda value: round(float(value), 3))
    table["mean_confidence"] = table["mean_confidence"].map(lambda value: round(float(value), 3))
    return table


def _run_wandb_logging(
    project: str,
    filename: str,
    diagnosis: dict,
    events_df: pd.DataFrame,
    waveform_fig,
    confidence_fig,
) -> None:
    try:
        import wandb
    except ImportError:
        st.warning("WandB is not installed. Install it with `pip install wandb` to enable logging.")
        return

    try:
        run = wandb.init(project=project, job_type="streamlit-demo", reinit=True)
        wandb.log(
            {
                "filename": filename,
                "status": diagnosis["status"],
                "dominant_artifact": diagnosis["dominant_artifact"],
                "number_of_events": diagnosis["event_count"],
                "total_affected_duration_s": diagnosis["affected_duration_s"],
                "artifact_coverage_percentage": diagnosis["coverage_pct"],
                "highest_confidence": diagnosis["highest_confidence"] or 0.0,
                "event_table": wandb.Table(dataframe=events_df),
                "waveform_plot_html": wandb.Html(waveform_fig.to_html(include_plotlyjs="cdn")),
                "confidence_plot_html": wandb.Html(confidence_fig.to_html(include_plotlyjs="cdn")),
            }
        )
        run.finish()
        st.success("Logged this inference result to WandB.")
    except Exception as exc:
        st.warning(f"WandB logging failed: {exc}")


def _metric_value(value, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return str(value)


_inject_style()

st.title("RTX Audio Artifact Inference Demo")
st.markdown(
    '<div class="rtx-subtitle">Frame-level 10 ms artifact diagnosis converted into event-level detections</div>',
    unsafe_allow_html=True,
)
st.info(
    "This model performs frame-level prediction every 10 ms. Since artifacts are sparse, the file is not classified "
    "by majority vote. Instead, artifact frames are grouped into detected events with location, duration, class, "
    "and confidence. Raw predictions are also aggregated into operational targets: packet loss PLC, packet loss "
    "no PLC, repeated packet, and encryption noise."
)

with st.sidebar:
    st.header("Demo controls")
    uploaded = st.file_uploader("Upload audio file", type=[ext.lstrip(".") for ext in SUPPORTED_AUDIO_EXTENSIONS])

    audio_files = _available_audio_files(tuple(str(root) for root in AUDIO_SEARCH_ROOTS), _demo_clip_cache_key())
    label_to_path = {_display_audio_label(path): path for path in audio_files}
    selected_label = st.selectbox(
        "Or select audio file",
        options=[""] + list(label_to_path.keys()),
        format_func=lambda value: "Select a bundled audio file" if value == "" else value,
        disabled=uploaded is not None,
    )
    selected_file = label_to_path.get(selected_label)

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.99,
        value=float(DEFAULT_CONFIDENCE_THRESHOLD),
        step=0.01,
    )
    min_duration_ms = int(DEFAULT_MIN_DURATION_MS)
    merge_gap_ms = int(DEFAULT_MERGE_GAP_MS)
    enable_wandb = st.toggle("Enable WandB logging", value=False)
    wandb_project = ""
    if enable_wandb:
        wandb_project = st.text_input("WandB project", value=os.environ.get("WANDB_PROJECT", "rtx-audio-artifact-demo"))

    run_clicked = st.button("Run inference", type="primary", use_container_width=True)

    st.divider()
    if is_model_artifact_dir(DEFAULT_MODEL_DIR):
        st.success("Model artifact ready")
    else:
        st.error("Model artifact missing")
    st.caption(str(DEFAULT_MODEL_DIR))

audio_path = _selected_audio_path(uploaded, selected_file, run_clicked)

if run_clicked:
    if audio_path is None or not audio_path.exists():
        st.warning("Upload an audio file or select one from the sidebar before running inference.")
    else:
        try:
            with st.spinner("Running frame-level inference and event aggregation..."):
                raw_result = run_inference(str(audio_path))
                result = aggregate_result_to_operational(raw_result)
                events_df = detect_events(
                    result["frame_times"],
                    result["predicted_classes"],
                    result["probabilities"],
                    result["class_names"],
                    normal_class=NORMAL_CLASS,
                    confidence_threshold=confidence_threshold,
                    min_duration_ms=min_duration_ms,
                    merge_gap_ms=merge_gap_ms,
                )
                diagnosis = _diagnosis(events_df, result)
                waveform_fig = plot_waveform_with_events(result["waveform"], result["sample_rate"], events_df)
                confidence_fig = plot_confidence_over_time(
                    result["frame_times"], result["probabilities"], result["class_names"]
                )
                distribution_fig = plot_class_distribution(result["predicted_classes"])
            st.session_state["last_run"] = {
                "audio_path": str(audio_path),
                "audio_name": st.session_state.get("active_audio_name", audio_path.name),
                "result": result,
                "events_df": events_df,
                "diagnosis": diagnosis,
                "waveform_fig": waveform_fig,
                "confidence_fig": confidence_fig,
                "distribution_fig": distribution_fig,
            }
        except Exception as exc:
            st.error("Inference failed. Check that the model artifacts and audio file are readable.")
            with st.expander("Error details"):
                st.code(str(exc))

last_run = st.session_state.get("last_run")

top_left, top_right = st.columns([1.1, 2.4], gap="large")
with top_left:
    st.subheader("Audio")
    if uploaded is not None and audio_path is None:
        st.audio(uploaded.getvalue(), format=_audio_mime(uploaded.name))
    elif audio_path is not None and audio_path.exists():
        st.audio(audio_path.read_bytes(), format=_audio_mime(audio_path))
        st.markdown(f'<div class="small-muted">{Path(audio_path).name}</div>', unsafe_allow_html=True)
    else:
        st.write("Choose an audio file in the sidebar.")

with top_right:
    st.subheader("Diagnosis")
    if last_run is None:
        st.write("Run inference to populate diagnosis metrics.")
    else:
        diagnosis = last_run["diagnosis"]
        row1 = st.columns(3)
        row1[0].metric("File status", diagnosis["status"])
        row1[1].metric("Dominant artifact", diagnosis["dominant_artifact"])
        row1[2].metric("Detected events", diagnosis["event_count"])
        row2 = st.columns(3)
        row2[0].metric("Total affected duration", f"{diagnosis['affected_duration_s']:.3f} s")
        row2[1].metric("Artifact coverage", f"{diagnosis['coverage_pct']:.2f}%")
        row2[2].metric(
            "Highest confidence",
            "N/A" if diagnosis["highest_confidence"] is None else f"{diagnosis['highest_confidence']:.1%}",
        )

if last_run is not None:
    st.subheader("Waveform")
    st.plotly_chart(last_run["waveform_fig"], use_container_width=True)

    st.subheader("Frame-Level Confidence")
    st.plotly_chart(last_run["confidence_fig"], use_container_width=True)

    table_col, dist_col = st.columns([1.6, 1], gap="large")
    with table_col:
        st.subheader("Detected Events")
        events_df = last_run["events_df"]
        if events_df.empty:
            st.info("No confident artifact events passed the current threshold and duration settings.")
        st.dataframe(
            _format_event_table(events_df),
            use_container_width=True,
            hide_index=True,
            column_config={
                "event_id": st.column_config.NumberColumn("Event"),
                "start_s": st.column_config.NumberColumn("Start (s)", format="%.3f"),
                "end_s": st.column_config.NumberColumn("End (s)", format="%.3f"),
                "duration_ms": st.column_config.NumberColumn("Duration (ms)", format="%.1f"),
                "predicted_class": st.column_config.TextColumn("Predicted class"),
                "frame_count": st.column_config.NumberColumn("Frames"),
                "peak_confidence": st.column_config.NumberColumn("Peak confidence", format="%.3f"),
                "mean_confidence": st.column_config.NumberColumn("Mean confidence", format="%.3f"),
            },
        )
    with dist_col:
        st.subheader("Frame Counts")
        st.plotly_chart(last_run["distribution_fig"], use_container_width=True)

    if enable_wandb:
        if st.button("Log result to WandB", use_container_width=True):
            _run_wandb_logging(
                wandb_project,
                last_run["audio_name"],
                last_run["diagnosis"],
                last_run["events_df"],
                last_run["waveform_fig"],
                last_run["confidence_fig"],
            )
else:
    st.subheader("Waveform")
    st.write("Detected artifact regions will appear here after inference.")
