from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from config import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_MODEL_DIR, MODEL_SRC_DIR


try:
    import streamlit as st

    cache_resource = st.cache_resource(show_spinner=False)
except Exception:
    from functools import lru_cache

    def cache_resource(func):
        return lru_cache(maxsize=1)(func)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    config: dict
    scalers: dict[str, np.ndarray]
    class_names: tuple[str, ...]
    device: torch.device
    batch_size: int


def _ensure_model_src_on_path(model_src_dir: Path = MODEL_SRC_DIR) -> None:
    if not model_src_dir.exists():
        raise FileNotFoundError(
            f"Could not find original model source directory: {model_src_dir}. "
            "Set RTX_MODEL_SRC or update config.py."
        )
    model_src = str(model_src_dir)
    if model_src not in sys.path:
        sys.path.insert(0, model_src)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("RTX_DEVICE is set to cuda, but CUDA is not available in this Python environment.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_model_dir(model_dir: Path) -> None:
    missing = []
    if not (model_dir / "config.json").exists():
        missing.append("config.json")
    if not (model_dir / "scalers.npz").exists():
        missing.append("scalers.npz")
    if not (model_dir / "model.pt").exists() and not (model_dir / "model_torchscript.pt").exists():
        missing.append("model.pt or model_torchscript.pt")
    if missing:
        raise FileNotFoundError(f"Model artifact directory {model_dir} is missing: {', '.join(missing)}")


@cache_resource
def load_model_bundle(
    model_dir: str = str(DEFAULT_MODEL_DIR),
    model_src_dir: str = str(MODEL_SRC_DIR),
    device_arg: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ModelBundle:
    _ensure_model_src_on_path(Path(model_src_dir))

    from features import CLASS_NAMES, load_scalers
    from model import SmallGRUConfig, load_model

    artifact_dir = Path(model_dir)
    _validate_model_dir(artifact_dir)

    config = _load_json(artifact_dir / "config.json")
    device = _resolve_device(str(device_arg))

    torchscript_path = artifact_dir / "model_torchscript.pt"
    if torchscript_path.exists():
        model = torch.jit.load(str(torchscript_path), map_location=device)
        model.to(device)
        model.eval()
    else:
        model_config = SmallGRUConfig(
            gru_hidden_dim=int(config.get("gru_hidden_dim", 48)),
            embed_dim=int(config.get("embed_dim", 32)),
            fusion_hidden_dim=int(config.get("fusion_hidden_dim", 64)),
        )
        model = load_model(artifact_dir / "model.pt", device, model_config)

    scalers = load_scalers(artifact_dir / "scalers.npz")
    class_names = tuple(config.get("class_names") or CLASS_NAMES)
    return ModelBundle(
        model=model,
        config=config,
        scalers=scalers,
        class_names=class_names,
        device=device,
        batch_size=int(batch_size),
    )


def _load_audio_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception as soundfile_error:
        try:
            import librosa
        except Exception as import_error:
            raise RuntimeError(
                f"Could not read {audio_path.name} with soundfile. Install librosa for MP3 fallback support."
            ) from import_error
        try:
            audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
        except Exception as librosa_error:
            raise RuntimeError(f"Could not read audio file {audio_path}: {librosa_error}") from soundfile_error

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return audio.astype(np.float32), int(sample_rate)


def _audio_to_sequences(
    audio: np.ndarray,
    input_sample_rate: int,
    target_sample_rate: int,
    frame_ms: float,
    sequence_length: int,
) -> tuple[np.ndarray, dict]:
    _ensure_model_src_on_path()

    from features import build_causal_sequences, compute_feature_matrix, resample_audio

    model_audio = resample_audio(audio, input_sample_rate, target_sample_rate)
    features, frame_samples = compute_feature_matrix(model_audio, target_sample_rate, frame_ms)
    metadata = {
        "input_sample_rate": int(input_sample_rate),
        "feature_sample_rate": int(target_sample_rate),
        "frame_samples": int(frame_samples),
        "frame_ms": 1000.0 * frame_samples / target_sample_rate,
        "total_frames": int(features.shape[0]),
        "duration_seconds": features.shape[0] * frame_samples / target_sample_rate,
    }
    return build_causal_sequences(features, sequence_length), metadata


def _predict_probabilities(bundle: ModelBundle, temporal: np.ndarray, spectral: np.ndarray) -> np.ndarray:
    probabilities = []
    with torch.no_grad():
        for start in range(0, temporal.shape[0], bundle.batch_size):
            end = start + bundle.batch_size
            x_temporal = torch.from_numpy(temporal[start:end]).to(bundle.device)
            x_spectral = torch.from_numpy(spectral[start:end]).to(bundle.device)
            logits = bundle.model(x_temporal, x_spectral)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(np.float32)


def run_inference(audio_path: str) -> dict:
    """
    Returns:
    {
        "sample_rate": int,
        "waveform": np.ndarray,
        "frame_times": np.ndarray,        # one timestamp per 10 ms frame
        "predicted_classes": list[str],   # one class per frame
        "probabilities": np.ndarray,      # shape [num_frames, num_classes]
        "class_names": list[str]
    }
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(path)

    bundle = load_model_bundle()
    waveform, sample_rate = _load_audio_mono(path)

    target_sample_rate = int(bundle.config.get("sample_rate", 16000))
    frame_ms = float(bundle.config.get("frame_ms", 10.0))
    sequence_length = int(bundle.config.get("sequence_length", 11))

    _ensure_model_src_on_path()
    from features import transform_for_model

    sequences, metadata = _audio_to_sequences(
        waveform,
        sample_rate,
        target_sample_rate=target_sample_rate,
        frame_ms=frame_ms,
        sequence_length=sequence_length,
    )
    temporal, spectral = transform_for_model(sequences, bundle.scalers)
    probabilities = _predict_probabilities(bundle, temporal, spectral)

    prediction_ids = np.argmax(probabilities, axis=1)
    class_names = list(bundle.class_names)
    predicted_classes = [class_names[int(class_id)] for class_id in prediction_ids]
    frame_times = np.arange(len(predicted_classes), dtype=np.float32) * (metadata["frame_ms"] / 1000.0)

    return {
        "sample_rate": int(sample_rate),
        "waveform": waveform,
        "frame_times": frame_times,
        "predicted_classes": predicted_classes,
        "probabilities": probabilities,
        "class_names": class_names,
    }
