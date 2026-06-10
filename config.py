from __future__ import annotations

import os
import tempfile
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent

DEFAULT_REPO_ROOT = Path(
    os.environ.get(
        "RTX_REPO_ROOT",
        r"C:\Users\tobia\Desktop\P4\p4_produkt\p4_rtx",
    )
)
MODEL_SRC_DIR = Path(os.environ.get("RTX_MODEL_SRC", str(DEFAULT_REPO_ROOT / "model" / "src")))
DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "RTX_MODEL_DIR",
        str(DEFAULT_REPO_ROOT / "runs" / "small_fusion_gru_full_simulated"),
    )
)

DEFAULT_DEVICE = os.environ.get("RTX_DEVICE", "auto")
DEFAULT_BATCH_SIZE = int(os.environ.get("RTX_BATCH_SIZE", "2048"))

NORMAL_CLASS = "normal"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MIN_DURATION_MS = 10
DEFAULT_MERGE_GAP_MS = 20

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")
TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "rtx_audio_artifact_demo_uploads"

AUDIO_SEARCH_ROOTS = (
    APP_DIR / "demo_clips",
    DEFAULT_REPO_ROOT / "data" / "external" / "RTX_test_data",
    DEFAULT_REPO_ROOT / "data" / "simulated_artifacts" / "wav",
    DEFAULT_REPO_ROOT / "data" / "original",
)

DISPLAY_CLASS_NAMES = {
    "normal": "Normal",
    "packet_loss_plc": "Packet loss PLC",
    "packet_loss_no_plc": "Packet loss no PLC",
    "dropout_plc": "Dropout PLC",
    "burst_plc": "Burst PLC",
    "dropout_no_plc": "Dropout no PLC",
    "burst_no_plc": "Burst no PLC",
    "repeated_packet": "Repeated packet",
    "encryption_noise": "Encryption noise",
    "corrupted_packet": "Corrupted packet",
}

OPERATIONAL_CLASS_NAMES = [
    "normal",
    "packet_loss_plc",
    "packet_loss_no_plc",
    "repeated_packet",
    "encryption_noise",
]

OPERATIONAL_CLASS_BY_MODEL_CLASS = {
    "normal": "normal",
    "dropout_plc": "packet_loss_plc",
    "burst_plc": "packet_loss_plc",
    "dropout_no_plc": "packet_loss_no_plc",
    "burst_no_plc": "packet_loss_no_plc",
    "repeated_packet": "repeated_packet",
    "encryption_noise": "encryption_noise",
    "corrupted_packet": "encryption_noise",
}


def display_class_name(class_name: str) -> str:
    return DISPLAY_CLASS_NAMES.get(str(class_name), str(class_name).replace("_", " ").title())


def is_model_artifact_dir(path: Path) -> bool:
    has_weights = (path / "model.pt").exists() or (path / "model_torchscript.pt").exists()
    return has_weights and (path / "scalers.npz").exists() and (path / "config.json").exists()
