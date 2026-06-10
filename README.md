# RTX Audio Artifact Inference Demo

Streamlit demo for the trained frame-level audio artifact model. The app runs one prediction per 10 ms frame, aggregates raw model classes into higher-level operational targets, groups confident non-normal frames into artifact events, and displays the file diagnosis with interactive Plotly views.

## Run

```powershell
cd C:\Users\tobia\Desktop\P4\live_demo
pip install -r requirements.txt
streamlit run app.py
```

## Model weights

The demo currently points at the trained local artifact found here:

```text
C:\Users\tobia\Desktop\P4\p4_produkt\p4_rtx\runs\small_fusion_gru_full_simulated
```

That folder contains the required files:

- `model.pt`
- `scalers.npz`
- `config.json`
- `metrics.json`

To use another trained model, either edit `DEFAULT_MODEL_DIR` in `config.py` or set:

```powershell
$env:RTX_MODEL_DIR="C:\path\to\model_artifact_folder"
```

The folder must contain `config.json`, `scalers.npz`, and either `model.pt` or `model_torchscript.pt`.

## Connecting existing inference code

The rest of the app calls only:

```python
from inference import run_inference
```

`run_inference(audio_path: str)` returns:

```python
{
    "sample_rate": int,
    "waveform": np.ndarray,
    "frame_times": np.ndarray,
    "predicted_classes": list[str],
    "probabilities": np.ndarray,
    "class_names": list[str],
}
```

This adapter currently imports the existing GRU feature extraction and model code from:

```text
C:\Users\tobia\Desktop\P4\p4_produkt\p4_rtx\model\src
```

If your production inference entry point changes, keep the same `run_inference` return contract and replace the inside of `inference.py`.

## Event-based aggregation

The model predicts one class every 10 ms. The app does not use whole-file majority voting, because sparse artifacts would usually be hidden by many normal frames.

Before event detection, `operational_targets.py` maps raw model classes into operational targets:

- `dropout_plc` + `burst_plc` -> `packet_loss_plc`
- `dropout_no_plc` + `burst_no_plc` -> `packet_loss_no_plc`
- `repeated_packet` -> `repeated_packet`
- `encryption_noise` + `corrupted_packet` -> `encryption_noise`
- `normal` -> `normal`

Probabilities are summed within each operational group, and the highest grouped probability becomes the displayed frame prediction.

Then `event_detection.py`:

- ignores frames predicted as Normal,
- keeps only artifact frames above the confidence threshold,
- groups consecutive frames of the same artifact class,
- merges same-class events separated by a short configurable gap,
- removes events shorter than the minimum event duration,
- returns event start, end, duration, class, frame count, peak confidence, and mean confidence.

The file-level diagnosis is then based on detected events. If events exist, the dominant artifact is the class with the highest total event duration.

## WandB

WandB is optional. The app initializes WandB only when you enable the sidebar toggle and click `Log result to WandB`. It logs summary metrics, the event table, and Plotly figures as HTML.
