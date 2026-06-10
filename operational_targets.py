from __future__ import annotations

import numpy as np


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


def aggregate_result_to_operational(result: dict) -> dict:
    raw_class_names = list(result["class_names"])
    raw_probabilities = np.asarray(result["probabilities"], dtype=np.float32)

    operational_class_names = list(OPERATIONAL_CLASS_NAMES)
    operational_index = {name: index for index, name in enumerate(operational_class_names)}
    operational_probabilities = np.zeros(
        (raw_probabilities.shape[0], len(operational_class_names)),
        dtype=np.float32,
    )

    for raw_index, raw_class in enumerate(raw_class_names):
        operational_class = OPERATIONAL_CLASS_BY_MODEL_CLASS.get(raw_class, raw_class)
        if operational_class not in operational_index:
            continue
        operational_probabilities[:, operational_index[operational_class]] += raw_probabilities[:, raw_index]

    prediction_ids = np.argmax(operational_probabilities, axis=1)
    operational_predicted_classes = [
        operational_class_names[int(class_id)] for class_id in prediction_ids
    ]

    operational_result = dict(result)
    operational_result["raw_class_names"] = raw_class_names
    operational_result["raw_predicted_classes"] = list(result["predicted_classes"])
    operational_result["raw_probabilities"] = raw_probabilities
    operational_result["class_names"] = operational_class_names
    operational_result["predicted_classes"] = operational_predicted_classes
    operational_result["probabilities"] = operational_probabilities
    return operational_result
