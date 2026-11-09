"""Turn model predictions and windowed graphs into structured alert evidence.

Inputs are the artefacts the team already produces:

* ``predictions.npz`` (schema in ``src.comparison.gnn_importer``)
* ``{split}_graph_{id:05d}.npz`` windows from ``build_*_graphs.py``
* ``scaler.joblib`` and ``feature_names.joblib`` from preprocessing

Ground-truth labels are used for two things only: checking that predictions
line up with the graph files, and sorting alerts for human review. They are
never written into the evidence an LLM sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Columns that preprocessing passes through log1p before standardizing.
# Identical in preprocess_features.py and preprocess_cicids.py.
LOG_COLUMNS = frozenset(
    {
        "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES", "DNS_TTL_ANSWER",
    }
)

REQUIRED_FEATURES = (
    "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS",
)

PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}
TCP_FLAG_BITS = ((1, "FIN"), (2, "SYN"), (4, "RST"), (8, "PSH"), (16, "ACK"), (32, "URG"))

OUTCOME_CORRECT = "correct_class"
OUTCOME_WRONG_CLASS = "wrong_attack_class"
OUTCOME_FALSE_POSITIVE = "false_positive"


@dataclass(frozen=True)
class Predictions:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_binary: np.ndarray
    proba: np.ndarray | None
    graph_id: np.ndarray
    edge_id: np.ndarray
    class_names: list[str]
    benign_id: int
    dataset: str
    split: str


def load_predictions(path: Path) -> Predictions:
    with np.load(path, allow_pickle=False) as npz:
        try:
            proba = npz["y_proba_multiclass"] if "y_proba_multiclass" in npz.files else None
        except ValueError:  # save_predictions_npz stores a missing array as a pickled None
            proba = None
        class_names = [str(c) for c in npz["class_names"]]
        if proba is not None and proba.shape != (len(npz["y_pred_multiclass"]), len(class_names)):
            raise ValueError(f"y_proba_multiclass has unexpected shape {proba.shape}")
        return Predictions(
            y_true=npz["y_true_multiclass"],
            y_pred=npz["y_pred_multiclass"],
            y_pred_binary=npz["y_pred_binary"],
            proba=proba,
            graph_id=npz["graph_id"],
            edge_id=npz["edge_id"],
            class_names=class_names,
            benign_id=int(npz["benign_class_id"]),
            dataset=str(npz["dataset"]),
            split=str(npz["split"]),
        )


def flagged_outcome(y_true: int, y_pred: int, benign_id: int) -> str:
    if y_true == benign_id:
        return OUTCOME_FALSE_POSITIVE
    return OUTCOME_CORRECT if y_true == y_pred else OUTCOME_WRONG_CLASS


def select_alerts(pred: Predictions, per_cell: int, seed: int = 42) -> np.ndarray:
    """Sample flagged flows evenly across (predicted class, outcome) cells.

    Rare classes and mistakes would almost never appear in a random sample, and
    they are where a grounded explanation is hardest to get right.
    """
    rng = np.random.default_rng(seed)
    flagged = np.flatnonzero(pred.y_pred_binary == 1)
    outcomes = np.array(
        [flagged_outcome(int(pred.y_true[i]), int(pred.y_pred[i]), pred.benign_id) for i in flagged]
    )
    chosen: list[np.ndarray] = []
    for cls in np.unique(pred.y_pred[flagged]):
        for outcome in (OUTCOME_CORRECT, OUTCOME_WRONG_CLASS, OUTCOME_FALSE_POSITIVE):
            cell = flagged[(pred.y_pred[flagged] == cls) & (outcomes == outcome)]
            if cell.size:
                chosen.append(rng.choice(cell, size=min(per_cell, cell.size), replace=False))
    return np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)


class FlowDecoder:
    """Undo standardization and log1p so alerts can quote real values."""

    def __init__(self, scaler_path: Path, feature_names_path: Path) -> None:
        scaler = joblib.load(scaler_path)
        self.names: list[str] = list(joblib.load(feature_names_path))
        missing = set(REQUIRED_FEATURES) - set(self.names)
        if missing:
            raise ValueError(f"feature_names is missing {sorted(missing)}")
        self.mean = np.asarray(scaler.mean_, dtype=np.float64)
        self.scale = np.asarray(scaler.scale_, dtype=np.float64)
        if self.mean.shape[0] != len(self.names):
            raise ValueError("scaler and feature_names have different lengths")
        self.index = {name: i for i, name in enumerate(self.names)}

    def column(self, edge_attr: np.ndarray, name: str) -> np.ndarray:
        i = self.index[name]
        values = edge_attr[..., i].astype(np.float64) * self.scale[i] + self.mean[i]
        if name in LOG_COLUMNS:
            values = np.expm1(values)
        return np.rint(values).astype(np.int64)


def graph_files(graph_dir: Path, split: str) -> list[Path]:
    """Same ordering as DataLoader, whose enumerate() index is ``graph_id``."""
    files = sorted(Path(graph_dir).glob(f"{split}_graph_*.npz"))
    if not files:
        raise FileNotFoundError(f"No {split}_graph_*.npz files in {graph_dir}")
    return files


def _tcp_flags(value: int) -> list[str]:
    return [name for bit, name in TCP_FLAG_BITS if value & bit]


def _flow_evidence(graph: dict[str, np.ndarray], edge: int, decoder: FlowDecoder) -> dict[str, Any]:
    row = graph["edge_attr"][edge]
    value = lambda name: int(decoder.column(row, name))
    src, dst = (int(x) for x in graph["edge_index"][:, edge])
    protocol = value("PROTOCOL")
    return {
        "source_host": f"host_{src}",
        "destination_host": f"host_{dst}",
        "destination_port": value("L4_DST_PORT"),
        "protocol": PROTOCOL_NAMES.get(protocol, f"protocol_{protocol}"),
        "bytes_source_to_destination": value("IN_BYTES"),
        "bytes_destination_to_source": value("OUT_BYTES"),
        "packets_source_to_destination": value("IN_PKTS"),
        "packets_destination_to_source": value("OUT_PKTS"),
        "duration_ms": value("FLOW_DURATION_MILLISECONDS"),
        "tcp_flags_seen": _tcp_flags(value("TCP_FLAGS")),
    }


def _window_context(
    graph: dict[str, np.ndarray],
    edge: int,
    dst_ports: np.ndarray,
    flagged_in_window: np.ndarray,
) -> dict[str, int]:
    sources, destinations = graph["edge_index"]
    from_src = sources == sources[edge]
    to_dst = destinations == destinations[edge]
    same_port = dst_ports == dst_ports[edge]
    return {
        "flows_in_window": int(sources.size),
        "source_outgoing_flows": int(from_src.sum()),
        "source_distinct_destination_hosts": int(np.unique(destinations[from_src]).size),
        "source_distinct_destination_ports": int(np.unique(dst_ports[from_src]).size),
        "destination_incoming_flows": int(to_dst.sum()),
        "destination_distinct_source_hosts": int(np.unique(sources[to_dst]).size),
        "flows_same_source_and_destination_port": int((from_src & same_port).sum()),
        "source_flows_flagged_malicious": int((from_src & flagged_in_window).sum()),
        "flows_flagged_malicious_in_window": int(flagged_in_window.sum()),
    }


def _model_output(pred: Predictions, i: int) -> dict[str, Any]:
    predicted = int(pred.y_pred[i])
    output: dict[str, Any] = {"predicted_class": pred.class_names[predicted]}
    if pred.proba is None:
        output["confidence_percent"] = None
        return output
    probs = pred.proba[i]
    output["confidence_percent"] = int(round(float(probs[predicted]) * 100))
    others = probs.copy()
    others[predicted] = -1
    runner_up = int(np.argmax(others))
    output["runner_up_class"] = pred.class_names[runner_up]
    output["runner_up_percent"] = int(round(float(probs[runner_up]) * 100))
    return output


def build_evidence(
    pred: Predictions,
    indices: np.ndarray,
    graph_dir: Path,
    decoder: FlowDecoder,
) -> list[dict[str, Any]]:
    """Return one evidence record per prediction index.

    Raises ``ValueError`` if a prediction does not match its graph file, which
    catches a different ``graph_id`` ordering in an imported model's export.
    """
    files = graph_files(graph_dir, pred.split)
    records: dict[int, dict[str, Any]] = {}

    for g in np.unique(pred.graph_id[indices]):
        g = int(g)
        if not 0 <= g < len(files):
            raise ValueError(f"graph_id {g} is outside the {len(files)} {pred.split} graph files")
        with np.load(files[g], allow_pickle=False) as npz:
            graph = {k: npz[k] for k in ("edge_index", "edge_attr", "edge_attack")}

        members = np.flatnonzero(pred.graph_id == g)
        expected = graph["edge_attack"][pred.edge_id[members]]
        if not np.array_equal(expected, pred.y_true[members]):
            raise ValueError(
                f"Labels in predictions.npz disagree with {files[g].name}: "
                "graph_id or edge_id does not follow the graph-file order."
            )

        flagged_in_window = np.zeros(graph["edge_index"].shape[1], dtype=bool)
        flagged_in_window[pred.edge_id[members[pred.y_pred_binary[members] == 1]]] = True
        dst_ports = decoder.column(graph["edge_attr"], "L4_DST_PORT")

        for i in indices[pred.graph_id[indices] == g]:
            i = int(i)
            edge = int(pred.edge_id[i])
            records[i] = {
                "alert_id": f"{pred.dataset}-{pred.split}-g{g:05d}-e{edge:05d}".lower(),
                "model_output": _model_output(pred, i),
                "flow": _flow_evidence(graph, edge, decoder),
                "window_context": _window_context(graph, edge, dst_ports, flagged_in_window),
            }

    return [records[int(i)] for i in indices]
