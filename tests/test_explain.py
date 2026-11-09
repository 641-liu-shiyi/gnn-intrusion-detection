"""Tests for the explanation pipeline, using files in the team's real formats."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.explain.arc_client import ARCClient, LLMError
from src.explain.evidence import (
    OUTCOME_CORRECT,
    OUTCOME_FALSE_POSITIVE,
    OUTCOME_WRONG_CLASS,
    FlowDecoder,
    build_evidence,
    flagged_outcome,
    load_predictions,
    select_alerts,
)
from src.explain.grounding import check_alert, summarize_review, write_review_sheet
from src.explain.prompting import SYSTEM_PROMPT, build_messages

REPO = Path(__file__).resolve().parents[1]
NAMES = ["L4_DST_PORT", "PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
         "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS"]
LOG = {"IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"}
CLASSES = ["Analysis", "Backdoor", "Benign", "DoS", "Exploits"]
BENIGN = 2


def transform(raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    x = raw.astype(np.float64).copy()
    for j, name in enumerate(NAMES):
        if name in LOG:
            x[:, j] = np.log1p(x[:, j])
    return scaler.transform(x).astype(np.float32)


@pytest.fixture()
def bundle(tmp_path: Path) -> dict:
    """One 8-flow window plus a second window, with hand-countable structure.

    Host 0 scans hosts 1, 2, 3 on ports 22, 80, 80 and also sends two flows to host 4.
    Hosts 5 and 6 send to host 4. Flows 0-2 and 4 are predicted malicious.
    """
    edge_index = np.array([[0, 0, 0, 0, 0, 5, 6, 5], [1, 2, 3, 4, 4, 4, 4, 1]], dtype=np.int64)
    #               port proto in_bytes out_bytes in_pkts out_pkts dur_ms flags
    raw = np.array([
        [22, 6, 60, 0, 1, 0, 0, 2],
        [80, 6, 1500000, 40, 1000, 1, 12345, 18],
        [80, 6, 60, 0, 1, 0, 0, 2],
        [443, 17, 100, 200, 2, 3, 9, 0],
        [443, 6, 70, 0, 1, 0, 1, 4],
        [443, 6, 800, 900, 8, 9, 50, 27],
        [8080, 1, 64, 64, 1, 1, 2, 0],
        [80, 6, 500, 600, 5, 6, 70, 24],
    ], dtype=np.float64)
    fit = raw.copy()
    for j, name in enumerate(NAMES):
        if name in LOG:
            fit[:, j] = np.log1p(fit[:, j])
    scaler = StandardScaler().fit(fit)
    attr = transform(raw, scaler)
    #                 flow:  0  1  2  3  4  5  6  7
    edge_attack = np.array([3, 3, 2, 2, 4, 2, 2, 2], dtype=np.int64)
    y_pred = np.array([3, 4, 3, 2, 4, 2, 2, 2], dtype=np.int64)

    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for g in range(2):
        np.savez_compressed(
            graphs / f"test_graph_{g:05d}.npz",
            edge_index=edge_index, edge_attr=attr,
            edge_label=(edge_attack != BENIGN).astype(np.int64),
            edge_attack=edge_attack, num_nodes=7,
        )
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(NAMES, tmp_path / "feature_names.joblib")

    proba = np.full((16, 5), 0.05)
    y_pred_all = np.concatenate([y_pred, y_pred])
    proba[np.arange(16), y_pred_all] = 0.80
    proba[0] = [0.1, 0.2, 0.05, 0.55, 0.1]

    arrays = dict(
        y_true_multiclass=np.concatenate([edge_attack, edge_attack]),
        y_pred_multiclass=y_pred_all,
        y_true_binary=np.concatenate([edge_attack != BENIGN] * 2).astype(np.int64),
        y_pred_binary=(y_pred_all != BENIGN).astype(np.int64),
        y_proba_multiclass=proba,
        graph_id=np.array([0] * 8 + [1] * 8, dtype=np.int64),
        edge_id=np.concatenate([np.arange(8), np.arange(8)]).astype(np.int64),
        class_names=np.array(CLASSES, dtype=str),
        benign_class_id=np.int64(BENIGN),
        dataset="UNSW",
        split="test",
    )
    np.savez_compressed(tmp_path / "predictions.npz", **arrays)
    return {"root": tmp_path, "graphs": graphs, "arrays": arrays}


def decoder(root: Path) -> FlowDecoder:
    return FlowDecoder(root / "scaler.joblib", root / "feature_names.joblib")


def evidence_for(bundle: dict, indices: list[int]) -> list[dict]:
    pred = load_predictions(bundle["root"] / "predictions.npz")
    return build_evidence(pred, np.array(indices), bundle["graphs"], decoder(bundle["root"]))


def test_restores_readable_flow_values(bundle):
    flow = evidence_for(bundle, [1])[0]["flow"]
    assert flow["destination_port"] == 80
    assert flow["protocol"] == "TCP"
    assert flow["bytes_source_to_destination"] == pytest.approx(1500000, rel=1e-4)
    assert flow["packets_source_to_destination"] == 1000
    assert flow["duration_ms"] == pytest.approx(12345, rel=1e-4)
    assert flow["tcp_flags_seen"] == ["SYN", "ACK"]
    assert (flow["source_host"], flow["destination_host"]) == ("host_0", "host_2")


def test_window_statistics_match_hand_counts(bundle):
    context = evidence_for(bundle, [1])[0]["window_context"]
    assert context == {
        "flows_in_window": 8,
        "source_outgoing_flows": 5,
        "source_distinct_destination_hosts": 4,
        "source_distinct_destination_ports": 3,
        "destination_incoming_flows": 1,
        "destination_distinct_source_hosts": 1,
        "flows_same_source_and_destination_port": 2,
        "source_flows_flagged_malicious": 4,
        "flows_flagged_malicious_in_window": 4,
    }
    fan_in = evidence_for(bundle, [4])[0]["window_context"]
    assert fan_in["destination_incoming_flows"] == 4
    assert fan_in["destination_distinct_source_hosts"] == 3


def test_model_output_uses_probabilities(bundle):
    output = evidence_for(bundle, [0])[0]["model_output"]
    assert output == {
        "predicted_class": "DoS", "confidence_percent": 55,
        "runner_up_class": "Backdoor", "runner_up_percent": 20,
    }


def test_prompt_carries_no_ground_truth(bundle):
    original = build_messages(evidence_for(bundle, [0])[0])

    arrays = dict(bundle["arrays"])
    relabeled = arrays["y_true_multiclass"].copy()
    relabeled[relabeled == 3] = 4
    arrays["y_true_multiclass"] = relabeled
    np.savez_compressed(bundle["root"] / "predictions.npz", **arrays)
    for g in range(2):
        path = bundle["graphs"] / f"test_graph_{g:05d}.npz"
        with np.load(path) as npz:
            data = {k: npz[k] for k in npz.files}
        data["edge_attack"] = relabeled[:8]
        np.savez_compressed(path, **data)

    assert build_messages(evidence_for(bundle, [0])[0]) == original
    record = json.loads(original[1]["content"])
    flat = json.dumps(record).lower()
    assert "true" not in flat and "label" not in flat
    assert not {"alert_id"} & set(record)


def test_rejects_misaligned_predictions(bundle):
    arrays = dict(bundle["arrays"])
    arrays["edge_id"] = np.concatenate([np.arange(8)[::-1], np.arange(8)]).astype(np.int64)
    np.savez_compressed(bundle["root"] / "predictions.npz", **arrays)
    with pytest.raises(ValueError, match="disagree"):
        evidence_for(bundle, [0])


def test_handles_predictions_saved_without_probabilities(bundle):
    arrays = dict(bundle["arrays"])
    arrays["y_proba_multiclass"] = None
    np.savez_compressed(bundle["root"] / "predictions.npz", **arrays)
    output = evidence_for(bundle, [0])[0]["model_output"]
    assert output == {"predicted_class": "DoS", "confidence_percent": None}


def test_outcomes_and_selection_cover_mistakes(bundle):
    assert flagged_outcome(3, 3, BENIGN) == OUTCOME_CORRECT
    assert flagged_outcome(3, 4, BENIGN) == OUTCOME_WRONG_CLASS
    assert flagged_outcome(BENIGN, 3, BENIGN) == OUTCOME_FALSE_POSITIVE

    pred = load_predictions(bundle["root"] / "predictions.npz")
    chosen = select_alerts(pred, per_cell=1, seed=1)
    outcomes = {flagged_outcome(int(pred.y_true[i]), int(pred.y_pred[i]), BENIGN) for i in chosen}
    assert outcomes == {OUTCOME_CORRECT, OUTCOME_WRONG_CLASS, OUTCOME_FALSE_POSITIVE}
    assert all(pred.y_pred_binary[i] == 1 for i in chosen)
    assert np.array_equal(chosen, select_alerts(pred, per_cell=1, seed=1))


def base_evidence() -> dict:
    return {
        "model_output": {"predicted_class": "DoS", "confidence_percent": 87,
                         "runner_up_class": "Exploits", "runner_up_percent": 6},
        "flow": {"source_host": "host_12", "destination_port": 80, "bytes_source_to_destination": 1500000},
        "window_context": {"source_outgoing_flows": 240},
    }


def test_grounded_alert_passes():
    text = ("Flagged as DoS (87% confidence). The model flagged host_12 sending 1,500,000 bytes "
            "to port 80, and the source opened 240 flows in the window, consistent with the prediction.")
    assert check_alert(text, base_evidence(), CLASSES)["passed"] is True


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("Flagged as DoS (87%). It sent 1.5 million bytes.", "unsupported_numbers", ["1.5"]),
        ("Flagged as DoS (87%). 45% of its flows were flagged.", "unsupported_numbers", ["45"]),
        ("Flagged as DoS, possibly Backdoor activity.", "other_classes_named", ["Backdoor"]),
        ("Flagged as DoS. The attacker used port 80.", "speculative_terms", ["attacker"]),
    ],
)
def test_ungrounded_alerts_are_flagged(text, field, expected):
    result = check_alert(text, base_evidence(), CLASSES)
    assert result[field] == expected
    assert result["passed"] is False


def test_host_labels_are_not_treated_as_numbers():
    text = "Flagged as DoS (87%). host_12 and host_99 appear in the record."
    assert check_alert(text, base_evidence(), CLASSES)["unsupported_numbers"] == []


def test_review_summary_tallies_only_scored_rows(tmp_path):
    path = tmp_path / "review.csv"
    write_review_sheet(path, [
        {"alert_id": "a", "outcome": "correct_class", "manual_factually_consistent": "y", "manual_clear": "y"},
        {"alert_id": "b", "outcome": "correct_class", "manual_factually_consistent": "n", "manual_clear": "y"},
        {"alert_id": "c", "outcome": "false_positive"},
    ])
    result = summarize_review(path)
    assert result["by_outcome"]["correct_class"] == {"reviewed": 2, "factually_consistent": 1, "clear": 2}
    assert result["unreviewed_rows"] == 1


class FakeGateway(BaseHTTPRequestHandler):
    calls = 0
    fail_first = 0
    bodies: list = []

    def do_POST(self):  # noqa: N802
        cls = type(self)
        cls.calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls.bodies.append((self.path, self.headers["Authorization"], body))
        if cls.calls <= cls.fail_first:
            self.send_response(503)
            self.end_headers()
            return
        record = json.loads(body["messages"][1]["content"])
        output = record["model_output"]
        text = (f"Flagged as {output['predicted_class']} ({output['confidence_percent']}% confidence). "
                f"The model flagged {record['flow']['source_host']}, which is consistent with the prediction.")
        payload = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture()
def gateway():
    FakeGateway.calls, FakeGateway.fail_first, FakeGateway.bodies = 0, 0, []
    server = HTTPServer(("127.0.0.1", 0), FakeGateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/api/v1"
    server.shutdown()


MESSAGES = [{"role": "system", "content": "s"},
            {"role": "user", "content": json.dumps({"model_output": {"predicted_class": "DoS", "confidence_percent": 90},
                                                    "flow": {"source_host": "host_1"}})}]


def test_client_retries_then_caches(gateway, tmp_path):
    FakeGateway.fail_first = 1
    client = ARCClient("sk-test", "m", tmp_path / "c.jsonl", base_url=gateway, backoff_seconds=0)
    first = client.complete(MESSAGES)
    assert FakeGateway.calls == 2
    path, auth, body = FakeGateway.bodies[-1]
    assert path == "/api/v1/chat/completions" and auth == "Bearer sk-test" and body["model"] == "m"

    reopened = ARCClient("sk-test", "m", tmp_path / "c.jsonl", base_url=gateway, backoff_seconds=0)
    assert reopened.complete(MESSAGES) == first
    assert FakeGateway.calls == 2


def test_client_gives_up_cleanly(gateway, tmp_path):
    FakeGateway.fail_first = 99
    client = ARCClient("k", "m", tmp_path / "c.jsonl", base_url=gateway, backoff_seconds=0, max_attempts=2)
    with pytest.raises(LLMError, match="Gave up"):
        client.complete(MESSAGES)


def run_cli(bundle: dict, out: Path, *extra: str, env: dict | None = None):
    root = bundle["root"]
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_explanations",
         "--predictions", str(root / "predictions.npz"), "--graphs", str(bundle["graphs"]),
         "--scaler", str(root / "scaler.joblib"), "--feature-names", str(root / "feature_names.joblib"),
         "--out", str(out), "--per-cell", "1", *extra],
        cwd=REPO, capture_output=True, text=True, env=env,
    )


def test_cli_dry_run_needs_no_key(bundle, tmp_path):
    out = tmp_path / "dry"
    result = run_cli(bundle, out, "--dry-run", env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    assert (out / "prompts.jsonl").is_file() and (out / "evidence.jsonl").is_file()
    assert not (out / "review_sheet.csv").exists()


def test_cli_full_run_against_gateway(bundle, tmp_path, gateway):
    out = tmp_path / "full"
    env = {"PATH": "/usr/bin:/bin", "ARC_API_KEY": "sk-test"}
    result = run_cli(bundle, out, "--model", "test-model", "--base-url", gateway, env=env)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["automated_checks"]["alerts"] >= 3
    assert summary["automated_checks"]["auto_passed"] == summary["automated_checks"]["alerts"]
    calls_after_first_run = FakeGateway.calls

    result = run_cli(bundle, out, "--model", "test-model", "--base-url", gateway, env=env)
    assert result.returncode == 0, result.stderr
    assert FakeGateway.calls == calls_after_first_run


def test_prompt_states_grounding_rules():
    for phrase in ("only source of facts", "Copy every number", "consistent with the prediction"):
        assert phrase in SYSTEM_PROMPT
