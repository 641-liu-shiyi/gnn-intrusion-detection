"""Phase 5: test a trained model on the other dataset (team decision: option C).
The source model keeps its multi-class head; any prediction other than the source's benign class
counts as "attack", and the target's true labels collapse the same way. Following the baselines'
protocol (src/eval/cross_dataset.py), target features are moved into the source's preprocessing
space with src/eval/preprocess_cross.inverse_transform_and_reorder, and metrics come from
src/eval/metrics. The same collapse is applied to the source's own split, so in-domain and
cross-domain benign-vs-attack numbers come from the same model.
Usage: python member3/transfer.py <source_run_dir> <target_config.json> [val|test]   (default val)"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from model import EGraphSAGE  # noqa: E402
from train import check_features, load_split  # noqa: E402
from src.eval.metrics import compute_metrics, compute_proba_metrics  # noqa: E402
from src.eval.preprocess_cross import inverse_transform_and_reorder  # noqa: E402

BINARY = ["Benign", "Attack"]


def predict(model, graphs, src_benign):
    """Per-graph inference; returns true class ids and P(attack) = 1 - P(source benign)."""
    ys, scores = [], []
    with torch.no_grad():
        for g in graphs:
            prob = model(g.x, g.edge_index, g.edge_attr).softmax(1)
            ys.append(g.y.numpy())
            scores.append((1.0 - prob[:, src_benign]).clamp(0, 1).numpy())
    return np.concatenate(ys), np.concatenate(scores)


def main(run_dir, target_cfg_path, split="val"):
    run = Path(run_dir)
    src = json.loads((run / "config.json").read_text())
    tgt = json.loads(Path(target_cfg_path).read_text())
    tag = f"{Path(target_cfg_path).stem}_{split}"
    out_path = run / f"transfer_to_{tag}.json"
    if out_path.exists():
        sys.exit(f"STOP: {out_path} exists; refusing to overwrite.")
    for name, c in (("source", src), ("target", tgt)):
        if c["data"].get("benign_class") is None:
            sys.exit(f"STOP: set data.benign_class in the {name} config (from Member 1's class mapping).")
    src_feats = check_features(src["data"]["feature_names"])
    tgt_feats = check_features(tgt["data"]["feature_names"])
    src_scaler = joblib.load(Path(src["data"]["feature_names"]).parent / "scaler.joblib")
    tgt_scaler = joblib.load(Path(tgt["data"]["feature_names"]).parent / "scaler.joblib")

    src_graphs = load_split(src["data"]["graph_dir"], split, src["data"]["label_key"])
    tgt_graphs = load_split(tgt["data"]["graph_dir"], split, tgt["data"]["label_key"])
    for g in tgt_graphs:  # target features -> source preprocessing space
        g.edge_attr = torch.from_numpy(inverse_transform_and_reorder(
            g.edge_attr.numpy(), tgt_scaler, list(tgt_feats), src_scaler, list(src_feats)).astype(np.float32))

    m = src["model"]
    model = EGraphSAGE(src_graphs[0].edge_attr.size(1), m["hidden_dim"], m["num_layers"], src["num_classes"],
                       m["dropout"], m["aggr"], m["head_edge_attr"], m["bidirectional"])
    model.load_state_dict(torch.load(run / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    src_benign = src["data"]["benign_class"]

    result: dict = {"source_run": str(run), "target_config": str(target_cfg_path), "split": split}
    for name, graphs, cfg in (("in_domain", src_graphs, src), ("cross_domain", tgt_graphs, tgt)):
        y_class, score = predict(model, graphs, src_benign)
        y = (y_class != cfg["data"]["benign_class"]).astype(np.int64)  # that dataset's own benign id
        p = (score >= 0.5).astype(np.int64)  # = argmax is non-benign only when P(benign) < 0.5
        metrics = compute_metrics(y, p, BINARY, benign_id=0)
        metrics.update(compute_proba_metrics(y, np.stack([1 - score, score], 1), BINARY, benign_id=0))
        names = json.loads((Path(cfg["data"]["graph_dir"]) / "class_names.json").read_text())
        flagged = {names[int(c)]: [int(p[y_class == c].sum()), int((y_class == c).sum())]
                   for c in np.unique(y_class)}
        result[name] = {"edges": len(y), "attack_fraction": float(y.mean()),
                        "flagged_by_true_class": flagged, "binary": metrics}
        print(f"\n{name} ({split}): {len(y):,} flows, {y.mean():.1%} attack | binary F1 "
              f"{metrics['binary_f1']:.3f} P {metrics['binary_precision']:.3f} R {metrics['binary_recall']:.3f} "
              f"AP {metrics['binary_average_precision']:.3f}")
        for c, (k, n) in flagged.items():
            print(f"  {c:28s} flagged as attack {k:>9,d} / {n:>9,d}")

    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
