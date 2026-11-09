"""Export a trained run in the team's shared GNN artefact format (src/comparison/gnn_importer.py).

Usage: python member3/export_predictions.py <run_dir> <split> [out_dir]
  split: val | test.  Default out_dir: member3/predictions/<run_name>_<split>/
Writes metrics.json (via src/eval/metrics.compute_metrics), run_config.json and predictions.npz
with graph_id = index of the sorted graph file, edge_id = row within that graph, and softmax
probabilities; then re-loads the folder with load_gnn_artifacts to check it validates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from model import EGraphSAGE  # noqa: E402
from train import load_split  # noqa: E402
from src.comparison.gnn_importer import load_gnn_artifacts  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402


def main(run_dir, split, out_dir=None):
    run = Path(run_dir)
    if split not in ("val", "test"):
        sys.exit("split must be val or test")
    cfg = json.loads((run / "config.json").read_text())
    d, m = cfg["data"], cfg["model"]
    out = Path(out_dir) if out_dir else HERE / "predictions" / f"{run.name}_{split}"
    if out.exists():
        sys.exit(f"STOP: {out} exists; refusing to overwrite.")

    names = json.loads((Path(d["graph_dir"]) / "class_names.json").read_text())
    benign = d["benign_class"]
    graphs = load_split(d["graph_dir"], split, d["label_key"])  # sorted files: index = graph_id
    net = EGraphSAGE(graphs[0].edge_attr.size(1), m["hidden_dim"], m["num_layers"], cfg["num_classes"],
                     m["dropout"], m["aggr"], m["head_edge_attr"], m["bidirectional"])
    net.load_state_dict(torch.load(run / "best.pt", weights_only=True, map_location="cpu"))
    net.eval()

    y, proba, gid, eid = [], [], [], []
    with torch.no_grad():
        for i, g in enumerate(graphs):
            p = net(g.x, g.edge_index, g.edge_attr).softmax(1)
            y.append(g.y.numpy())
            proba.append(p.numpy().astype(np.float32))
            gid.append(np.full(len(g.y), i, dtype=np.int64))
            eid.append(np.arange(len(g.y), dtype=np.int64))
    y, proba = np.concatenate(y).astype(np.int64), np.concatenate(proba)
    pred = proba.argmax(1).astype(np.int64)
    y_bin, pred_bin = (y != benign).astype(np.int64), (pred != benign).astype(np.int64)

    out.mkdir(parents=True)
    np.savez_compressed(
        out / "predictions.npz",
        y_true_multiclass=y, y_pred_multiclass=pred, y_true_binary=y_bin, y_pred_binary=pred_bin,
        y_proba_multiclass=np.clip(proba, 0.0, 1.0), graph_id=np.concatenate(gid), edge_id=np.concatenate(eid),
        class_names=np.array(names, dtype=str), benign_class_id=np.int64(benign),
        dataset=np.str_(Path(d["graph_dir"]).name.removeprefix("graphs_")), split=np.str_(split),
    )
    metrics = compute_metrics(y, pred, names, binary_true=y_bin, binary_pred=pred_bin)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))

    loaded = load_gnn_artifacts(out)  # raises if the shared schema is not met
    assert len(loaded["predictions"]["y_true_multiclass"]) == len(y)
    print(f"{out}: {len(y):,} flows | macro F1 {metrics['macro_f1']:.3f} | "
          f"weighted F1 {metrics['weighted_f1']:.3f} | binary F1 {metrics['binary_f1']:.3f} | validated")


if __name__ == "__main__":
    main(*sys.argv[1:4])
