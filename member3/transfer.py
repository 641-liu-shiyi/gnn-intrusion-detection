"""Phase 5: test a trained model on the other dataset (team decision: option C).
The source model keeps its multi-class head. Any prediction other than the source's benign class
counts as "attack", and the target's true labels collapse the same way (anything not the target's
benign class). The identical collapse is applied to the source's own test set, so the in-domain and
cross-domain benign-vs-attack numbers come from the same model and are directly comparable.
Usage: python member3/transfer.py <source_run_dir> <target_config.json>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model import EGraphSAGE  # noqa: E402
from train import check_features, load_evaluator, load_split  # noqa: E402


def predict(model, graphs, device, batch_graphs):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=batch_graphs):
            b = b.to(device)
            ps.append(model(b.x, b.edge_index, b.edge_attr).argmax(1).cpu())
            ys.append(b.y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main(run_dir, target_cfg_path):
    run = Path(run_dir)
    src = json.loads((run / "config.json").read_text())
    tgt = json.loads(Path(target_cfg_path).read_text())
    tag = Path(target_cfg_path).stem
    out_path = run / f"transfer_to_{tag}.json"
    if out_path.exists():
        sys.exit(f"STOP: {out_path} exists; refusing to overwrite.")
    for name, c in (("source", src), ("target", tgt)):
        if c["data"].get("benign_class") is None:
            sys.exit(f"STOP: set data.benign_class in the {name} config (from Member 1's class mapping).")
    if check_features(src["data"]["feature_names"]) != check_features(tgt["data"]["feature_names"]):
        sys.exit("STOP: source and target feature lists differ (names or order); columns would not line up.")

    device = torch.device(src["device"] if torch.cuda.is_available() else "cpu")
    src_test = load_split(src["data"]["graph_dir"], "test", src["data"]["label_key"])
    tgt_test = load_split(tgt["data"]["graph_dir"], "test", tgt["data"]["label_key"])
    m = src["model"]
    model = EGraphSAGE(src_test[0].edge_attr.size(1), m["hidden_dim"], m["num_layers"], src["num_classes"],
                       m["dropout"], m["aggr"], m["head_edge_attr"], m["bidirectional"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", weights_only=True, map_location=device))
    evaluate = load_evaluator(src.get("eval", {}))
    src_benign = src["data"]["benign_class"]

    result: dict = {"source_run": str(run), "target_config": str(target_cfg_path)}
    for split, graphs, true_benign in (("in_domain", src_test, src_benign),
                                       ("cross_domain", tgt_test, tgt["data"]["benign_class"])):
        y_class, p_class = predict(model, graphs, device, src["train"]["batch_graphs"])
        y = (y_class != true_benign).astype(np.int64)  # true labels: that dataset's own benign id
        p = (p_class != src_benign).astype(np.int64)   # predictions live in the source label space
        np.savez(run / f"transfer_{split}_{tag}.npz", y_true_class=y_class, y_true=y, y_pred=p)
        flagged = {int(c): [int(p[y_class == c].sum()), int((y_class == c).sum())]
                   for c in np.unique(y_class)}
        r = {"edges": len(y), "attack_fraction": float(y.mean()), "flagged_by_true_class": flagged}
        if evaluate:
            r["binary"] = evaluate(y, p)
        result[split] = r
        print(f"\n{split}: {len(y):,} edges, {y.mean():.1%} attack")
        for c, (k, n) in flagged.items():
            print(f"  true class {c:>3d}: flagged as attack {k:>9,d} / {n:>9,d}")

    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
