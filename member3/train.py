"""Config-driven E-GraphSAGE training.
Usage: python member3/train.py member3/configs/<name>.json [seed]
The optional seed overrides the config's and is recorded in the run's saved config.json."""
import importlib
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import EGraphSAGE

# Brief: IPs, timestamps and IP-header TTL must never enter the feature matrix.
# DNS_TTL_ANSWER (DNS cache lifetime) and FLOW_DURATION_* are allowed (decided 2026-10-09).
FORBIDDEN_SUBSTR = ("ADDR", "FLOW_START", "FLOW_END")
FORBIDDEN_EXACT = ("MIN_TTL", "MAX_TTL")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ponytail: CUDA scatter (mean aggregation) is not bitwise deterministic; warn_only keeps it runnable.
    torch.use_deterministic_algorithms(True, warn_only=True)


def check_features(feature_names_path):
    if not Path(feature_names_path).exists():
        sys.exit(f"STOP: {feature_names_path} not found. Member 1's data is not in place yet.")
    names = joblib.load(feature_names_path)
    bad = [n for n in names if n in FORBIDDEN_EXACT or any(f in n for f in FORBIDDEN_SUBSTR)]
    if bad:
        sys.exit(f"STOP: forbidden features in edge_attr: {bad}. Fix belongs to Member 1's pipeline.")
    return names


def load_split(graph_dir, split, label_key):
    """label_key: "edge_attack" (attack-type ids, the main task) or "edge_label" (binary)."""
    files = sorted(Path(graph_dir).glob(f"{split}_graph_*.npz"))
    if not files:
        sys.exit(f"STOP: no {split} graphs in {graph_dir}")
    with np.load(files[0]) as g:
        if label_key not in g.files:
            sys.exit(f"STOP: {files[0].name} has no '{label_key}' (keys: {g.files}).")
    graphs = []
    # ponytail: whole split held in memory (~2 GB for CICIDS train); stream from disk if memory gets tight.
    for f in files:
        with np.load(f) as g:
            n = int(g["num_nodes"])
            graphs.append(Data(
                x=torch.ones(n, 1),
                edge_index=torch.from_numpy(g["edge_index"]).long(),
                edge_attr=torch.from_numpy(g["edge_attr"]).float(),
                y=torch.from_numpy(g[label_key]).long(),
                num_nodes=n,
            ))
    return graphs


def class_weights(graphs, num_classes, scheme):
    counts = torch.bincount(torch.cat([g.y for g in graphs]), minlength=num_classes).float()
    if scheme == "none":
        return None, counts
    w = counts.sum() / (num_classes * counts.clamp(min=1))
    if scheme == "inverse_sqrt":
        w = w.sqrt()
    elif scheme != "inverse":
        sys.exit(f"unknown class_weighting: {scheme}")
    return w, counts


def run_epoch(model, loader, loss_fn, device, opt=None):
    model.train(opt is not None)
    total, n, ys, ps = 0.0, 0, [], []
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            b = b.to(device)
            out = model(b.x, b.edge_index, b.edge_attr)
            loss = loss_fn(out, b.y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * b.y.numel()
            n += b.y.numel()
            ys.append(b.y.cpu())
            ps.append(out.argmax(1).cpu())
    return total / n, torch.cat(ys).numpy(), torch.cat(ps).numpy()


def load_evaluator(cfg):
    """Member 2's shared metrics module. Never reimplemented here."""
    if not cfg.get("module"):
        print("WARNING: no eval module configured -> loss only, numbers NOT reportable.")
        return None
    return getattr(importlib.import_module(cfg["module"]), cfg["function"])


def main(config_path, seed=None):
    cfg = json.loads(Path(config_path).read_text())
    if seed is not None:
        cfg["seed"] = seed
    run_dir = Path(cfg["out_dir"]) / f"{Path(config_path).stem}_seed{cfg['seed']}"
    if run_dir.exists():
        sys.exit(f"STOP: {run_dir} exists; refusing to overwrite.")

    # Validate inputs before creating anything on disk.
    d = cfg["data"]
    check_features(d["feature_names"])
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    train, val, test = (load_split(d["graph_dir"], s, d["label_key"]) for s in ("train", "val", "test"))

    C = cfg["num_classes"]
    if C is None:
        sys.exit("STOP: num_classes is not set; take it from Member 1's class mapping (Phase 1 output).")
    seen = torch.cat([g.y for g in train]).unique().tolist()
    if max(seen) >= C:
        sys.exit(f"STOP: label {max(seen)} >= num_classes {C}")
    if C > 2 and len(seen) <= 2:
        sys.exit(f"STOP: num_classes={C} but train labels are only {seen}; these look like binary "
                 "benign/attack labels, not attack types.")
    print(f"train labels seen: {seen} (num_classes={C})")

    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    m = cfg["model"]
    model = EGraphSAGE(train[0].edge_attr.size(1), m["hidden_dim"], m["num_layers"], C,
                       m["dropout"], m["aggr"], m["head_edge_attr"], m["bidirectional"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | model: {m} | params: {n_params:,}")
    w, counts = class_weights(train, C, cfg["train"]["class_weighting"])
    print("train class counts:", counts.long().tolist())
    loss_fn = torch.nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    t = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=t["batch_graphs"], shuffle=True, generator=g)
    val_loader = DataLoader(val, batch_size=t["batch_graphs"])
    test_loader = DataLoader(test, batch_size=t["batch_graphs"])
    evaluate = load_evaluator(cfg.get("eval", {}))

    # Model selection on val loss until Member 2's module is wired in.
    best, bad_epochs = float("inf"), 0
    with open(run_dir / "epochs.jsonl", "w") as log:
        for epoch in range(1, t["epochs"] + 1):
            t0 = time.time()
            tr_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, opt)
            va_loss, yv, pv = run_epoch(model, val_loader, loss_fn, device)
            rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                   "sec": round(time.time() - t0, 1)}
            if evaluate:
                rec["val"] = evaluate(yv, pv)
            log.write(json.dumps(rec, default=float) + "\n")
            log.flush()
            print(json.dumps(rec, default=float))
            if va_loss < best:
                best, best_rec, bad_epochs = va_loss, rec, 0
                torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= t["patience"]:
                    print(f"early stop at epoch {epoch}")
                    break

    if best == float("inf"):
        sys.exit("STOP: validation loss was never finite (NaN/inf); training diverged. Try a lower lr.")
    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))
    te_loss, yt, pt = run_epoch(model, test_loader, loss_fn, device)
    # Pick hyperparameters from "val_at_best" only; "test" is reported, never used for selection.
    result = {"test_loss": te_loss, "best_val_loss": best, "best_epoch": best_rec["epoch"],
              "params": n_params, "train_class_counts": counts.long().tolist(),
              "val_at_best": best_rec.get("val", {})}
    if evaluate:
        result["test"] = evaluate(yt, pt)
    np.savez(run_dir / "test_predictions.npz", y_true=yt, y_pred=pt)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print("TEST:", json.dumps(result, default=float))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
