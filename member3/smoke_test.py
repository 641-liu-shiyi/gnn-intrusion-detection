"""THROWAWAY smoke test: synthetic graphs in a temp dir, deleted on exit.
Only checks that load -> forward -> backward -> checkpoint runs. Not data, not results."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train  # noqa: E402
from model import EdgeSAGEConv, EGraphSAGE  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    gdir = tmp / "graphs"
    gdir.mkdir()
    rng = np.random.default_rng(0)
    for split in ["train", "val", "test"]:
        for i in range(3):
            np.savez(gdir / f"{split}_graph_{i:05d}.npz",
                     edge_index=rng.integers(0, 20, (2, 200)),
                     edge_attr=rng.normal(size=(200, 37)).astype(np.float32),
                     edge_label=rng.integers(0, 10, 200),
                     num_nodes=20)

    # Guard must reject TTL features.
    joblib.dump(["IN_BYTES", "MIN_TTL"], tmp / "bad.joblib")
    try:
        train.check_features(tmp / "bad.joblib")
        raise AssertionError("TTL guard did not fire")
    except SystemExit:
        pass

    joblib.dump(["IN_BYTES", "OUT_BYTES"], tmp / "ok.joblib")
    for aggr in ["mean", "attention"]:
        cfg = json.loads((Path(__file__).parent / "configs" / f"unsw_egraphsage_{aggr}.json").read_text())
        cfg["out_dir"] = str(tmp / "runs")
        cfg["data"] = {"graph_dir": str(gdir), "feature_names": str(tmp / "ok.joblib"), "label_key": "edge_label", "benign_class": 0}
        cfg["train"]["epochs"] = 2
        (tmp / f"smoke_{aggr}.json").write_text(json.dumps(cfg))
        train.main(tmp / f"smoke_{aggr}.json", seed=1)
        res = json.loads((tmp / "runs" / f"smoke_{aggr}_seed1" / "results.json").read_text())
        assert np.isfinite(res["test_loss"]), aggr

    # Sweep: 2-config grid runs end to end, a rerun skips finished runs, summarize ranks them.
    here = Path(__file__).parent
    spec = {"base": str(tmp / "smoke_mean.json"), "config_dir": str(tmp / "sweep_cfgs"),
            "out_dir": str(tmp / "sweep_runs"), "seeds": [0],
            "grid": {"model.hidden_dim": [8, 16], "train.class_weighting": ["inverse"]}}
    (tmp / "spec.json").write_text(json.dumps(spec))
    for _ in range(2):  # second pass must skip both runs
        out = subprocess.run([sys.executable, str(here / "sweep.py"), str(tmp / "spec.json"), "--run"],
                             capture_output=True, text=True)
        assert "failed: none" in out.stdout, out.stdout + out.stderr
    assert "TEST:" not in out.stdout, "rerun did not skip finished runs"
    cfg8 = json.loads((tmp / "sweep_cfgs" / "h8_winverse.json").read_text())
    assert cfg8["model"]["hidden_dim"] == 8 and cfg8["out_dir"] == spec["out_dir"]
    out = subprocess.run([sys.executable, str(here / "summarize.py"), spec["out_dir"], "best_val_loss"],
                         capture_output=True, text=True, check=True).stdout
    assert "#1 " in out and "#2 " in out, out
    out = subprocess.run([sys.executable, str(here / "summarize.py"), spec["out_dir"], "best_val_loss",
                          "--latex=best_val_loss,test_loss,missing_key"],
                         capture_output=True, text=True, check=True).stdout
    assert r"\begin{tabular}{lccc}" in out and r"h8\_winverse" in out and "--" in out, out

    # Every ablation variant builds and gives one prediction per edge, including the
    # edge whose source host only sends (bidirectional messages).
    import sweep  # noqa: E402
    ablation = json.loads((here / "configs" / "ablation_unsw.json").read_text())
    ablation["base"] = str(here / "configs" / "unsw_egraphsage_mean.json")
    variants = dict(sweep.expand(ablation))
    assert list(variants) == list(ablation["variants"])
    for name, vcfg in variants.items():
        m = vcfg["model"]
        net = EGraphSAGE(4, 8, m["num_layers"], 3, m["dropout"], m["aggr"],
                         m["head_edge_attr"], m["bidirectional"])
        out = net(torch.ones(3, 1), torch.tensor([[0, 0], [1, 2]]), torch.randn(2, 4))
        assert out.shape == (2, 3), name
    assert variants["no_graph"]["model"]["num_layers"] == 0

    # Error analysis runs on a finished run and writes the confusion matrix.
    out = subprocess.run([sys.executable, str(here / "analyze.py"), str(tmp / "runs" / "smoke_mean_seed1")],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "confusion matrix saved" in out.stdout, out.stdout + out.stderr

    # Transfer (option C): runs, refuses to overwrite, refuses mismatched feature lists.
    tgt = json.loads((tmp / "smoke_mean.json").read_text())
    (tmp / "smoke_target.json").write_text(json.dumps(tgt))
    run1 = str(tmp / "runs" / "smoke_mean_seed1")
    t = [sys.executable, str(here / "transfer.py"), run1]
    out = subprocess.run(t + [str(tmp / "smoke_target.json")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    tr = json.loads((tmp / "runs" / "smoke_mean_seed1" / "transfer_to_smoke_target.json").read_text())
    assert {"in_domain", "cross_domain"} <= tr.keys()
    assert sum(n for _, n in tr["cross_domain"]["flagged_by_true_class"].values()) == tr["cross_domain"]["edges"]
    out = subprocess.run(t + [str(tmp / "smoke_target.json")], capture_output=True, text=True)
    assert "refusing to overwrite" in out.stderr, out.stderr
    joblib.dump(["IN_BYTES", "IN_PKTS"], tmp / "other.joblib")
    tgt["data"]["feature_names"] = str(tmp / "other.joblib")
    (tmp / "smoke_target2.json").write_text(json.dumps(tgt))
    out = subprocess.run(t + [str(tmp / "smoke_target2.json")], capture_output=True, text=True)
    assert "feature lists differ" in out.stderr, out.stderr

    # Attention weights sum to 1 per node: with identical incoming messages,
    # attention must reproduce mean exactly (any normalised weighting of equal vectors does).
    mean_conv = EdgeSAGEConv(1, 4, 8, "mean")
    att_conv = EdgeSAGEConv(1, 4, 8, "attention")
    att_conv.load_state_dict(mean_conv.state_dict(), strict=False)
    ei = torch.tensor([[0, 1, 3], [2, 2, 2]])
    ea = torch.ones(3, 4)
    x = torch.ones(4, 1)
    assert torch.allclose(mean_conv(x, ei, ea), att_conv(x, ei, ea), atol=1e-6)

print("SMOKE TEST PASSED (synthetic data deleted)")
