"""Grid sweep: expand a sweep spec into config files (commit them), then optionally run them.
  python member3/sweep.py member3/configs/sweep_unsw_arch.json         # write configs + count runs
  python member3/sweep.py member3/configs/sweep_unsw_arch.json --run   # run all; skips finished runs
"""
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

ABBR = {"num_layers": "L", "hidden_dim": "h", "lr": "lr", "class_weighting": "w",
        "aggr": "a", "dropout": "d", "weight_decay": "wd"}


def apply(cfg, overrides):
    """Return a copy of cfg with {"section.key": value} overrides set."""
    cfg = copy.deepcopy(cfg)
    for key, v in overrides.items():
        *path, leaf = key.split(".")
        node = cfg
        for p in path:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"{key} is not in the base config")  # catches typos in the spec
        node[leaf] = v
    return cfg


def expand(spec):
    """Yield (name, config) for spec["grid"] ({"section.key": [values]}, every combination)
    or spec["variants"] ({"name": {"section.key": value}}, one named config each)."""
    base = json.loads(Path(spec["base"]).read_text())
    base["out_dir"] = spec["out_dir"]
    if "variants" in spec:
        for name, overrides in spec["variants"].items():
            yield name, apply(base, overrides)
        return
    keys = list(spec["grid"])
    for values in itertools.product(*spec["grid"].values()):
        name = "_".join(f"{ABBR.get(k.split('.')[-1], k.split('.')[-1])}{v}" for k, v in zip(keys, values))
        yield name, apply(base, dict(zip(keys, values)))


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(spec["config_dir"])
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cfg in expand(spec):
        p = out / f"{name}.json"
        text = json.dumps(cfg, indent=2)
        if p.exists() and p.read_text() != text:
            sys.exit(f"STOP: {p} exists with different contents; refusing to overwrite.")
        p.write_text(text)
        paths.append(p)
    print(f"{len(paths)} configs x {len(spec['seeds'])} seeds = "
          f"{len(paths) * len(spec['seeds'])} runs -> {out}")
    if "--run" not in sys.argv:
        return

    failed = []
    for seed in spec["seeds"]:  # seed outermost: one full pass finishes before repeats start
        for p in paths:
            if (Path(spec["out_dir"]) / f"{p.stem}_seed{seed}" / "results.json").exists():
                continue
            r = subprocess.run([sys.executable, str(Path(__file__).parent / "train.py"), str(p), str(seed)])
            if r.returncode:
                failed.append(f"{p.stem} seed{seed}")
    print("failed:", failed or "none")


if __name__ == "__main__":
    main()
