# Baselines Experiment Runner

This document describes the **experiment runner** used to launch the three
baseline training scripts (Random Forest, XGBoost and PyTorch MLP) in a uniform
and reproducible way.

## Architecture

* **CLI entry point** – ``python -m src.experiment.runner``
* **Configuration drives execution** – a JSON file specifies the model, dataset,
  seeds, hyper‑parameters and any one‑factor‑at‑a‑time ablations.
* **Multi‑seed expansion** – the runner iterates over every seed listed in the
  configuration and creates an isolated output directory for each run.
* **Ablation handling** – for each ablation factor only the values listed are
  varied while everything else (including the seed list) stays fixed.
* **Dry‑run mode** – the default behaviour prints the commands that *would* be
  executed and exits.  Add ``--run`` to actually launch the trainers.
* **Safe resume** – a run is considered finished when a ``metrics.json`` file
  exists and can be parsed.  Completed runs are skipped on subsequent invocations.

## JSON schema (simplified)

```json
{
  "experiment_name": "unsw_rf_ablations",
  "model": "rf",
  "dataset": "unsw",
  "seeds": [42, 43, 44],
  "output_root": "results/experiments",
  "max_samples": 20000,
  "base_params": {
    "n_estimators": 100,
    "max_depth": 20,
    "workers": 4
  },
  "ablations": {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30]
  }
}
```

* **Required keys** – ``experiment_name``, ``model``, ``dataset``, ``seeds`` and
  ``output_root``.
* **Optional keys** – ``max_samples``, ``base_params`` (model‑specific defaults)
  and ``ablations``.
* **Ablations** – only one factor is changed at a time; the runner generates the
  Cartesian product *per factor* (not the full cross‑product).

## Supported model‑specific hyper‑parameters

| Model | Flags recognised by the underlying trainer |
|-------|--------------------------------------------|
| rf    | ``--n_estimators``, ``--max_depth``, ``--min_samples_leaf``, ``--workers`` |
| xgb   | ``--n_estimators``, ``--max_depth``, ``--learning_rate``, ``--workers`` |
| mlp   | ``--learning_rate``, ``--batch_size``, ``--max_epochs``, ``--dropout``, ``--workers`` |

The runner forwards any additional ``--key value`` pairs verbatim to the trainer.

## Multi‑seed workflow

1. The runner reads the JSON configuration.
2. It expands the configuration into a list of concrete runs (one per seed and
   per ablation value).
3. For each run it computes a deterministic *configuration ID* (SHA‑1 of the
   JSON fragment) and creates a directory hierarchy:

   ````text
   <output_root>/
       <experiment_name>/
           <model>/
               <config_id>/
                   seed_<seed>/
   ````

4. The runner checks for ``metrics.json`` inside the seed directory – if present
   the run is considered complete and is skipped.
5. If not completed, the runner builds the command line for the appropriate
   ``train_*.py`` script, adding ``--output-dir`` pointing at the seed directory.
6. In dry‑run mode the commands are printed; with ``--run`` they are executed via
   ``subprocess.run``.

## Ablation workflow

Only *one* factor is varied per generated configuration.  For the example above
the runner would create the following configurations (abbreviated):

* ``n_estimators = 50`` (all other parameters at baseline values)
* ``n_estimators = 100``
* ``n_estimators = 200``
* ``max_depth = 10``
* ``max_depth = 20``
* ``max_depth = 30``

Each of those configurations is then expanded over the list of seeds.

## Training sample budgets & class weighting

The ``--max_samples`` flag limits the number of training edges (0 = use all).
Class weighting is already built into the trainer scripts via ``compute_class_weight``
and is enabled by default.  If a trainer ever lacks a weighting flag, a small
backward‑compatible option can be added without affecting the runner.

## Dry‑run command example

```bash
uv run --locked python -m src.experiment.runner \
    --config configs/experiments/unsw_rf_ablations.json
```

The output will list the number of planned commands and the exact command lines.

## Actual execution (with ``--run``)

```bash
uv run --locked python -m src.experiment.runner \
    --config configs/experiments/unsw_rf_ablations.json \
    --run
```

The runner will invoke the appropriate trainer scripts.  Existing successful runs
are skipped; failures are reported but do not delete prior artefacts.

## Resume behaviour

* **Completed run** – ``metrics.json`` exists and parses → **skip**.
* **Incomplete run** – missing or malformed ``metrics.json`` → the runner will
  attempt the run again.
* **Incompatible configuration** – a different ``config_id`` creates a new
  directory, never overwriting old results.

## Result aggregation (outside the scope of this file)

Aggregating the metrics across seeds is performed by helper functions in the
``src.experiment`` package (not covered by the current tests).  The aggregation
produces CSV/JSON summary files containing mean and sample standard deviation for
each metric.

---

*The runner deliberately avoids any heavy data processing – the heavy lifting is
performed by the existing ``train_*.py`` scripts.*
