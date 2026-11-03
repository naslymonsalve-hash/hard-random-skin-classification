
"""Executes HARD and RANDOM with five rates, ten seeds, and ONE fixed split.

Execute this file from the environment you already use for training.
There is no fold loop. Only fully completed executions are skipped.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


# CONFIGURATION: these are the five already selected rates, not a new sweep.
R_VALUES = [0.0, 0.1, 0.2, 0.3, 0.6]
TRAINING_SEEDS = [42, 123, 2024, 7, 21, 77, 101, 256, 512, 1024] #20 seeds: 42, 84, 152, 345, 389, 412, 576, 780, 891, 1024, 1105, 1243, 1367, 1450, 1589, 1634, 1756, 1890, 1945, 2012
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.20  # Fraction of lesion groups; not of the external test.
SHARED_BASELINE = True  # True: 90 trainings. False: 100 trainings.

RESULTS_FOLDER = Path(__file__).resolve().parent / "results_10seeds_no_cv_flops" #stage 2 without cross validation
MAIN_FILE = Path(__file__).with_name("4_train_ISIC2019_no_cv_new.py")

METRICS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_avg",
    "auc_macro_ovr", "validation_loss", "test_loss", "training_time_seconds",
    "training_time_minutes", "executed_epochs", "mean_time_per_epoch_seconds",
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_plan():
    """One execution per method, rate, and seed; shared baseline once."""
    jobs = []
    for seed in TRAINING_SEEDS:
        for rate in sorted(R_VALUES):
            for method in ("hard", "random"):
                if SHARED_BASELINE and rate == 0 and method == "random":
                    continue
                jobs.append((method, float(rate), int(seed)))
    return jobs


def run_folder(root, method, rate, seed):
    token = f"{rate:.1f}".replace(".", "")
    return Path(root) / f"results_NasNet_ISIC2019_3classes_{method}_r{token}_Seed{seed}_NoCV"


def read_csv(path):
    return pd.read_csv(path, sep=";", decimal=",")


def is_completed(root, job, trainer_hash):
    method, rate, seed = job
    folder = run_folder(root, method, rate, seed)
    marker = folder / "run_completed.json"
    if not marker.exists():
        return None
    done = json.loads(marker.read_text(encoding="utf-8"))
    reference_file = Path(root) / "_fixed_split" / "fixed_split.json"
    if not reference_file.exists():
        raise RuntimeError("There are completed results but the fixed split record is missing.")
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    expected = {
        "selection_method": method, "drop_rate": rate, "seed": seed,
        "split_seed": SPLIT_SEED, "evaluation_protocol": "fixed_holdout",
        "validation_fraction_groups": VALIDATION_FRACTION,
        "trainer_sha256": trainer_hash, "split_id": reference["split_id"],
    }
    if any(done.get(k) != value for k, value in expected.items()):
        raise RuntimeError(f"Incompatible configuration with the results from {folder}. "
                           "Use another results folder for a different protocol.")
    required = {"validation_metrics.csv", "test_metrics.csv", "training_time.csv",
                "classification_report.csv", "confusion_matrix.csv",
                "train_pure_split.csv", "validation_split.csv", "test_split.csv"}
    saved = done.get("artifacts_sha256", {})
    if not required.issubset(saved):
        raise RuntimeError(f"Incomplete completion record: {folder}")
    for name in sorted(required):
        path = folder / name
        if not path.is_file() or file_hash(path) != saved[name]:
            raise RuntimeError(f"The completed result was modified or is incomplete: {path}")
    for name in ("validation_metrics.csv", "test_metrics.csv"):
        frame = read_csv(folder / name)
        if len(frame) != 1 or not math.isfinite(float(frame.iloc[0]["f1_macro"])):
            raise RuntimeError(f"Invalid metrics: {folder / name}")
    return folder


def execute(root, job):
    method, rate, seed = job
    env = os.environ.copy()
    env.update({
        "EXPERIMENT_OUTPUT_ROOT": str(root),
        "EXPERIMENT_DROP_RATE": str(rate),
        "EXPERIMENT_SEED": str(seed),
        "EXPERIMENT_SPLIT_SEED": str(SPLIT_SEED),
        "EXPERIMENT_SELECTION_METHOD": method,
        "EXPERIMENT_CV_FOLD": "-1",  # -1 WITHOUT cross-validation.
        "EXPERIMENT_VALIDATION_FRACTION": str(VALIDATION_FRACTION),
        "PYTHONHASHSEED": str(seed),
    })
    subprocess.run([sys.executable, str(MAIN_FILE)],
                   cwd=MAIN_FILE.resolve().parent, env=env, check=True)


def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def consolidate(root, trainer_hash):
    """Summarizes available seeds; does not average non-existent folds."""
    frames = {"validation": [], "test": []}
    for job in full_plan():
        folder = is_completed(root, job, trainer_hash)
        if folder is None:
            continue
        method, rate, seed = job
        configuration = ("Baseline" if SHARED_BASELINE and rate == 0
                         else f"{method.upper()} {rate:.1f}")
        for dataset, filename in [("validation", "validation_metrics.csv"),
                                  ("test", "test_metrics.csv")]:
            row = read_csv(folder / filename).iloc[0].to_dict()
            row["configuration"] = configuration
            row["source_file"] = str(folder / filename)
            frames[dataset].append(row)
    for dataset, rows in frames.items():
        if not rows:
            continue
        data = pd.DataFrame(rows).sort_values(["configuration", "seed"])
        target = Path(root) / "summary_no_cv"
        save_csv(data, target / f"results_{dataset}_per_seed.csv")
        summaries = []
        for (configuration, method, rate), group in data.groupby(
                ["configuration", "selection_method", "drop_rate"], sort=False):
            for metric in METRICS:
                if metric not in group:
                    continue
                values = pd.to_numeric(group[metric], errors="raise")
                summaries.append({
                    "configuration": configuration, "selection_method": method,
                    "drop_rate": rate, "metric": metric,
                    "n_seeds": len(group), "expected_seeds": len(TRAINING_SEEDS),
                    "configuration_complete": len(group) == len(TRAINING_SEEDS),
                    "valid_values": values.count(), "mean": values.mean(),
                    "std": values.std(ddof=1), "median": values.median(),
                    "min": values.min(), "max": values.max(),
                })
        summary = pd.DataFrame(summaries)
        save_csv(summary, target / f"summary_{dataset}_{len(TRAINING_SEEDS)}seeds.csv")
    return frames


def validate():
    if not MAIN_FILE.is_file():
        raise FileNotFoundError(f"Keep both files together. Missing: {MAIN_FILE}")
    if len(R_VALUES) != 5 or len(set(R_VALUES)) != 5:
        raise ValueError("R_VALUES must contain the five selected rates, without duplicates.")
    if any(r not in [i / 10 for i in range(10)] for r in R_VALUES):
        raise ValueError("Use rates between 0.0 and 0.9, in steps of 0.1.")
    if 0.0 not in R_VALUES:
        raise ValueError("Include r=0.0 as the baseline for this experiment.")
    if len(TRAINING_SEEDS) != 10 or len(set(TRAINING_SEEDS)) != 10:
        raise ValueError("Provide ten different seeds.")
    if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in TRAINING_SEEDS):
        raise ValueError("Seeds must be non-negative integers less than 2**32.")
    if not 0 < VALIDATION_FRACTION < 1:
        raise ValueError("VALIDATION_FRACTION must be between 0 and 1.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["both", "hard", "random"], default="both")
    parser.add_argument("--output", type=Path, default=RESULTS_FOLDER)
    parser.add_argument("--only-plan", action="store_true", help="Shows the plan without training.")
    args = parser.parse_args()
    validate()
    jobs = [j for j in full_plan()
            if args.method == "both" or j[0] == args.method
            or (SHARED_BASELINE and j[1] == 0)]
    root = args.output.resolve()
    print("Selected rates:", R_VALUES)
    print("Training seeds:", TRAINING_SEEDS)
    print("Fixed split seed:", SPLIT_SEED)
    print("No CV: one train/validation split and a separate test.")
    print("Shared baseline:", SHARED_BASELINE)
    print("Executions in this plan:", len(jobs))
    print("Results:", root)
    if args.only_plan:
        print(pd.DataFrame(jobs, columns=["method", "drop_rate", "seed"]).to_string(index=False))
        return
    trainer_hash = file_hash(MAIN_FILE)
    root.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(jobs, 1):
        method, rate, seed = job
        label = f"[{index}/{len(jobs)}] {method.upper()} r={rate:.1f}, seed={seed}"
        if is_completed(root, job, trainer_hash) is not None:
            print(label, "SKIPPED: training and testing already completed.", flush=True)
            continue
        print("\n" + label, "STARTING", flush=True)
        try:
            execute(root, job)
        except subprocess.CalledProcessError as error:
            consolidate(root, trainer_hash)
            print("The current execution was interrupted. Re-run this file: "
                  "completed ones are skipped and the incomplete one restarts from the first epoch.")
            raise SystemExit(error.returncode) from error
        if is_completed(root, job, trainer_hash) is None:
            raise RuntimeError("The trainer finished without registering complete training and testing.")
        consolidate(root, trainer_hash)
    frames = consolidate(root, trainer_hash)
    print("\nPLAN COMPLETED. Summaries:", root / "summary_no_cv")
    if frames["validation"]:
        view = pd.DataFrame(frames["validation"])
        print(view.groupby("configuration")["f1_macro"].agg(["count", "mean", "std", "median"]).to_string())


if __name__ == "__main__":
    main()
