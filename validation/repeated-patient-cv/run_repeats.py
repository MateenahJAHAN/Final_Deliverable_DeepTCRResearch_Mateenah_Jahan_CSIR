"""Run the declared patient-separated splits and checkpoint each complete fit."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.model_selection import StratifiedKFold

import run as runner

ROOT = Path(__file__).resolve().parent


def declared_jobs(plan):
    jobs = [(plan["primary_folds"], partition, seed)
            for partition in plan["patient_partition_seeds"] for seed in plan["training_seeds"]]
    jobs += [(n, plan["sensitivity_partition_seed"], plan["sensitivity_training_seed"])
             for n in plan["sensitivity_folds"]]
    return [(*job, fold) for job in jobs for fold in range(1, job[0] + 1)]


def folds_for(labels, k, partition_seed):
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=partition_seed)
                .split(np.zeros(len(labels)), labels))


def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--data-dir", type=Path, required=True)
    cmd.add_argument("--release-archive", type=Path, required=True)
    cmd.add_argument("--expected-csv-sha256", required=True)
    cmd.add_argument("--workers", type=int, default=2)
    cmd.add_argument("--worker", type=int, required=True)
    cmd.add_argument("--threads", type=int, default=5)
    args = cmd.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        cmd.error("worker must be between zero and workers minus one")
    args.device = "CPU"
    args.smoke = False

    cfg = json.loads((ROOT / "protocol.json").read_text())
    plan = json.loads((ROOT / "repeat_plan.json").read_text())
    worker = ROOT / "private" / ("worker-" + str(args.worker))
    worker.mkdir(parents=True, exist_ok=True)
    runner.HERE = worker
    runner.PRIVATE = worker
    csv = runner.verify_inputs(cfg, args)
    data, groups, labels = runner.read_cohort(csv, cfg)
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in (
        ROOT / "repeat_plan.json", ROOT / "protocol.json", ROOT / "run.py", ROOT / "run_repeats.py"
    ))).hexdigest()
    runtime = {
        "python": platform.python_version(), "tensorflow": runner.tf.__version__,
        "scikit_learn": runner.sklearn_version, "numpy": np.__version__,
        "intra_op_threads": args.threads, "host": platform.system() + " " + platform.machine(),
    }
    pending = declared_jobs(plan)[args.worker::args.workers]
    scores = ROOT / "private" / "checkpoint"
    scores.mkdir(parents=True, exist_ok=True)
    runner.log(f"Worker {args.worker}: verified {len(labels)} patients and {len(groups):,} eligible records; {len(pending)} complete fits assigned")
    model, encoders, codes = runner.encode(data, groups, labels, cfg)
    for k, partition_seed, training_seed, fold in pending:
        path = scores / f"k{k}-p{partition_seed}-t{training_seed}-f{fold}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["source_fingerprint"] != fingerprint or saved["runtime"] != runtime
                    or saved["input_digest"] != args.expected_csv_sha256):
                raise ValueError(f"Checkpoint parameters differ for {path.name}")
            continue
        train, test = folds_for(labels, k, partition_seed)[fold-1]
        runner.log(f"k={k}, partition seed {partition_seed}, training seed {training_seed}, fold {fold}")
        metrics, predictions = runner.fit_fold(model, encoders, codes, groups, labels, train, test, fold,
                                               {**cfg, "random_seed": training_seed}, args)
        runner.write_json(path, {
            "runtime": runtime, "source_fingerprint": fingerprint,
            "input_digest": args.expected_csv_sha256, "metrics": metrics, "scores": predictions,
        })
    runner.log(f"Worker {args.worker} completed all assigned fits")


if __name__ == "__main__":
    main()
