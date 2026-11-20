"""Independently verify the 57 fits and measure the matched composition baseline."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-csv-sha256", required=True)
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def verified_cohort(args, config):
    source = args.data_dir / "deeptcr_trb_ready.csv"
    if sha256(source.read_bytes()).hexdigest() != args.expected_csv_sha256:
        raise ValueError("Unexpected cohort digest")
    source_labels = np.load(args.data_dir / "y_labels.npy", allow_pickle=False)
    records = pd.read_csv(source, usecols=["patient_id", "response_binary", "aminoAcid"])
    assert len(records) == config["data_rows_before_max_length"]
    assert records[["patient_id", "response_binary", "aminoAcid"]].notna().all().all()
    assert np.array_equal(records.response_binary, source_labels)
    records = records.loc[records.aminoAcid.str.len().le(config["max_length"])]
    patients = records.groupby("patient_id", sort=True)["response_binary"]
    assert patients.nunique().eq(1).all()
    outcomes = patients.first()
    counts = patients.size().to_numpy()
    groups = outcomes.index.get_indexer(records.patient_id)
    assert groups.min() == 0
    assert len(outcomes) == config["patients"] and outcomes.sum() == config["positive_patients"]
    assert len(records) == config["data_rows_after_max_length"] == counts.sum()
    return records.aminoAcid.to_numpy(), groups, outcomes.to_numpy(), counts


def composition(sequences, groups, record_counts):
    patients = len(record_counts)
    monomers = np.zeros((patients, 20))
    dimers = np.zeros((patients, 400))
    lengths = np.zeros((patients, 2))
    index = {aa: i for i, aa in enumerate(ALPHABET)}
    for group, seq in zip(groups, sequences):
        codes = [index.get(aa, -1) for aa in seq]
        for code in codes:
            if code >= 0:
                monomers[group, code] += 1
        for left, right in zip(codes, codes[1:]):
            if left >= 0 and right >= 0:
                dimers[group, left * 20 + right] += 1
        lengths[group] += len(seq), len(seq)**2
    lengths /= record_counts[:, None]
    lengths[:, 1] = np.sqrt(np.maximum(0, lengths[:, 1] - lengths[:, 0]**2))
    array = np.column_stack((lengths, monomers / monomers.sum(1)[:, None],
                            dimers / dimers.sum(1)[:, None]))
    assert array.shape == (patients, 422) and np.isfinite(array).all()
    return array


def make_folds(outcomes, folds, seed):
    sets = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                .split(np.zeros(len(outcomes)), outcomes))
    held = np.concatenate([test for _, test in sets])
    assert np.array_equal(np.sort(held), np.arange(len(outcomes)))
    assert all(len(np.intersect1d(train, test)) == 0 for train, test in sets)
    assert all(len(np.unique(outcomes[test])) == 2 for _, test in sets)
    return sets


def evaluate(sets, outcomes, scores):
    assert np.isfinite(scores).all()
    wins, pairs = 0.0, 0
    for _, test in sets:
        y = outcomes[test]
        s = scores[test]
        comparison = np.subtract.outer(s[y == 1], s[y == 0])
        wins += float((comparison > 0).sum() + .5 * (comparison == 0).sum())
        pairs += int(comparison.size)
    return {
        "within_fold_auroc": wins / pairs,
        "concordant_pair_equivalents": wins,
        "pairs": pairs,
        "pooled_oof_auroc": float(roc_auc_score(outcomes, scores)),
        "pooled_average_precision": float(average_precision_score(outcomes, scores)),
    }


def measure_baseline(features, outcomes, sets, seed):
    scores = np.empty(len(outcomes))
    for train, test in sets:
        learner = make_pipeline(StandardScaler(),
                                LogisticRegression(C=.1, solver="liblinear", random_state=seed))
        learner.fit(features[train], outcomes[train])
        scores[test] = learner.predict_proba(features[test])[:, 1]
    return evaluate(sets, outcomes, scores)


def load_verified_folds(checkpoints, scores, sets, outcomes, counts, config):
    fold_results = []
    for fold, (path, (train, test)) in enumerate(zip(checkpoints, sets), start=1):
        result = json.loads(path.read_text())
        metrics = result["metrics"]
        assert metrics["fold"] == fold
        assert metrics["train_patients"] == len(train) and metrics["test_patients"] == len(test)
        assert metrics["test_positive_patients"] == outcomes[test].sum()
        assert metrics["train_rows"] == counts[train].sum()
        assert metrics["test_rows"] == counts[test].sum()
        assert [e["epoch"] for e in metrics["epochs"]] == list(range(1, config["epochs"] + 1))
        assert all(np.isfinite(e["batch_mean_loss"]) for e in metrics["epochs"])
        assert np.isfinite(result["scores"]).all() and len(result["scores"]) == len(test)
        assert np.isnan(scores[test]).all()
        scores[test] = result["scores"]
        assert np.isclose(roc_auc_score(outcomes[test], scores[test]), metrics["patient_auroc"])
        fold_results.append(metrics)
    return fold_results


def describe(values):
    return {"median": float(np.median(values)), "minimum": float(min(values)),
            "maximum": float(max(values))}


def study_summary(records, primary_folds):
    primary = [x for x in records if x["folds"] == primary_folds]
    models = [x["deeptcr"]["within_fold_auroc"] for x in primary]
    by_partition = {x["partition_seed"]: x["composition"]["within_fold_auroc"] for x in primary}
    gaps = [x["deeptcr"]["within_fold_auroc"] - x["composition"]["within_fold_auroc"] for x in primary]
    paired_seeds = [[x["deeptcr"]["within_fold_auroc"] for x in primary
                     if x["partition_seed"] == partition] for partition in by_partition]
    assert all(len(pair) == 2 for pair in paired_seeds)
    return {
        "primary_evaluations": len(primary),
        "primary_partitions": len(by_partition),
        "deeptcr_within_fold": describe(models),
        "baseline_within_fold": describe(list(by_partition.values())),
        "deeptcr_pooled_oof": describe([x["deeptcr"]["pooled_oof_auroc"] for x in primary]),
        "paired_deeptcr_minus_baseline": describe(gaps),
        "absolute_training_seed_gap": describe([abs(a - b) for a,b in paired_seeds]),
        "deeptcr_above_baseline": int(sum(x > 1e-12 for x in gaps)),
        "baseline_above_deeptcr": int(sum(x < -1e-12 for x in gaps)),
        "ties": int(sum(abs(x) <= 1e-12 for x in gaps)),
    }


def main():
    args = arguments()
    config = json.loads((ROOT / "protocol.json").read_text())
    plan = json.loads((ROOT / "repeat_plan.json").read_text())
    sequences, groups, outcomes, counts = verified_cohort(args, config)
    features = composition(sequences, groups, counts)
    partitions = [(plan["primary_folds"], seed) for seed in plan["patient_partition_seeds"]]
    partitions.extend((k, plan["sensitivity_partition_seed"]) for k in plan["sensitivity_folds"])
    split = {(k, seed): make_folds(outcomes, k, seed) for k, seed in partitions}
    baseline = {key: measure_baseline(features, outcomes, sets, key[1])
                for key, sets in split.items()}
    if args.baseline_only:
        (ROOT / "baseline.checked.json").write_text(json.dumps([
            {"folds": k, "partition_seed": seed, **baseline[k, seed]} for k, seed in partitions
        ], indent=2) + "\n")
        print(f"Baseline complete on {len(partitions)} partitions of all {len(outcomes)} patients")
        return

    jobs = [(k, p, t) for k,p in partitions if k == plan["primary_folds"] for t in plan["training_seeds"]]
    jobs += [(k, p, plan["sensitivity_training_seed"]) for k,p in partitions if k in plan["sensitivity_folds"]]
    folder = ROOT / "private" / "checkpoint"
    expected = {f"k{k}-p{p}-t{t}-f{f}.json" for k,p,t in jobs for f in range(1,k + 1)}
    present = {path.name for path in folder.glob("*.json")}
    if expected != present:
        raise ValueError(f"Expected {len(expected)} checkpoints, found {len(present)}")
    records = []
    runtime_signatures = set()
    checksums = set()
    for path in sorted(folder.glob("*.json")):
        saved = json.loads(path.read_text())
        runtime_signatures.add(json.dumps(saved["runtime"], sort_keys=True))
        checksums.add(saved["source_fingerprint"])
        assert saved["input_digest"] == args.expected_csv_sha256
    fingerprint = sha256(b"".join((ROOT / file).read_bytes() for file in (
        "repeat_plan.json", "protocol.json", "run.py", "run_repeats.py"
    ))).hexdigest()
    assert len(runtime_signatures) == 1 and checksums == {fingerprint}
    for k, part, seed in jobs:
        sets = split[k,part]
        scores = np.full(len(outcomes), np.nan)
        paths = [folder / f"k{k}-p{part}-t{seed}-f{f}.json" for f in range(1,k + 1)]
        fold_results = load_verified_folds(paths, scores, sets, outcomes, counts, config)
        arch = evaluate(sets, outcomes, scores)
        entry = {"folds": k, "partition_seed": part, "training_seed": seed,
                 "deeptcr": arch, "composition": baseline[k,part], "fold_metrics": fold_results}
        records.append(entry)
    payload = {
        "patients": int(len(outcomes)), "responders": int(outcomes.sum()),
        "eligible_records": int(counts.sum()), "completed_fits": len(expected),
        "runtime": json.loads(next(iter(runtime_signatures))),
        "summary": study_summary(records, plan["primary_folds"]),
        "runs": records,
    }
    (ROOT / "repeat_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (ROOT / "repeat_results.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["folds", "partition_seed", "training_seed", "deeptcr_within_fold_auroc",
                         "composition_within_fold_auroc", "deeptcr_pooled_oof_auroc", "comparison_pairs"])
        for row in records:
            d, b = row["deeptcr"], row["composition"]
            writer.writerow([row["folds"], row["partition_seed"], row["training_seed"],
                             d["within_fold_auroc"], b["within_fold_auroc"], d["pooled_oof_auroc"], d["pairs"]])
    print(f"Verified all {len(expected)} full fits and {len(jobs)} full patient OOF evaluations.")


if __name__ == "__main__":
    main()
