#!/usr/bin/env python3
"""Train the package's actual DeepTCR_SS graph with preassigned patient folds."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / ".private"
VENDOR = HERE / "vendor/deeptcr-2.1.29"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".numba-cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from DeepTCR.DeepTCR import DeepTCR_SS
from DeepTCR.functions.utils_s import Run_Graph_SS

INPUTS = (
    "X_Seq_alpha", "X_Seq_beta", "v_beta_num", "d_beta_num",
    "j_beta_num", "v_alpha_num", "j_alpha_num", "hla_data_seq_num",
)
GENES = ("v_beta", "j_beta", "v_alpha", "j_alpha")


class SequentialPool:
    """Use package encoding without forking an active TensorFlow process."""

    def starmap(self, process, iterable):
        return [process(*args) for args in iterable]


def sha256(path):
    with path.open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(config, args):
    csv = args.data_dir / "deeptcr_trb_ready.csv"
    if args.expected_csv_sha256 and sha256(csv) != args.expected_csv_sha256:
        raise ValueError("Archived processed input does not match the committed digest")
    sdist = args.release_archive
    if sha256(sdist) != config["deeptcr_sdist_sha256"]:
        raise ValueError("DeepTCR source archive failed digest verification")
    with tarfile.open(sdist) as archive:
        for path in (VENDOR / "DeepTCR").rglob("*.py"):
            relative = str(path.relative_to(VENDOR))
            upstream = archive.extractfile(f"deeptcr-2.1.29/{relative}").read()
            if path.read_bytes() != upstream:
                raise ValueError(f"Modified upstream Python module: {relative}")
    return csv


def read_cohort(csv, config):
    cols = ["aminoAcid", "vGeneName", "jGeneName", "patient_id", "response_binary"]
    data = pd.read_csv(csv, usecols=cols)
    archive_labels = np.load(csv.with_name("y_labels.npy"), allow_pickle=False)
    assert len(data) == config["data_rows_before_max_length"]
    assert np.array_equal(data.response_binary.to_numpy(), archive_labels)
    data = data.loc[data.aminoAcid.str.len() <= config["max_length"]].reset_index(drop=True)
    groups, _ = pd.factorize(data.patient_id, sort=True)
    labels = np.array([
        np.unique(data.response_binary[groups == i]).item()
        for i in range(config["patients"])
    ])
    assert data[cols].notna().all().all()
    assert len(data) == config["data_rows_after_max_length"]
    assert len(np.unique(groups)) == len(labels) == config["patients"]
    assert set(labels) == {0, 1} and labels.sum() == config["positive_patients"]
    return data.drop(columns="patient_id"), groups, labels


def write_json(path, value):
    # Atomically replace aggregate results or the private resume journal.
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def encode(data, groups, labels, config):
    model = DeepTCR_SS(str(PRIVATE / "package_output"), max_length=config["max_length"], tf_verbosity=2)
    model.Load_Data(
        beta_sequences=data.aminoAcid.to_numpy(),
        v_beta=data.vGeneName.to_numpy(),
        j_beta=data.jGeneName.to_numpy(),
        alpha_sequences=np.full(len(data), "AAA"),
        v_alpha=np.full(len(data), "TRAV1-1"),
        j_alpha=np.full(len(data), "TRAJ1"),
        class_labels=labels[groups].astype(str),
        sample_labels=groups.astype(str),
        p=SequentialPool(),
    )
    assert not model.regression and list(model.lb.classes_) == ["0", "1"]
    assert np.array_equal(model.Y[:, 1], labels[groups])
    model.var_dict = {name: i for i, name in enumerate(INPUTS)}
    encoders = {gene: getattr(model, "lb_" + gene) for gene in GENES}
    global_codes = {gene: getattr(model, gene + "_num").copy() for gene in GENES}
    return model, encoders, global_codes


def apply_fold_vocab(model, train_rows, encoders, global_codes):
    unknown_counts = {}
    for gene in GENES:
        global_encoder = encoders[gene]
        codes = global_codes[gene]
        seen = np.unique(codes[train_rows])
        keep = np.union1d(seen, [0])
        new_codes = np.zeros(len(global_encoder.classes_), dtype=np.int64)
        new_codes[keep] = np.arange(len(keep))
        fold_encoder = type(global_encoder)()
        fold_encoder.classes_ = global_encoder.classes_[keep]
        setattr(model, "lb_" + gene, fold_encoder)
        setattr(model, gene + "_num", new_codes[codes])
        unknown_counts[gene] = int(((new_codes[codes] == 0) & (codes != 0)).sum())
    return unknown_counts


def fit_fold(model, encoders, codes, groups, labels, train, test, fold, config, args, control=False):
    rng = np.random.default_rng(config["random_seed"] + 2000 + fold)
    fold_labels = labels.copy()
    if control:
        fold_labels[train] = rng.permutation(fold_labels[train])
    train_rows = np.flatnonzero(np.isin(groups, train))
    test_rows = np.flatnonzero(np.isin(groups, test))
    assert not np.intersect1d(groups[train_rows], groups[test_rows]).size
    assert len(train_rows) + len(test_rows) == len(groups)
    if args.smoke:
        train_rows = train_rows[: config["batch_size"] * 2]
    unknown = apply_fold_vocab(model, train_rows, encoders, codes)
    train_vars = [getattr(model, name)[train_rows] for name in INPUTS]
    train_vars.append(np.eye(2)[fold_labels[groups[train_rows]]])
    # Feed neutral placeholders during inference so outer labels enter evaluation only.
    eval_vars = [getattr(model, name)[test_rows] for name in INPUTS]
    eval_vars.append(np.zeros((len(test_rows), 2), dtype=np.float64))

    model.device = "/device:" + args.device + ":0"
    graph_seed = config["random_seed"] + fold
    model._build(
        **config["graph"],
        batch_size=config["batch_size"],
        epochs_min=config["epochs"],
        graph_seed=graph_seed,
        suppress_output=True,
    )
    graph = model.graph_model
    session_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=args.threads,
        inter_op_parallelism_threads=2,
    )
    session_config.gpu_options.allow_growth = True
    epoch_records = []
    label = "shuffled-label control" if control else "true labels"
    with graph.as_default():
        setup = tf.compat.v1.global_variables_initializer()
        params = sum(v.shape.num_elements() for v in tf.compat.v1.trainable_variables())
    log(f"Fold {fold} ({label}), {len(train_rows):,} train rows, {len(test_rows):,} held-out rows, {params:,} trainable parameters")
    with tf.compat.v1.Session(graph=graph, config=session_config) as session:
        session.run(setup)
        for epoch in range(1, 2 if args.smoke else config["epochs"] + 1):
            np.random.seed(graph_seed + 10_000 * epoch)
            stamp = time.monotonic()
            loss, accuracy, _, _ = Run_Graph_SS(
                train_vars, session, model, model.GO, config["batch_size"],
                random=True, train=True, drop_out_rate=config["graph"]["drop_out_rate"],
            )
            if not np.isfinite(loss):
                raise RuntimeError("Training loss diverged")
            duration = round(time.monotonic() - stamp, 3)
            epoch_records.append(
                {"epoch": epoch, "seconds": duration, "batch_mean_loss": float(loss), "batch_mean_accuracy": float(accuracy)}
            )
            log(f"Fold {fold} epoch {epoch}, loss {loss:.5f}, train accuracy {accuracy:.4f}, {duration:.1f} seconds")
            if not args.smoke:
                write_json(HERE / "progress.json", {"fold": fold, "training": label, "epochs": epoch_records})
        if args.smoke:
            return {"fold": fold, "train_rows": len(train_rows), "epochs": epoch_records}, None
        inference_start = time.monotonic()
        _, _, predicted, _ = Run_Graph_SS(
            eval_vars, session, model, model.GO, config["batch_size"], random=False, train=False,
        )

    summed = np.bincount(groups[test_rows], weights=predicted[:, 1], minlength=len(labels))
    counts = np.bincount(groups[test_rows], minlength=len(labels))
    scores = summed[test] / counts[test]
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    metrics = {
        "fold": fold,
        "train_patients": len(train),
        "test_patients": len(test),
        "test_positive_patients": int(labels[test].sum()),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "trainable_parameters": params,
        "held_out_rows_with_unseen_genes": unknown,
        "patient_auroc": float(roc_auc_score(labels[test], scores)),
        "inference_seconds": round(time.monotonic() - inference_start, 3),
        "epochs": epoch_records,
    }
    if control:
        metrics["train_patient_labels_changed"] = int(np.sum(fold_labels[train] != labels[train]))
    log(f"Fold {fold} ({label}) finished. Held-out patient AUROC {metrics['patient_auroc']:.4f}")
    return metrics, scores.tolist()


def within_fold_wins(labels, scores, parts):
    wins = 0.0
    pairs = 0
    for part in parts:
        idx = np.asarray(part)
        pos = scores[idx[labels[idx] == 1]]
        neg = scores[idx[labels[idx] == 0]]
        wins += float(np.sum(pos[:, None] > neg) + 0.5 * np.sum(pos[:, None] == neg))
        pairs += len(pos) * len(neg)
    return wins, pairs


def summarize(labels, splits, journal, config):
    scores = np.empty(len(labels))
    parts = [test for _, test in splits]
    for i, part in enumerate(parts, start=1):
        scores[part] = journal["fit"][str(i)]["scores"]
    wins, pairs = within_fold_wins(labels, scores, parts)
    primary = wins / pairs
    rng = np.random.default_rng(config["random_seed"] + 3000)
    boots = []
    fold_label_groups = [[part[labels[part] == value] for value in (0, 1)] for part in parts]
    for _ in range(config["bootstrap_replicates"]):
        drawn = [np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
                 for groups in fold_label_groups]
        boot_wins, _ = within_fold_wins(labels, scores, drawn)
        boots.append(boot_wins / pairs)
    permuted_at_least_as_high = 0
    for _ in range(config["prediction_permutations"]):
        permuted = labels.copy()
        for part in parts:
            permuted[part] = rng.permutation(labels[part])
        perm_wins, _ = within_fold_wins(permuted, scores, parts)
        permuted_at_least_as_high += int(perm_wins >= wins)

    results = {
        "status": "complete",
        "model": "Official DeepTCR_SS 2.1.29, large, classification",
        "runtime": journal["runtime"],
        "patients": len(labels),
        "positive_patients": int(labels.sum()),
        "retained_rows": int(sum(item["metrics"]["test_rows"] for item in journal["fit"].values())),
        "within_fold_patient_auroc": primary,
        "patient_pairs_concordant_ties_half": wins,
        "patient_pairs": pairs,
        "within_fold_patient_auroc_bootstrap_ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
        "pooled_oof_patient_auroc": float(roc_auc_score(labels, scores)),
        "pooled_oof_patient_average_precision": float(average_precision_score(labels, scores)),
        "correct_at_threshold_0_5": int(np.sum((scores >= config["accuracy_threshold"]) == labels)),
        "fixed_score_shuffles_at_least_as_high": permuted_at_least_as_high,
        "fixed_score_shuffle_draws": config["prediction_permutations"],
        "folds": [journal["fit"][str(i + 1)]["metrics"] for i in range(len(splits))],
        "shuffled_training_control": journal["shuffled_training_control"],
    }
    assert results["retained_rows"] == config["data_rows_after_max_length"]
    return results


def run(args):
    config_path = HERE / "protocol.json"
    config = json.loads(config_path.read_text())
    private = PRIVATE / "private_journal.json"
    csv = verify_inputs(config, args)
    runtime = {
        "tensorflow": tf.__version__,
        "scikit_learn": sklearn_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host": platform.system() + " " + platform.machine(),
        "device": args.device,
        "threads": args.threads,
        "visible_devices": [d.device_type for d in tf.config.list_physical_devices()],
    }
    if args.device not in runtime["visible_devices"]:
        raise RuntimeError(f"TensorFlow does not see the requested {args.device}")
    log(f"Runtime {json.dumps(runtime)}")
    data, groups, labels = read_cohort(csv, config)
    split = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=config["random_seed"])
    splits = list(split.split(np.zeros(len(labels)), labels))
    assert np.array_equal(np.sort(np.concatenate([test for _, test in splits])), np.arange(len(labels)))
    log(f"Verified {len(data):,} eligible rows, {len(labels)} patients, {int(labels.sum())} responders")
    model, encoders, codes = encode(data, groups, labels, config)
    if args.smoke:
        fit_fold(model, encoders, codes, groups, labels, *splits[0], 1, config, args)
        return

    fingerprint = {"config": sha256(config_path), "code": sha256(HERE / "run.py"), "runtime": runtime, "data_sha256": sha256(csv)}
    journal = {"fingerprint": fingerprint, "runtime": runtime, "fit": {}}
    if private.exists():
        journal = json.loads(private.read_text())
        if journal["fingerprint"] != fingerprint:
            raise ValueError("Saved folds use a different protocol, runner, or runtime")
    for fold, (train, test) in enumerate(splits, start=1):
        if str(fold) in journal["fit"]:
            log(f"Resuming with verified patient fold {fold} already complete")
            continue
        metrics, scores = fit_fold(model, encoders, codes, groups, labels, train, test, fold, config, args)
        journal["fit"][str(fold)] = {"metrics": metrics, "scores": scores}
        write_json(private, journal)
        write_json(HERE / "completed_folds.json", [item["metrics"] for item in journal["fit"].values()])
    if "shuffled_training_control" not in journal:
        fold = config["shuffled_training_control_fold"]
        train, test = splits[fold - 1]
        metrics, _ = fit_fold(model, encoders, codes, groups, labels, train, test, fold, config, args, control=True)
        journal["shuffled_training_control"] = metrics
        write_json(private, journal)
    write_json(HERE / "aggregate_results.json", summarize(labels, splits, journal, config))
    log("Full patient-grouped DeepTCR results written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Authorized CSV and label array directory")
    parser.add_argument("--release-archive", type=Path, required=True, help="Official DeepTCR 2.1.29 source archive")
    parser.add_argument("--expected-csv-sha256", help="Optional trusted archive digest for the supplied CSV")
    parser.add_argument("--smoke", action="store_true", help="Train two batches and never score test data")
    parser.add_argument("--device", default="CPU", choices=("CPU", "GPU"))
    parser.add_argument("--threads", default=10, type=int)
    args = parser.parse_args()
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    lock_name = f"smoke-{args.device}.lock" if args.smoke else "run.lock"
    with (PRIVATE / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
