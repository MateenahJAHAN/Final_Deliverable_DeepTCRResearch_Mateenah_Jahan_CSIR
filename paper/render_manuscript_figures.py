"""Render the revised report panels from public run aggregates and authorized cohort data."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED, TEAL, GRAY = "#23679a", "#b74443", "#227f71", "#52667a"
EXPECTED_INPUT = "80d39ff4d5eb95a47555d1d609825e6cedb366bd71b320bae6f3d619743296de"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def save(fig, directory, name):
    fig.savefig(directory / (name + ".png"), dpi=240, facecolor="white")
    plt.close(fig)


def diagram(directory, name, steps):
    fig, ax = plt.subplots(figsize=(11, 2), layout="constrained")
    ax.set(xlim=(-0.52, len(steps) - 0.48), ylim=(0, 1))
    ax.axis("off")
    for index, (title, detail) in enumerate(steps):
        color = [BLUE, TEAL, GRAY, RED][index]
        ax.add_patch(plt.Rectangle((index - 0.43, 0.16), 0.86, 0.68,
                                   facecolor=color, edgecolor="none"))
        ax.text(index, 0.65, title, color="white", ha="center",
                va="center", fontsize=12, fontweight="bold")
        ax.text(index, 0.43, detail, color="white", ha="center",
                va="center", fontsize=10, linespacing=1.35)
        if index < len(steps) - 1:
            ax.annotate("", xy=(index + 0.57, 0.5), xytext=(index + 0.43, 0.5),
                        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.4})
    save(fig, directory, name)


def model_panels(directory, repeated, earlier):
    folds = [(run["folds"], fold) for run in repeated["runs"] for fold in run["fold_metrics"]]
    assert len(folds) == repeated["completed_fits"] == 57
    logged_epochs = np.array([[epoch["batch_mean_loss"] for epoch in fold["epochs"]]
                              for _, fold in folds])
    assert logged_epochs.shape == (57, 10) and np.isfinite(logged_epochs).all()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    x = np.arange(1, 11)
    for losses in logged_epochs:
        a.plot(x, losses, color=BLUE, alpha=0.16, lw=0.8)
    a.plot(x, np.median(logged_epochs, axis=0), color=BLUE, lw=2, label="Median")
    a.set(xlabel="Epoch", ylabel="Mean minibatch loss", title="A  All 57 training fits")
    a.legend(frameon=False)
    for control, label, color in [(earlier["folds"][0], "Recorded labels", BLUE),
                                  (earlier["shuffled_training_control"], "Shuffled labels", RED)]:
        b.plot(x, [epoch["batch_mean_loss"] for epoch in control["epochs"]],
               marker="o", ms=3, color=color, label=label)
    b.set(xlabel="Epoch", ylabel="Mean minibatch loss", title="B  Earlier fold-1 control")
    b.legend(frameon=False)
    save(fig, directory, "training-loss")

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for position, count in enumerate((3, 4, 5), 1):
        subset = [fold for total, fold in folds if total == count]
        positions = np.linspace(position - 0.16, position + 0.16, len(subset))
        a.scatter(positions, [sum(epoch["seconds"] for epoch in fold["epochs"])
                              for fold in subset], color=BLUE, alpha=0.64, s=20)
        b.scatter(positions, [fold["inference_seconds"] for fold in subset],
                  color=TEAL, alpha=0.64, s=20)
    for axis, label, title in [(a, "Ten-epoch training time (s)", "A  Training"),
                                (b, "Held-out inference time (s)", "B  Inference")]:
        axis.set(xticks=[1, 2, 3], xticklabels=["3 folds", "4 folds", "5 folds"],
                 xlim=(0.6, 3.4), ylabel=label, title=title)
    save(fig, directory, "fit-runtime")


def sequence_panels(directory, cohort):
    digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    if digest != EXPECTED_INPUT:
        raise ValueError("The cohort file does not match the verified input.")
    data = pd.read_csv(cohort, usecols=["patient_id", "response_binary", "aminoAcid"])
    data["length"] = data["aminoAcid"].str.len()
    data = data[data["length"] <= 40]
    assert len(data) == 239634 and data.patient_id.nunique() == 34
    groups = [(label, group) for label, group in data.groupby("response_binary")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    frequencies = []
    for (label, group), color, position in zip(groups, (RED, BLUE), (0, 1)):
        name = "Responders" if label else "Nonresponders"
        patient_length = pd.crosstab(group["patient_id"], group["length"])
        lengths = patient_length.columns.to_numpy()
        normalized = patient_length.div(patient_length.sum(axis=1), axis=0)
        a.plot(lengths, normalized.mean(axis=0) * 100, color=color, label=name)
        means = group.groupby("patient_id")["length"].mean().to_numpy()
        b.scatter(position + np.linspace(-0.13, 0.13, len(means)), np.sort(means),
                  color=color, alpha=0.8, s=24)
        b.hlines(np.median(means), position - 0.23, position + 0.23, color=color, lw=2)
        per_patient = []
        for _, sample in group.groupby("patient_id"):
            counts = Counter("".join(sample["aminoAcid"]))
            residues = sum(counts.values())
            per_patient.append([100 * counts[aa] / residues for aa in AMINO_ACIDS])
        frequencies.append(np.asarray(per_patient))
    a.set(xlabel="CDR3 length (residues)", ylabel="Mean within-patient share (%)",
          title="A  Repertoire length distribution")
    a.legend(frameon=False)
    b.set(xticks=[0, 1], xticklabels=["Nonresponders", "Responders"],
          xlim=(-0.46, 1.46), ylabel="Mean CDR3 length (residues)",
          title="B  Individual patient means")
    save(fig, directory, "response-length")

    fig, ax = plt.subplots(figsize=(10, 3.3), layout="constrained")
    x = np.arange(len(AMINO_ACIDS))
    for position, array, color, title in zip((-0.19, 0.19), frequencies,
                                             (RED, BLUE), ("Nonresponders", "Responders")):
        lower, median, upper = np.percentile(array, [25, 50, 75], axis=0)
        ax.errorbar(x + position, median, [median - lower, upper - median], color=color,
                    fmt="o", markersize=3, capsize=2, label=title)
    ax.set(xticks=x, xticklabels=AMINO_ACIDS, xlabel="Amino acid",
           ylabel="Within-patient residue share (%)")
    ax.legend(frameon=False, ncol=2)
    save(fig, directory, "sequence-composition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    directory = args.output or root / "figures/manuscript"
    directory.mkdir(parents=True, exist_ok=True)
    repeated = json.loads((root / "validation/repeated-patient-cv/repeat_results.json").read_text())
    earlier = json.loads((root / "validation/correction-rerun/aggregate_results.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": "#162d44"})
    diagram(directory, "pipeline", [
        ("Archived repertoire", "239,637 records\n34 patient labels"),
        ("Length filter", "239,634 eligible records\nAll 34 patients retained"),
        ("Partition by patient", "Stratify the response labels\nKeep each repertoire intact"),
        ("Train and evaluate", "Fit on training patients\nScore held-out patients"),
    ])
    diagram(directory, "architecture", [
        ("CDR3 sequence", "DeepTCR sequence encoder\nLarge convolutional network"),
        ("Join V and J features", "Training-fold gene vocabularies\nUnknown category for unseen genes"),
        ("Sequence classifier", "Two-class output\nResponder-class score per record"),
        ("Patient score", "Mean of sequence scores\nOne result per held-out patient"),
    ])
    model_panels(directory, repeated, earlier)
    sequence_panels(directory, args.cohort)


if __name__ == "__main__":
    main()
