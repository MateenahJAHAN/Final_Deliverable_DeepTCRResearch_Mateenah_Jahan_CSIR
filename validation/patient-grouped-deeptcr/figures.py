#!/usr/bin/env python3
"""Render the finished DeepTCR aggregate results without patient-level data."""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))

import matplotlib.pyplot as plt
import numpy as np


def render(results):
    fig, (folds, losses) = plt.subplots(
        1, 2, figsize=(10, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.05, 1.2]}
    )
    fig.suptitle("DeepTCR on held-out CAR-T patients", x=0.03, ha="left", weight="bold", fontsize=17)
    primary = results["within_fold_patient_auroc"]
    ci = results["within_fold_patient_auroc_bootstrap_ci95"]
    folds.axvline(0.5, color="#949da6", linewidth=1.2, linestyle="--")
    folds.errorbar(
        primary, 6, xerr=[[max(0, primary - ci[0])], [max(0, ci[1] - primary)]],
        fmt="D", color="#0b6d62", elinewidth=2.5, capsize=5,
    )
    for fold in results["folds"]:
        i = fold["fold"]
        folds.scatter(fold["patient_auroc"], 6 - i, color="#246080", s=39)
        records = fold["epochs"]
        losses.plot([e["epoch"] for e in records], [e["batch_mean_loss"] for e in records],
                    marker="o", markersize=2.3, linewidth=1.4, label=f"Fold {i}")
    control = results["shuffled_training_control"]
    folds.scatter(control["patient_auroc"], 0, marker="s", color="#aa5634", s=38)
    folds.set_yticks(
        range(7),
        [f"Shuffled train (fold {control['fold']})"]
        + [f"Fold {f['fold']} (n={f['test_patients']})" for f in reversed(results["folds"])]
        + ["Within-fold aggregate"],
    )
    folds.set_xlim(-0.045, 1.045)
    folds.set_xticks(np.linspace(0, 1, 6))
    folds.set_xlabel("Patient AUROC")
    folds.set_title("Held-out discrimination", loc="left", fontsize=11, weight="bold")
    records = control["epochs"]
    losses.plot([e["epoch"] for e in records], [e["batch_mean_loss"] for e in records],
                linewidth=1.8, color="#aa5634", linestyle="--", label="Shuffled labels")
    losses.legend(frameon=False, fontsize=8, ncol=2)
    losses.set_xticks(range(1, 11))
    losses.set_xlabel("Training epoch")
    losses.set_ylabel("Mean minibatch loss")
    losses.set_title("Training loss", loc="left", fontsize=11, weight="bold")
    for ax in (folds, losses):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "deeptcr_patient_run.png", dpi=180)
    svg = HERE / "deeptcr_patient_run.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    render(json.loads((HERE / "aggregate_results.json").read_text()))
