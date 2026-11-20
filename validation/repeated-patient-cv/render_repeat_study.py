"""Render all predeclared comparisons from complete, independently verified runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def band(stats):
    return f'{stats["median"]:.3f} ({stats["minimum"]:.3f}–{stats["maximum"]:.3f})'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=ROOT / "repeated-patient-auroc.png")
    args = parser.parse_args()
    result = json.loads((ROOT / "repeat_results.json").read_text())
    assert result["completed_fits"] == 57
    primary = [run for run in result["runs"] if run["folds"] == 5]
    sensitivity = [run for run in result["runs"] if run["folds"] != 5]
    split_seeds = sorted({run["partition_seed"] for run in primary})
    seed_orders = sorted({run["training_seed"] for run in primary})
    assert len(primary) == 10 and len(split_seeds) == 5 and len(sensitivity) == 2
    stats = result["summary"]
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'text.color': '#152b40', 'axes.labelcolor': '#24384c',
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.45, 1]})
    c1, c2, cb = "#18766d", "#3c6c93", "#a65332"
    for axis in (left, right):
        axis.axvline(.5, ls="--", lw=1, color="#8a99a6", zorder=0)
        axis.set(xlim=(-.03, 1.03), xlabel="Within-fold patient AUROC", xticks=[0, .25, .5, .75, 1])
    for idx, partition in enumerate(split_seeds):
        row = [run for run in primary if run["partition_seed"] == partition]
        row = sorted(row, key=lambda run: run["training_seed"])
        y = len(split_seeds) - idx
        for i, run in enumerate(row):
            left.scatter(run["deeptcr"]["within_fold_auroc"], y + [.18, 0][i], color=[c1,c2][i],
                         marker=["o","s"][i], s=37, label=f"DeepTCR seed {i+1}" if idx == 0 else None)
        left.scatter(row[0]["composition"]["within_fold_auroc"], y - .18, color=cb, marker="D", s=30,
                     label="Composition baseline" if idx == 0 else None)
    left.set_yticks(range(1,6), [f"Split {i}" for i in reversed(range(1,6))])
    left.set_ylim(.55,5.55)
    left.legend(frameon=False, fontsize=8, bbox_to_anchor=(0, 1.02, 1, .15), loc="lower left", ncol=3)
    for idx, run in enumerate(sorted(sensitivity,key=lambda run: run["folds"])):
        y = 2 - idx
        right.scatter(run["deeptcr"]["within_fold_auroc"], y + .1, color=c1, s=37, marker="o")
        right.scatter(run["composition"]["within_fold_auroc"], y - .1, color=cb, s=30, marker="D")
    right.set_yticks([1,2],["Four folds","Three folds"])
    right.set_ylim(.55,2.45)
    fig.savefig(args.figure, dpi=225, facecolor="white")
    plt.close(fig)

    content = [
        "# Repeated patient-held-out evaluation", "",
        f"The study completed **{result['completed_fits']} ten-epoch DeepTCR fits** on "
        f"{result['patients']} patients and {result['eligible_records']:,} eligible records. "
        "Every evaluation scored each patient exactly once. Five patient partitions crossed "
        "with two training seeds produced ten five-fold evaluations. One three-fold and one "
        "four-fold evaluation checked sensitivity to fold count.", "",
        "| Primary analysis | Median (minimum–maximum) |", "| --- | ---: |",
        f"| DeepTCR within-fold AUROC, ten evaluations | {band(stats['deeptcr_within_fold'])} |",
        f"| Composition within-fold AUROC, five partitions | {band(stats['baseline_within_fold'])} |",
        f"| DeepTCR minus baseline on matched partitions | {band(stats['paired_deeptcr_minus_baseline'])} |",
        f"| Absolute difference between training seeds within a partition | {band(stats['absolute_training_seed_gap'])} |", "",
        f"DeepTCR scored higher than the matched baseline in {stats['deeptcr_above_baseline']} of "
        f"{stats['primary_evaluations']} evaluations. The baseline scored higher in "
        f"{stats['baseline_above_deeptcr']}, with {stats['ties']} ties.", "",
        "The DeepTCR median was below 0.50 and the range extended on both sides of 0.50. This cohort did not establish consistent patient-level discrimination.", "",
        "![DeepTCR and matched composition AUROC by split and fold count](../../figures/correction/repeated-patient-auroc.png)", "",
        "## All evaluations", "",
        "| Folds | Patient split seed | Training seed | DeepTCR AUROC | Composition AUROC | Comparable pairs |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        d, b = run["deeptcr"], run["composition"]
        content.append(f"| {run['folds']} | {run['partition_seed']} | {run['training_seed']} | "
                       f"{d['within_fold_auroc']:.3f} | {b['within_fold_auroc']:.3f} | {d['pairs']} |")
    content += [
        "", "## What the estimates measure", "",
        "The primary AUROC weights every responder–nonresponder comparison made within its own "
        "test fold equally. Grouping by patient keeps one patient's sequences together. DeepTCR "
        "fits its V/J vocabulary on training patients. The baseline uses sequence-length summaries "
        "and normalized amino-acid monomer and dimer frequencies. Its scaler and logistic regression "
        "are fitted on the matching training patients.", "",
        "The ranges describe variation across splits and fits of the same 34 people. The evaluations "
        "share patients, and the ranges are descriptive. The study supplies internal validation. "
        "An independent cohort is needed to assess generalization.", "",
        "[Declared study plan](repeat_plan.json) · [Aggregate result JSON](repeat_results.json) · "
        "[Results CSV](repeat_results.csv) · [Reproduction](../../SETUP.md)", "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(content))


if __name__ == "__main__":
    main()
