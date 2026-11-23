#!/usr/bin/env python3
"""Write the result interpretation from the completed aggregate without patient data."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE / "aggregate_results.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Aggregate results are not complete")
    interval = result["within_fold_patient_auroc_bootstrap_ci95"]
    control = result["shuffled_training_control"]
    same_fold = result["folds"][control["fold"] - 1]
    balance = result["positive_patients"]
    scope = result["patients"]
    real = result["within_fold_patient_auroc"]
    verdict = (
        "The within-fold estimate fell below 0.50 in this fitted evaluation."
        if real < 0.5 else "The within-fold estimate was at or above 0.50 in this fitted evaluation."
    )
    lines = [
        "# Fixed-partition patient evaluation",
        "",
        (
            f"The official DeepTCR_SS 2.1.29 CNN completed five patient-exclusive folds using "
            f"{result['retained_rows']:,} eligible sequence records across {scope} CAR-T patients. "
            f"Every patient received one prediction from a model trained on other patients. "
            f"Within-fold patient AUROC was **{real:.3f}** "
            f"({result['patient_pairs_concordant_ties_half']:g}/{result['patient_pairs']} concordant "
            f"responder–nonresponder comparisons, with ties counted as half). "
            f"The conditional 95% patient bootstrap interval was **{interval[0]:.3f}–{interval[1]:.3f}**. "
            f"{verdict}"
        ),
        "",
        (
            f"Pooled out-of-fold patient AUROC was **{result['pooled_oof_patient_auroc']:.3f}**. "
            f"Average precision was **{result['pooled_oof_patient_average_precision']:.3f}** "
            f"with {balance}/{scope} responders ({balance/scope:.3f} prevalence). "
            f"At a score threshold of 0.5, {result['correct_at_threshold_0_5']}/{scope} patients "
            "were assigned their recorded class. Pooled scores compare outputs of different fitted models."
        ),
        "",
        "| Fold | Train patients | Test patients (responders) | Train sequence rows | Test sequence rows | Patient AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in result["folds"]:
        lines.append(
            f"| {f['fold']} | {f['train_patients']} | {f['test_patients']} ({f['test_positive_patients']}) | "
            f"{f['train_rows']:,} | {f['test_rows']:,} | {f['patient_auroc']:.3f} |"
        )
    lines += [
        "",
        "## Controls",
        "",
        (
            f"Within-fold label shuffles of the fixed predictions matched or exceeded the observed "
            f"concordance in {result['fixed_score_shuffles_at_least_as_high']:,}/{result['fixed_score_shuffle_draws']:,} "
            "draws. This fixed-score shuffle is a reference for the five fitted models. "
            "A patient's label can train the other folds’ models, so this shuffle supplies no formal study-wide p-value. "
            "The bootstrap also conditions on the fitted models and omits variability from retraining on a new cohort."
        ),
        "",
        (
            f"Retraining fold {control['fold']} with patient labels shuffled in the training partition "
            f"changed the assigned classes of {control['train_patient_labels_changed']}/{control['train_patients']} "
            f"training patients. This model scored AUROC {control['patient_auroc']:.3f} on the same "
            f"{control['test_patients']} held-out patients. The original-label fit for that fold scored "
            f"{same_fold['patient_auroc']:.3f}. Final mean minibatch loss was "
            f"(original labels {same_fold['epochs'][-1]['batch_mean_loss']:.3f}, shuffled labels "
            f"{control['epochs'][-1]['batch_mean_loss']:.3f}). The seven test patients give this "
            "single-fold control limited precision."
        ),
        "",
        "## Reproduction",
        "",
        (
            "The run used the archived label array, the package’s original large sequence CNN, "
            "training-fold V/J vocabularies and ten fixed epochs per fold. Each patient score is "
            "the mean of the model's responder-class sequence scores. Three archived sequences "
            "exceeded DeepTCR’s 40-residue limit. Model initialization, patient folds and batch "
            "ordering used fixed seeds. Each fold optimizer fit on its training patients. "
            "The primary AUROC gives each eligible within-fold responder–nonresponder pair equal weight."
        ),
        "",
        (
            f"TensorFlow {result['runtime']['tensorflow']} ran on the local "
            f"{result['runtime']['host']} {result['runtime']['device']} backend. "
            "The classifier used a two-class head and patient-exclusive folds. "
            "The study includes one 34-patient cohort and "
            "evaluates patient aggregates of DeepTCR_SS sequence scores."
        ),
        "",
        "In the [figure](deeptcr_patient_run.png), the horizontal whisker marks the conditional 95% "
        "bootstrap interval and the dashed vertical line marks an AUROC of 0.50.",
        "",
        "[Frozen protocol](PROTOCOL.md) · [Aggregate JSON](aggregate_results.json) · "
        "[Model and loss figure](deeptcr_patient_run.png) · [Runner](run.py)",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
