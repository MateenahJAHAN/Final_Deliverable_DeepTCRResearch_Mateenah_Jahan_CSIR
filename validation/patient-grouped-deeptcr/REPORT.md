# Fixed-partition patient evaluation

The official DeepTCR_SS 2.1.29 CNN completed five patient-exclusive folds using 239,634 eligible sequence records across 34 CAR-T patients. Every patient received one prediction from a model trained on other patients. Within-fold patient AUROC was **0.333** (19/57 concordant responder–nonresponder comparisons, with ties counted as half). The conditional 95% patient bootstrap interval was **0.140–0.544**. The within-fold estimate fell below 0.50 in this fitted evaluation.

Pooled out-of-fold patient AUROC was **0.455**. Average precision was **0.560** with 18/34 responders (0.529 prevalence). At a score threshold of 0.5, 15/34 patients were assigned their recorded class. Pooled scores compare outputs of different fitted models.

| Fold | Train patients | Test patients (responders) | Train sequence rows | Test sequence rows | Patient AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 27 | 7 (4) | 197,696 | 41,938 | 0.167 |
| 2 | 27 | 7 (4) | 182,334 | 57,300 | 0.167 |
| 3 | 27 | 7 (4) | 188,560 | 51,074 | 0.417 |
| 4 | 27 | 7 (3) | 191,013 | 48,621 | 0.417 |
| 5 | 28 | 6 (3) | 198,933 | 40,701 | 0.556 |

## Controls

Within-fold label shuffles of the fixed predictions matched or exceeded the observed concordance in 4,748/4,999 draws. This fixed-score shuffle is a reference for the five fitted models. A patient's label can train the other folds’ models, so this shuffle supplies no formal study-wide p-value. The bootstrap also conditions on the fitted models and omits variability from retraining on a new cohort.

Retraining fold 1 with patient labels shuffled in the training partition changed the assigned classes of 14/27 training patients. This model scored AUROC 0.750 on the same 7 held-out patients. The original-label fit for that fold scored 0.167. Final mean minibatch loss was (original labels 0.453, shuffled labels 0.452). The seven test patients give this single-fold control limited precision.

## Reproduction

The run used the archived label array, the package’s original large sequence CNN, training-fold V/J vocabularies and ten fixed epochs per fold. Each patient score is the mean of the model's responder-class sequence scores. Three archived sequences exceeded DeepTCR’s 40-residue limit. Model initialization, patient folds and batch ordering used fixed seeds. Each fold optimizer fit on its training patients. The primary AUROC gives each eligible within-fold responder–nonresponder pair equal weight.

TensorFlow 2.12.0 ran on the local Darwin arm64 CPU backend. The classifier used a two-class head and patient-exclusive folds. The study includes one 34-patient cohort and evaluates patient aggregates of DeepTCR_SS sequence scores.

In the [figure](deeptcr_patient_run.png), the horizontal whisker marks the conditional 95% bootstrap interval and the dashed vertical line marks an AUROC of 0.50.

[Frozen protocol](PROTOCOL.md) · [Aggregate JSON](aggregate_results.json) · [Model and loss figure](deeptcr_patient_run.png) · [Runner](run.py)
