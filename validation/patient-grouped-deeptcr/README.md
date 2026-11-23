# Fixed-partition patient evaluation

Five stratified folds assign each patient to one held-out fold. A sixth fit shuffles labels among the training patients in fold 1 and evaluates the same held-out patients. The classifier is DeepTCR_SS 2.1.29 with a binary output. Every fit uses ten epochs.

[Results and fold denominators](REPORT.md) · [Protocol](PROTOCOL.md) · [Settings](protocol.json) · [Aggregate output](aggregate_results.json) · [Run instructions](../../SETUP.md)
