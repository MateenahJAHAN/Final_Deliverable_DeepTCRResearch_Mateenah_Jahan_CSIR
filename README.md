# DeepTCR and CD19 CAR-T response

This correction revisits the CSIR-IGIB trainee analysis by Mateenah Jahan and Sherry Bhalla. It evaluates the DeepTCR sequence classifier on held-out patients from the archived cohort.

- [Corrected report with supplement](paper/main.pdf)
- [Standalone supplement](paper/supplement.pdf)
- [Corrected presentation](results/DeepTCR_Results_Presentation.pptx)
- [Setup and reproduction](SETUP.md)

## Repeated patient-held-out evaluation

The study completed **57 full ten-epoch fits**. Five stratified patient partitions crossed with two training seeds yielded ten five-fold evaluations of all 34 patients. Median within-fold DeepTCR AUROC was **0.456** (range 0.298–0.579). The matched composition baseline had median AUROC **0.386** (range 0.368–0.456 across five partitions). DeepTCR scored higher in 8 of ten paired evaluations. The report includes both fold-count sensitivity checks.

[All results and methods](validation/repeated-patient-cv/REPORT.md) · [Results CSV](validation/repeated-patient-cv/repeat_results.csv)

![DeepTCR and the matched composition baseline across splits and fold counts](figures/correction/repeated-patient-auroc.png)

## Patient-separated rerun

These two complete evaluations used one earlier patient partition.

The repeat scored all 34 patients across five disjoint held-out folds. The official DeepTCR_SS 2.1.29 sequence classifier used 239,634 eligible TCRβ records and trained for ten epochs per fold. The within-fold AUROC was **0.439** (25 of 57 concordant-pair equivalents, conditional 95% bootstrap interval 0.246–0.649). The earlier full run on the same split scored **0.316** (0.140–0.509). Both are estimates from the same small cohort.

[Repeat-run aggregate data](validation/correction-rerun/aggregate_results.json) · [Earlier full run](validation/patient-grouped-deeptcr/REPORT.md)

![Held-out patient AUROC and observed training loss](figures/correction/held-out-performance.png)

## Original trainee submission

The January 2026 report, figures, and presentation document the original analysis. Its reported 0.754 AUROC summarizes logged sequence-split epochs without patient separation. The archived attention scores came from a random fallback. The corrected report explains the revised evaluation.

[Historical PDF](https://github.com/MateenahJAHAN/Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR/blob/ce1fa3a421edb9f4025a45cb79dc7b014867ad83/paper/main.pdf) · [Historical figure set](https://github.com/MateenahJAHAN/Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR/tree/ce1fa3a421edb9f4025a45cb79dc7b014867ad83/figures/paper_final) · [Historical slide deck](https://github.com/MateenahJAHAN/Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR/blob/ce1fa3a421edb9f4025a45cb79dc7b014867ad83/results/DeepTCR_Results_Presentation.pptx)
