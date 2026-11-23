# From the sequencing libraries to the training rows

Li et al. deposited 59 CD19 CAR-T infusion-product libraries. The TCR contig CSVs in the local ZIP match the files streamed from all 58 GEO libraries that contain them. The clinical spreadsheet matches the publisher's Table S1A. Thirty-five patients had an evaluable three-month PET/CT outcome. One complete responder lacked a deposited TCR contig file. The analysis therefore includes 14 complete responses, four partial responses and 16 cases of progressive disease. [GEO files](geo-files.json) · [Cohort counts](summary.json) · [Publisher's clinical table](https://ars.els-cdn.com/content/image/1-s2.0-S1535610823003161-mmc2.xlsx)

Joining productive alpha and beta contigs within each cell reproduced all 239,637 archived pair rows, with zero missing or extra pairs and zero outcome-label discrepancies. These rows represent 198,267 cells. Among them, 32,844 yielded multiple pairings. The DeepTCR model consumes the beta sequence of each eligible pair, so a beta contig paired with two alpha contigs enters twice. Three rows exceeded the maximum CDR3 length.

## Does the counting unit change the composition baseline?

The comparison retains the same patients, outcomes, five primary partitions and fold-specific logistic regression. The original model used all eligible pair rows. The other two counts give one row to each productive beta contig. Including beta-only cells also changes which cells enter the repertoire.

| Repertoire input | Eligible rows | Median patient AUROC | Range across five partitions |
| --- | ---: | ---: | ---: |
| Original alpha–beta pair rows | 239,634 | 0.386 | 0.368–0.456 |
| Beta contigs in the same paired cells | 208,926 | 0.421 | 0.368–0.544 |
| All productive beta contigs in the same 34 products | 230,382 | 0.386 | 0.386–0.474 |

Counting each beta contig once in paired cells increased AUROC on four partitions and gave the same AUROC on the fifth. None of the three median estimates exceeded 0.5. The sensitivity applies to the composition comparator. A classifier trained with cell-level or contig-level weights would be a separate experiment. [Every partition, including three- and four-fold sensitivity checks](composition-sensitivity.json) · [Reproduction commands](../../SETUP.md#reconstruct-the-cohort-from-the-primary-files)

## Stability of the exploratory clusters

The original 50,000 records appeared in exactly the same order on replay. Adjusted Rand agreement with the archived five-cluster assignment was 0.228 using the original code and 0.444 after fixing the PCA seed. Across 34 patients and ten matched clusters, the 30,000-record analysis differed from the archived occupancies by 5.51 percentage points on average. The updated scripts fix the PCA and visualization draws. Two independent executions produced identical bytes for all 13 saved tables, arrays and PNG figures. [Archive comparison](descriptive-replay.json) · [Repeat comparison](descriptive-determinism.json)
