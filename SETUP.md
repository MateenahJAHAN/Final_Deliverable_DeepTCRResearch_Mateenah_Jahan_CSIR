# Reproduce the patient-held-out evaluation

DeepTCR 2.1.29 runs in Python 3.9.18 with TensorFlow 2.12.0 on Apple Silicon CPU. The primary evaluation uses five patient partitions and two training seeds. A separate fixed partition supplies a bootstrap interval and a shuffled-training-label control. Seeding both the graph and Keras fixes the starting weights in the tested environment. [Initialization measurements](validation/source-cohort/graph-initialization.json) · [Independent full-fold comparison](validation/source-cohort/full-fold-repeat.json)

## Inputs

- The authorized study directory containing `deeptcr_trb_ready.csv` and `y_labels.npy`.
- The [DeepTCR 2.1.29 source archive](https://pypi.org/project/DeepTCR/2.1.29/#files). Its SHA-256 is in [the protocol](validation/patient-grouped-deeptcr/protocol.json).
- The [pinned Python environment](validation/patient-grouped-deeptcr/requirements-lock.txt).

Start from the repository root. Set `AUTHORIZED_DEEPTCR_INPUT_DIR` and `DEEPTCR_RELEASE_ARCHIVE` to absolute paths. The commands create a temporary run directory for the outputs and resumable journal.

```sh
fixed_run="$(mktemp -d)"
base_source="$(pwd)/validation/patient-grouped-deeptcr"
cp "$base_source/run.py" "$base_source/protocol.json" "$base_source/requirements-lock.txt" "$fixed_run/"
uv venv --python 3.9.18 "$fixed_run/.venv"
export DEEPTCR_PYTHON="$fixed_run/.venv/bin/python"
uv pip sync --python "$DEEPTCR_PYTHON" "$fixed_run/requirements-lock.txt"
mkdir "$fixed_run/vendor"
tar -xzf "$DEEPTCR_RELEASE_ARCHIVE" -C "$fixed_run/vendor"
PYTHONUNBUFFERED=1 NUMBA_NUM_THREADS=2 "$DEEPTCR_PYTHON" -B "$fixed_run/run.py" \
  --device CPU --threads 10 \
  --data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" \
  --release-archive "$DEEPTCR_RELEASE_ARCHIVE" \
  --expected-csv-sha256 80d39ff4d5eb95a47555d1d609825e6cedb366bd71b320bae6f3d619743296de
```

A completed run writes `aggregate_results.json` in the temporary directory after five ten-epoch patient folds and a separate ten-epoch shuffled-label fit. It verifies all 34 patients and 239,634 eligible records. Patient scores and model intermediates are kept in the private run directory.

With the environment installed and the three absolute environment variables exported, `./run_training.sh fixed "$fixed_run"` runs the fixed study. Use `./run_training.sh repeated "$study"` with a separate output directory for the repeated study. The wrapper checks the release archive and propagates worker failures.

To retest initializers, copy `validation/patient-grouped-deeptcr/check_initialization.py` beside the fixed-run `run.py`, then run:

```sh
"$DEEPTCR_PYTHON" -B "$fixed_run/check_initialization.py" \
  --data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" --release-archive "$DEEPTCR_RELEASE_ARCHIVE" \
  --output "$fixed_run/initializers.json"
```

## Repeat across partitions and training seeds

[The repeated study plan](validation/repeated-patient-cv/repeat_plan.json) defines five stratified patient partitions, two training seeds, and separate three- and four-fold sensitivity runs. It sets the same 10 epochs and model settings for all 57 fits. Use the environment from the first section and set `DEEPTCR_PYTHON` to the absolute path of its Python executable. Start these commands from the repository root.

```sh
study="$(mktemp -d)"
base_source="$(pwd)/validation/patient-grouped-deeptcr"
repeat_source="$(pwd)/validation/repeated-patient-cv"
cp "$base_source/run.py" "$base_source/protocol.json" "$study/"
cp "$repeat_source/repeat_plan.json" "$repeat_source/run_repeats.py" "$repeat_source/evaluate_repeats.py" "$study/"
mkdir -p "$study/vendor"
tar -xzf "$DEEPTCR_RELEASE_ARCHIVE" -C "$study/vendor"
digest=80d39ff4d5eb95a47555d1d609825e6cedb366bd71b320bae6f3d619743296de
export PYTHONUNBUFFERED=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
fit() {
  "$DEEPTCR_PYTHON" -B "$study/run_repeats.py" \
    --data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" \
    --release-archive "$DEEPTCR_RELEASE_ARCHIVE" \
    --expected-csv-sha256 "$digest" --threads 5 --workers 2 --worker "$1"
}
fit 0 > "$study/worker-0.log" 2>&1 &
first=$!
fit 1 > "$study/worker-1.log" 2>&1 &
second=$!
failed=0
wait "$first" || failed=1
wait "$second" || failed=1
test "$failed" -eq 0 || exit 1
"$DEEPTCR_PYTHON" -B "$study/evaluate_repeats.py" \
  --data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" --expected-csv-sha256 "$digest"
```

The evaluator verifies all 57 fits and writes aggregate JSON and CSV. It fits the composition baseline on the same training folds and uses the private checkpoint scores to recalculate each held-out AUROC. Keep the study's `private/` directory on the authorized system.

## Reconstruct the cohort from the primary files

Use the GEO series TAR and its tab-separated [file list](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE241nnn/GSE241783/suppl/filelist.txt), the project's local per-library contig ZIP, and the [publisher's Table S1](https://ars.els-cdn.com/content/image/1-s2.0-S1535610823003161-mmc2.xlsx). The check reads all 59 libraries, matches productive alpha and beta chains within each cell, and verifies the full multiset of paired records and the clinical labels. Install pandas and openpyxl for this check.

```sh
python validation/check_source_cohort.py --geo-tar "$GEO_TAR" --geo-manifest "$GEO_MANIFEST" \
  --clinical "$TABLE_S1" --contigs-zip "$CONTIGS_ZIP" \
  --paired-csv "data_raw/deeptcr_complete_dataset (5).csv" \
  --processed-csv "$AUTHORIZED_DEEPTCR_INPUT_DIR/deeptcr_trb_ready.csv"
```

Scripts 07–13 and the original figure generator are preserved as historical source. Running them exits before they can substitute scores, plot the archived random attention weights, or overwrite figures. The patient-held-out workflow above uses measured predictions.

To repeat the chain-count sensitivity, add `--beta-sensitivity-dir "$BETA_DIR"` to the source check, choosing a new local directory for the per-contig tables. Then run the following command in the locked classifier environment.

```sh
"$DEEPTCR_PYTHON" validation/repeated-patient-cv/compare_beta_units.py \
  --paired-rows "$AUTHORIZED_DEEPTCR_INPUT_DIR/deeptcr_trb_ready.csv" \
  --source-beta-dir "$BETA_DIR" --output composition-sensitivity.json
```

## Build the research report

The two aggregate files are `validation/repeated-patient-cv/repeat_results.json` and `validation/patient-grouped-deeptcr/aggregate_results.json`. The paper uses these files and the chain-count sensitivity table. From the repository root:

```sh
python validation/repeated-patient-cv/render_repeat_study.py --figure figures/correction/repeated-patient-auroc.png
python validation/patient-grouped-deeptcr/report.py
python validation/patient-grouped-deeptcr/figures.py
python paper/render_manuscript_figures.py --cohort "$AUTHORIZED_DEEPTCR_INPUT_DIR/deeptcr_trb_ready.csv"
python paper/format_results.py
(cd paper && latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex)
```

To update the eight-slide presentation from the same aggregates and plots, install Pillow and lxml and run `python paper/render_slides.py` from the repository root. This refreshes the existing PowerPoint file.

The exploratory cluster figure is the occupancy plot produced by `scripts/15_unsupervised_vae_analysis.py` from 30,000 pair rows. The response-group means are in [cluster-occupancy.csv](validation/source-cohort/cluster-occupancy.csv).
