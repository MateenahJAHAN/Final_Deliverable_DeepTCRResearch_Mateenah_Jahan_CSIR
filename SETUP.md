# Reproduce the patient-held-out evaluation

The completed repeat used Python 3.9.18, TensorFlow 2.12.0 and the official DeepTCR 2.1.29 source release on an Apple Silicon CPU. The same five stratified patient splits were used for both full runs.

## Inputs

- The authorized study directory containing `deeptcr_trb_ready.csv` and `y_labels.npy`.
- The [DeepTCR 2.1.29 source archive](https://pypi.org/project/DeepTCR/2.1.29/#files). Its SHA-256 is in [the protocol](validation/patient-grouped-deeptcr/protocol.json).
- The [pinned Python environment](validation/patient-grouped-deeptcr/requirements-lock.txt).

Use a fresh copy of `validation/patient-grouped-deeptcr` for each complete run. This keeps earlier outputs and the resumable journal separate. Set `AUTHORIZED_DEEPTCR_INPUT_DIR` and `DEEPTCR_RELEASE_ARCHIVE` to absolute paths.

```sh
cd validation/patient-grouped-deeptcr
uv venv --python 3.9 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt
mkdir -p vendor
tar -xzf "$DEEPTCR_RELEASE_ARCHIVE" -C vendor
PYTHONUNBUFFERED=1 NUMBA_NUM_THREADS=2 .venv/bin/python run.py \
  --device CPU --threads 10 \
  --data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" \
  --release-archive "$DEEPTCR_RELEASE_ARCHIVE" \
  --expected-csv-sha256 80d39ff4d5eb95a47555d1d609825e6cedb366bd71b320bae6f3d619743296de
```

A completed run writes `aggregate_results.json` after five ten-epoch patient folds and a separate ten-epoch shuffled-label fit. It verifies all 34 patients and 239,634 eligible records. Patient scores and model intermediates are kept in the private run directory.

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

## Build the research report

The [LaTeX source](paper/main.tex) uses the figures in `figures/correction/` and the bibliography in `paper/references.bib`. Run these commands from the repository root with Matplotlib and a LaTeX installation that includes `latexmk`.

```sh
python validation/repeated-patient-cv/render_repeat_study.py --figure figures/correction/repeated-patient-auroc.png
cd paper
build="$(mktemp -d)"
latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir="$build" main.tex
cp "$build/main.pdf" main.pdf
```

The performance table uses the [57-fit aggregate results](validation/repeated-patient-cv/repeat_results.json). Cohort and clustering figures use the [descriptive summaries](validation/correction-rerun/checked_descriptives.json) and the training-loss figure shows the [earlier fixed-partition repeat](validation/correction-rerun/aggregate_results.json).

[Earlier protocol and results](validation/patient-grouped-deeptcr/README.md) · [Historical setup guide](https://github.com/MateenahJAHAN/Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR/blob/ce1fa3a421edb9f4025a45cb79dc7b014867ad83/SETUP.md)
