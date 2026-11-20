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

## Rebuild the corrected documents

Run `python paper/build_correction.py` in an environment with NumPy, Matplotlib and ReportLab. The renderer reads the [verified repeat aggregates](validation/correction-rerun/aggregate_results.json) and [descriptive summary](validation/correction-rerun/checked_descriptives.json). The latter summarizes all eligible records for S2 and reaggregates the original exploratory cluster shares for S12.

[Earlier protocol and results](validation/patient-grouped-deeptcr/README.md) · [Historical setup guide](https://github.com/MateenahJAHAN/Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR/blob/ce1fa3a421edb9f4025a45cb79dc7b014867ad83/SETUP.md)
