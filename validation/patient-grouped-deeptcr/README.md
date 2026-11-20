# Reproduce the patient-separated DeepTCR rerun

[Results](REPORT.md) · [Study design](PROTOCOL.md) · [Prespecified settings](protocol.json) · [Aggregate data](aggregate_results.json)

The run used the official [DeepTCR 2.1.29 source release](https://pypi.org/project/DeepTCR/2.1.29/#files), TensorFlow 2.12.0 and Python 3.9 on Apple Silicon. The runner calls the package’s own DeepTCR_SS graph builder and training batches. Its public entry point accepts an explicit input directory and source archive.

Run from this directory with the authorized study inputs. The directory must contain `deeptcr_trb_ready.csv` with `aminoAcid`, `vGeneName`, `jGeneName`, `patient_id` and `response_binary`, and its corresponding `y_labels.npy`. The runner checks row and patient counts, validates the archived labels, and can additionally verify the CSV against a trusted digest via `--expected-csv-sha256`. The SHA-256 of the public source distribution is fixed in [protocol.json](protocol.json). Keep study inputs, `.private/` and the run journal on the authorized system.

```sh
uv venv --python 3.9 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt
mkdir -p vendor
tar -xzf "$DEEPTCR_RELEASE_ARCHIVE" -C vendor
PYTHONUNBUFFERED=1 NUMBA_NUM_THREADS=2 .venv/bin/python run.py \
  --data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" \
  --release-archive "$DEEPTCR_RELEASE_ARCHIVE"
```

Set `DEEPTCR_RELEASE_ARCHIVE` to the absolute path of the downloaded source distribution. Set `AUTHORIZED_DEEPTCR_INPUT_DIR` to the approved input directory.
