# Reproduce the patient-separated DeepTCR rerun

[Results](REPORT.md) · [Study design](PROTOCOL.md) · [Prespecified settings](protocol.json) · [Aggregate data](aggregate_results.json)

The run used the official [DeepTCR 2.1.29 source release](https://pypi.org/project/DeepTCR/2.1.29/#files), TensorFlow 2.12.0 and Python 3.9 on Apple Silicon. The runner calls the package’s own DeepTCR_SS graph builder and training batches. Its public entry point accepts an explicit input directory and source archive.

This directory describes the earlier fixed-partition evaluation. Follow the [setup guide](../../SETUP.md) for commands that run in a separate directory. Use the authorized study inputs. The directory must contain `deeptcr_trb_ready.csv` with `aminoAcid`, `vGeneName`, `jGeneName`, `patient_id` and `response_binary`, and its corresponding `y_labels.npy`. The runner checks row and patient counts, validates the archived labels, and can additionally verify the CSV against a trusted digest via `--expected-csv-sha256`. The SHA-256 of the public source distribution is fixed in [protocol.json](protocol.json). Keep study inputs, `.private/` and the run journal on the authorized system.
