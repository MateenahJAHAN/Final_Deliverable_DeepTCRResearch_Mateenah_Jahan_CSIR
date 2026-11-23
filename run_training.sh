#!/bin/bash
# Run the published patient-separated study design in a private working directory.
set -euo pipefail
umask 077
usage() {
  echo 'Usage: run_training.sh {repeated|fixed} OUTPUT_DIRECTORY' >&2
  echo 'Set AUTHORIZED_DEEPTCR_INPUT_DIR, DEEPTCR_RELEASE_ARCHIVE and DEEPTCR_PYTHON to absolute paths. See SETUP.md.' >&2
}
if [ "$#" -ne 2 ]; then usage; exit 2; fi
mode=$1
case "$mode" in fixed|repeated) ;; *) usage; exit 2 ;; esac
: "${AUTHORIZED_DEEPTCR_INPUT_DIR:?Set the directory containing the cohort CSV and labels.}"
: "${DEEPTCR_RELEASE_ARCHIVE:?Set the verified DeepTCR source archive path.}"
: "${DEEPTCR_PYTHON:?Set the Python executable in the study environment.}"
base=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
fixed="$base/validation/patient-grouped-deeptcr"
repeat="$base/validation/repeated-patient-cv"
digest=80d39ff4d5eb95a47555d1d609825e6cedb366bd71b320bae6f3d619743296de
archive_digest=7c4e8c37b1749a0fae5a16a7d56586e74cc84d3deacd1b2caf9599bdb54b4d70
test "$(shasum -a 256 "$DEEPTCR_RELEASE_ARCHIVE" | cut -d' ' -f1)" = "$archive_digest" || { echo 'DeepTCR archive digest mismatch' >&2; exit 1; }
for source in "$fixed/run.py" "$fixed/protocol.json"; do
  target="$out/${source##*/}"
  if [ -e "$target" ]; then cmp "$source" "$target"; else cp "$source" "$target"; fi
done
if [ ! -e "$out/vendor/deeptcr-2.1.29" ]; then
  mkdir -p "$out/vendor"
  tar -xzf "$DEEPTCR_RELEASE_ARCHIVE" -C "$out/vendor"
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
args=(--data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" --release-archive "$DEEPTCR_RELEASE_ARCHIVE" --expected-csv-sha256 "$digest")
if [ "$mode" = fixed ]; then
  exec "$DEEPTCR_PYTHON" -B "$out/run.py" --device CPU --threads 10 "${args[@]}"
fi
for source in "$repeat/run_repeats.py" "$repeat/evaluate_repeats.py" "$repeat/repeat_plan.json"; do
  target="$out/${source##*/}"
  if [ -e "$target" ]; then cmp "$source" "$target"; else cp "$source" "$target"; fi
done
"$DEEPTCR_PYTHON" -B "$out/run_repeats.py" --threads 5 --workers 2 --worker 0 "${args[@]}" >> "$out/worker-0.log" 2>&1 &
first=$!
"$DEEPTCR_PYTHON" -B "$out/run_repeats.py" --threads 5 --workers 2 --worker 1 "${args[@]}" >> "$out/worker-1.log" 2>&1 &
second=$!
failed=0
wait "$first" || failed=1
wait "$second" || failed=1
test "$failed" = 0
exec "$DEEPTCR_PYTHON" -B "$out/evaluate_repeats.py" --data-dir "$AUTHORIZED_DEEPTCR_INPUT_DIR" --expected-csv-sha256 "$digest"
