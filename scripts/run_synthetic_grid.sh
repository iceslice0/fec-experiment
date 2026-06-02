#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="${LOG_DIR:-$ROOT/outputs/synthetic/logs}"
CODEBOOK_DIR="${CODEBOOK_DIR:-$ROOT/outputs/synthetic/codebooks}"
mkdir -p "$LOG_DIR"
mkdir -p "$CODEBOOK_DIR"

read -r -a QS <<< "${QS:-0.1 0.2 0.3 0.4}"
read -r -a MS <<< "${MS:-1 8 16}"

for q in "${QS[@]}"; do
  for m in "${MS[@]}"; do
    LOG_FILE="$LOG_DIR/q${q}_m${m}.log"
    echo "=== q=$q m=$m ===" | tee "$LOG_FILE"
    "$PYTHON" -m is_fec_experiments.synthetic.bernoulli_experiment \
      --q "$q" --m "$m" \
      --codebook-out "$CODEBOOK_DIR/q${q}_m${m}.csv" \
      "$@" \
      | tee -a "$LOG_FILE"
  done
done

"$PYTHON" -m is_fec_experiments.synthetic.extract_r2 --log-dir "$LOG_DIR"
