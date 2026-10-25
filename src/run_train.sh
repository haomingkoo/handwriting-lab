#!/bin/sh
set -eux

: "${TRAIN_DATA_DIR:?Set TRAIN_DATA_DIR to the training data directory}"
LOG_DIR="${LOG_DIR:-/tmp/train_logs}"
mkdir -p "$LOG_DIR" /home/aisg/mnist

echo "=== date ===" | tee -a "$LOG_DIR/run.log"
date | tee -a "$LOG_DIR/run.log"

echo "=== whoami / pwd ===" | tee -a "$LOG_DIR/run.log"
whoami | tee -a "$LOG_DIR/run.log"
pwd | tee -a "$LOG_DIR/run.log"

echo "=== listing $TRAIN_DATA_DIR ===" | tee -a "$LOG_DIR/run.log"
ls -lah "$TRAIN_DATA_DIR" | tee -a "$LOG_DIR/run.log" || true

echo "=== linking dataset ===" | tee -a "$LOG_DIR/run.log"
ln -sfn "$TRAIN_DATA_DIR" /home/aisg/mnist/data

echo "=== training start ===" | tee -a "$LOG_DIR/run.log"
python -u src/train_model.py 2>&1 | tee -a "$LOG_DIR/run.log"
