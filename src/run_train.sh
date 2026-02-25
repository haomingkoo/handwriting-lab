#!/bin/sh
set -eux

LOG_DIR=/data/workspaces/haoming_koo/tmp/train_logs
mkdir -p "$LOG_DIR" /home/aisg/mnist

echo "=== date ===" | tee -a "$LOG_DIR/run.log"
date | tee -a "$LOG_DIR/run.log"

echo "=== whoami / pwd ===" | tee -a "$LOG_DIR/run.log"
whoami | tee -a "$LOG_DIR/run.log"
pwd | tee -a "$LOG_DIR/run.log"

echo "=== listing /data/workspaces/haoming_koo ===" | tee -a "$LOG_DIR/run.log"
ls -lah /data/workspaces/haoming_koo | tee -a "$LOG_DIR/run.log" || true

echo "=== linking dataset ===" | tee -a "$LOG_DIR/run.log"
ln -sfn /data/workspaces/haoming_koo/aiap-dsp-mlops-cicd/data /home/aisg/mnist/data

echo "=== training start ===" | tee -a "$LOG_DIR/run.log"
python -u src/train_model.py 2>&1 | tee -a "$LOG_DIR/run.log"