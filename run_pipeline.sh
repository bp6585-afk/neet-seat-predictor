#!/usr/bin/env bash
# One-click: create venv → install → scrape → train → launch app
set -e

VENV=".venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "=== Step 1: Set up virtualenv ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
$PIP install --quiet --upgrade pip
$PIP install -r requirements.txt

echo ""
echo "=== Step 2: Scrape closing rank data ==="
$PY -m scraper.scrape

echo ""
echo "=== Step 3: Train the DNN model ==="
$PY -m model.train --epochs 25

echo ""
echo "=== Step 4: Launch the app ==="
$VENV/bin/streamlit run app.py
