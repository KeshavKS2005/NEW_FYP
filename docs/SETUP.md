# Setup & Installation Guide
## 12-Module Open-World Continual Discovery Pipeline

> **This guide walks you through every step needed to get the project running from scratch on macOS.**

---

## ⚠️ Prerequisites Checklist (Before Anything Else)

You need these installed on your Mac before running any code:

| Requirement | Check Command | Minimum Version |
|---|---|---|
| Python | `python3 --version` | **3.10+** (3.9 will have issues with `google-genai`) |
| pip | `pip3 --version` | Any recent version |
| Homebrew (optional but helpful) | `brew --version` | Any |
| PostgreSQL | `psql --version` | 14+ |
| Git | `git --version` | Any |
| Git LFS | `git lfs --version` | Any (for model weights) |

> Your current system has **Python 3.9 (Xcode)** with **zero packages**. You must install Python 3.11 or 3.12 first (see Step 1).

---

## Step 1 — Install Python 3.11 (Recommended)

The project needs Python ≥ 3.10. The Xcode Python 3.9 bundled on your Mac should NOT be used.

**Option A: Homebrew (Recommended)**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Verify
python3.11 --version   # should say: Python 3.11.x
```

**Option B: Download directly from python.org**
1. Go to https://www.python.org/downloads/
2. Download Python 3.11.x for macOS
3. Run the `.pkg` installer

---

## Step 2 — Create a Virtual Environment

**Always use a virtual environment** — never install project packages to the system Python.

```bash
# Navigate to your project
cd /Users/mayur/Documents/NEW_FYP

# Create virtual environment using Python 3.11
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Verify you're inside the venv (should show venv path)
which python
# → /Users/mayur/Documents/NEW_FYP/venv/bin/python

# Upgrade pip inside the venv first
pip install --upgrade pip
```

> Every time you open a new terminal to work on this project, run `source venv/bin/activate` again.

---

## Step 3 — Install All Python Dependencies

```bash
# Make sure venv is activated first (source venv/bin/activate)

pip install -r requirements.txt
```

This installs **all packages** needed across all 12 modules. It will take 3–10 minutes depending on your internet speed (PyTorch alone is ~2 GB).

### What Gets Installed

| Package | Used By | Purpose |
|---|---|---|
| `opencv-python` | Module 1, 3 | MOG2 motion extraction, mask operations |
| `numpy` | All modules | Array math everywhere |
| `torch` + `torchvision` | Module 5, 6 | DINOv2 and FastSAM deep learning |
| `ultralytics` | Module 2, 5, 10, 11 | YOLOv11 + FastSAM + YOLO trainer |
| `faiss-cpu` | Module 7 | Cosine similarity vector search |
| `psycopg2-binary` | Module 8 | PostgreSQL database driver |
| `google-genai` | Module 9 | Gemini 2.5 Flash VLM API |
| `python-dotenv` | All modules | Load `.env` credentials |
| `Pillow` | Module 6, 9 | Image handling for VLM prompts |
| `PyYAML` | Module 10 | dataset.yaml generation |
| `pydantic` | Module 9 | Structured JSON schema from Gemini |
| `yt-dlp` | Test scripts | Resolve public CCTV stream URLs |

### GPU Users Only (CUDA)
If you have an NVIDIA GPU with CUDA:
```bash
# Replace torch with GPU version (check https://pytorch.org for exact command)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Replace faiss-cpu with faiss-gpu
pip install faiss-gpu
```

---

## Step 4 — Download Model Weights

Two model weight files are required and must be in the project root:

### 4a. FastSAM-s.pt (already in repo)
```bash
ls -lh FastSAM-s.pt
# If missing, download:
wget https://huggingface.co/CASIA-IVA-Lab/FastSAM-s/resolve/main/FastSAM-s.pt
```

### 4b. YOLOv11n base weights (already in repo)
```bash
ls -lh yolo11n.pt
# If missing, ultralytics auto-downloads on first run
```

### 4c. Champion model bootstrap
```bash
# Creates models/ directory and copies base YOLO weights as starting champion
mkdir -p models
python3 -c "
from ultralytics import YOLO
import shutil
m = YOLO('yolo11n.pt')
shutil.copy('yolo11n.pt', 'models/champion_latest.pt')
print('Champion weights ready at models/champion_latest.pt')
"
```

### 4d. DINOv2 (auto-downloaded on first run)
DINOv2 is downloaded automatically via `torch.hub` when the pipeline first runs. Requires internet access. Downloads ~340 MB to `~/.cache/torch/hub/`.

---

## Step 5 — Set Up PostgreSQL Database

Module 8 needs a running PostgreSQL server.

### Install PostgreSQL (macOS)
```bash
brew install postgresql@16
brew services start postgresql@16
```

### Create the surveillance database
```bash
# Connect to PostgreSQL
psql postgres

# Inside psql:
CREATE USER surveillance_user WITH PASSWORD 'your_password';
CREATE DATABASE surveillance OWNER surveillance_user;
\q

# Initialize the schema
psql -U surveillance_user -d surveillance -f schema.sql
```

---

## Step 6 — Create the `.env` Credentials File

The project **strictly** reads credentials from `../FYP_phase1/.env` (a sibling directory). Create it:

```bash
# Go one level up from the project
cd /Users/mayur/Documents

# Create sibling FYP_phase1 directory
mkdir -p FYP_phase1

# Create the .env file
cat > FYP_phase1/.env << 'EOF'
# PostgreSQL Credentials
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=surveillance
POSTGRES_USER=surveillance_user
POSTGRES_PASSWORD=your_password

# Google Gemini API Key (get from https://aistudio.google.com/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Override Gemini trigger count for faster testing
# GEMINI_TRIGGER_COUNT=15

# Optional: Point to a test video
# SURVEILLANCE_STREAM_URL=datasets/test_videos/bike_traffic_montreal.webm
EOF
```

---

## Step 7 — Create Required Directories

```bash
cd /Users/mayur/Documents/NEW_FYP

mkdir -p models
mkdir -p outputs
mkdir -p storage/frames
mkdir -p storage/crops
mkdir -p storage/masked_crops
mkdir -p storage/replay_buffer/images
mkdir -p storage/replay_buffer/labels
mkdir -p storage/datasets
```

---

## Step 8 — Verify Installation

Run this quick check to confirm all critical imports work:

```bash
python3 -c "
import cv2
import numpy as np
import torch
from ultralytics import YOLO, FastSAM
import faiss
import psycopg2
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
import yaml
print('✅ All packages imported successfully!')
print(f'   OpenCV:     {cv2.__version__}')
print(f'   PyTorch:    {torch.__version__}')
print(f'   CUDA:       {torch.cuda.is_available()}')
print(f'   MPS (Mac):  {torch.backends.mps.is_available()}')
print(f'   FAISS:      OK')
print(f'   Psycopg2:   OK')
"
```

---

## Step 9 — Run the Pipeline

### Run Phase 1 only (Modules 1–4, visual debug output)
```bash
python run_pipeline.py \
    --video "datasets/test_videos/bike_traffic_montreal.webm" \
    --output "outputs/pipeline_test.mp4" \
    --max_frames 120
```

### Run the full 12-module pipeline
```bash
python pipeline_12_modules.py
```

### Run discovery pipeline (Phases 1 + 2 only, no retraining)
```bash
python run_discovery_pipeline.py
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cv2` | OpenCV not installed | `pip install opencv-python` |
| `No module named 'ultralytics'` | Ultralytics not installed | `pip install ultralytics` |
| `RuntimeError: Failed to load DINOv2` | No internet / torch.hub blocked | Run with internet, or set `transformers` fallback |
| `psycopg2.OperationalError` | PostgreSQL not running | `brew services start postgresql@16` |
| `404 Model not found: gemini-2.5-flash` | Wrong API key or old google-genai | `pip install --upgrade google-genai` |
| `CRITICAL: Could not find FYP_phase1 .env file` | `.env` file in wrong location | Create at `/Users/mayur/Documents/FYP_phase1/.env` |
| `USE_SQLITE=1` mode | To skip PostgreSQL during dev | `export USE_SQLITE=1` before running |
