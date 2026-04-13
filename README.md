---
title: Leaf CT Scan Segmentation
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: "6.11.0"
app_file: app.py
pinned: true
---

# Leaf micro-CT Scan Segmentation

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Gradio%20%7C%20Colab-orange.svg)](https://gradio.app/)
![Science](https://img.shields.io/badge/Science-Plant%20Phenomics-green.svg)
![Research](https://img.shields.io/badge/Research-USDA--ARS-navy.svg)

Web application and utilization for automatic leaf micro-CT scan segmentation using a transformer-based model.

## How to Run

### Option 1 — HuggingFace Spaces (CPU)
Visit the live app — no setup required:
[https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation](https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation)

### Option 2 — Google Colab (GPU, recommended)
Run on a free T4 GPU:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WorasitSangjan/WebApp-Leaf-microCT-Segmentation/blob/main/run_colab.ipynb)

1. Click the "Open in Colab" button above 
2. In the Google Colab notebook, set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Click **Run All**
4. Click the `gradio.live` link that appears

### Option 3 — Run Locally (Python 3.10+)

We strongly recommend creating a Python virtual environment to run the application to avoid any dependency conflicts:

```bash
# 1. Clone the repository
git clone https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation.git
cd WebApp-Leaf-microCT-Segmentation

# 2. Create and activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate    # On Windows, use `venv\Scripts\activate`

# 3. Install requirements and run
pip install -r requirements.txt
python app.py
```

## Features
- **Single Image** — upload a PNG/JPG/TIF and run segmentation
- **Stack Image** — upload a multi-page TIFF stack, process all slices, and export volume statistics

## Output
| Output | Description |
|---|---|
| Class Label Mask | Grayscale mask with class indices |
| Color Mask | Per-class color visualization |
| Overlay | Mask blended on original image |
| Area Statistics (CSV) | Pixel count & percentage per class |
| Volume Statistics (CSV) | Voxel count & volume % across all slices (stack only) |
| Per-Slice Statistics (CSV) | Per-slice breakdown (stack only) |
| Full Stack TIFF | All slice masks as multi-page TIFF (stack only) |

## Tissue Classes
| Class | Color |
|---|---|
| Background | Black |
| Epidermis | Red |
| Vascular Region | Green |
| Mesophyll | Blue |
| Air Space | Yellow |

## Model
- **Architecture**: Encoder-only Mask Transformer (EoMT)
- **Backbone**: DINOv3 ViT-L/16
- **Weights**: [WorasitSangjan/Leaf-CT-Segmentation-Model](https://huggingface.co/WorasitSangjan/Leaf-CT-Segmentation-Model)
