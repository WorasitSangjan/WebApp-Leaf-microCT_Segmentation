import sys

# Suppress Python 3.13 asyncio cleanup noise (Invalid file descriptor: -1)
def _suppress_invalid_fd(unraisable):
    if isinstance(unraisable.exc_value, ValueError) and "Invalid file descriptor" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = _suppress_invalid_fd

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel
import os
import csv
import tempfile

# ── PLATFORM SWITCH ──────────────────────────────────────────────────────────
# When deploying on HuggingFace Spaces, uncomment the two lines below.
# On NERSC Spin or the laptop, leave them commented out.
#
# import spaces
# USE_ZEROGPU = True

MODEL_REPO = "WorasitSangjan/Leaf-CT-Segmentation-Model"
MODEL_FILE = "best_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

EXAMPLE_PATHS = [
    ("examples/Lantana.png",  "Lantana"),
    ("examples/Olive.png",    "Olive"),
    ("examples/Pine.png",     "Pine"),
    ("examples/Viburnum.png", "Viburnum"),
    ("examples/Wheat.png",    "Wheat"),
]


# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────
class EoMT_ViTL(nn.Module):
    """
    Encoder-only Mask Transformer with DINOv3 ViT-L/16 backbone.
    Backbone: facebook/dinov3-vitl16-pretrain-lvd1689m
    """
    def __init__(self, num_classes=5, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_q       = num_queries

        self.conv_in  = nn.Conv2d(1, 3, kernel_size=1)
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            local_files_only=False,
        )
        embed_dim        = self.backbone.config.hidden_size  # 1024
        patch_size       = self.backbone.config.patch_size   # 16
        self._embed_dim  = embed_dim
        self._patch_size = patch_size

        self.q          = nn.Embedding(num_queries, embed_dim)
        self.query_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=16, dropout=0.0, batch_first=True
        )
        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        self.mask_head  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
        )

    def forward(self, images):
        B, C, H, W = images.shape
        if C == 1:
            images = self.conv_in(images)

        hidden  = self.backbone(images).last_hidden_state
        spatial = hidden[:, 1:, :]  # skip CLS token

        q_tokens     = self.q.weight[None, :, :].expand(B, -1, -1)
        q_out, _     = self.query_attn(q_tokens, spatial, spatial)
        class_logits = self.class_head(q_out)

        grid_h, grid_w = H // self._patch_size, W // self._patch_size
        expected       = grid_h * grid_w
        if spatial.shape[1] > expected:
            spatial = spatial[:, :expected, :]
        spatial_grid = spatial.transpose(1, 2).reshape(B, self._embed_dim, grid_h, grid_w)
        spatial_up   = self.upscale(spatial_grid)

        mask_probs      = torch.einsum("bqd,bdhw->bqhw", self.mask_head(q_out), spatial_up).sigmoid()
        semantic_logits = torch.einsum("bqc,bqhw->bchw", class_logits[..., :-1], mask_probs)
        return F.interpolate(semantic_logits, size=(H, W), mode="bilinear", align_corners=False)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_model():
    """Download and load the trained EoMT ViT-L segmentation model from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        print(f"Loading model on {DEVICE}...")
        model      = EoMT_ViTL(num_classes=5)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Remap only layer keys: backbone.layer.X.* → backbone.model.layer.X.*
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("backbone.layer."):
                remapped["backbone.model.layer." + k[len("backbone.layer."):]] = v
            else:
                remapped[k] = v
        model.load_state_dict(remapped)
        model.eval()
        model.to(DEVICE)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"WARNING: Could not load model — {e}. Running in demo mode.")
        return None

# Load model once at startup (not on every user request)
model = load_model()

# ── PATCH-BASED INFERENCE (matches evaluate.py exactly) ──────────────────────
PATCH_SIZE = 320
STRIDE     = 80
NUM_CLASSES = 5

def _gaussian_kernel(size):
    ax = torch.linspace(-1, 1, size)
    g  = torch.exp(-ax**2 / 0.5)
    k  = torch.outer(g, g)
    return k / k.max()

def run_patch_inference(model, image: Image.Image):
    """
    Tile the image into 320x320 patches (stride 80), run inference on each,
    stitch back with Gaussian weighting — identical to evaluate.py.
    Returns mask_array (H, W) of class indices.
    """
    # Step 1: grayscale numpy
    img_np = np.array(image.convert("L"), dtype=np.float32)
    H, W   = img_np.shape

    # Step 2: per-image z-score on non-zero pixels
    img_t = torch.tensor(img_np).unsqueeze(0)  # (1, H, W)
    valid = img_t[img_t > 0]
    if len(valid) > 0:
        mean, std = valid.mean(), valid.std()
        if std > 1e-5:
            img_t = (img_t - mean) / std

    # Step 3: build patch grid (same logic as evaluate.py _grid_positions)
    def grid_pos(size, patch, stride):
        positions = list(range(0, max(1, size - patch + 1), stride))
        if not positions or positions[-1] != max(0, size - patch):
            positions.append(max(0, size - patch))
        return positions

    tops  = grid_pos(H, PATCH_SIZE, STRIDE)
    lefts = grid_pos(W, PATCH_SIZE, STRIDE)

    # Step 4: accumulate logits with Gaussian weighting
    accum = torch.zeros(NUM_CLASSES, H, W)
    count = torch.zeros(H, W)
    gk    = _gaussian_kernel(PATCH_SIZE)

    with torch.no_grad():
        for t in tops:
            for l in lefts:
                ph = min(PATCH_SIZE, H - t)
                pw = min(PATCH_SIZE, W - l)

                patch = img_t[:, t:t+ph, l:l+pw]

                # Pad to PATCH_SIZE if needed
                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    patch = F.pad(patch, (0, PATCH_SIZE-pw, 0, PATCH_SIZE-ph), value=0.0)

                out = model(patch.unsqueeze(0).to(DEVICE))  # (1, C, 320, 320)
                out = out.squeeze(0).float().cpu()          # (C, 320, 320)

                weight = gk[:ph, :pw]
                accum[:, t:t+ph, l:l+pw] += out[:, :ph, :pw] * weight.unsqueeze(0)
                count[t:t+ph, l:l+pw]    += weight

    # Step 5: normalize and argmax
    count = count.clamp(min=1e-6)
    mask_array = (accum / count.unsqueeze(0)).numpy().argmax(axis=0).astype(np.uint8)
    return mask_array

# ── CLASS DEFINITIONS ─────────────────────────────────────────────────────────
CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]

CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0 Background — black
    [255, 100, 100],   # 1 Epidermis  — red
    [100, 200, 100],   # 2 Vascular   — green
    [100, 100, 255],   # 3 Mesophyll  — blue
    [255, 230, 50 ],   # 4 Air Space  — yellow
], dtype=np.uint8)

# ── OVERLAY HELPER ────────────────────────────────────────────────────────────
def overlay_mask(original: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Blend the segmentation mask on top of the original CT scan image."""
    original_rgb = original.convert("RGB").resize(mask.size)
    return Image.blend(original_rgb, mask, alpha=alpha)

def mask_to_images(mask_array):
    """Convert class index mask to color mask and grayscale label images."""
    color_mask = Image.fromarray(CLASS_COLORS[mask_array])
    label_img  = Image.fromarray((mask_array * 50).astype(np.uint8)).convert("RGB")
    return color_mask, label_img

def save_csv(stats: list, filename: str = "area_stats.csv") -> str:
    """Save area stats list to a temporary CSV file. Returns the file path."""
    path = os.path.join(tempfile.mkdtemp(), filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Pixels", "Percentage"])
        writer.writerows(stats)
    return path

def save_volume_csv(stats: list, filename: str = "volume_stats_stack.csv") -> str:
    """Save volume stats for stack (voxel counts) to a temporary CSV file."""
    path = os.path.join(tempfile.mkdtemp(), filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Voxels", "Volume %"])
        writer.writerows(stats)
    return path

def save_image_png(img: Image.Image, filename: str = "output.png") -> str:
    """Save a PIL image as PNG to a temp file. Returns the file path."""
    path = os.path.join(tempfile.mkdtemp(), filename)
    img.save(path)
    return path

def save_tiff_stack(images: list, filename: str = "stack.tif") -> str:
    """Save a list of PIL images as a multi-page TIFF. Returns the file path."""
    path = os.path.join(tempfile.mkdtemp(), filename)
    images[0].save(path, save_all=True, append_images=images[1:])
    return path

def save_perslice_csv(per_slice: list, filename: str = "area_stats_per_slice.csv") -> str:
    """Save per-slice stats to CSV. per_slice is a list of (slice_idx, mask_array)."""
    path = os.path.join(tempfile.mkdtemp(), filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Slice", "Class", "Pixels", "Percentage"])
        for slice_idx, mask_array in per_slice:
            total = mask_array.size
            for c, name in enumerate(CLASS_NAMES):
                count = int(np.sum(mask_array == c))
                pct = round(100 * count / total, 2)
                writer.writerow([slice_idx + 1, name, f"{count:,}", f"{pct:.2f}%"])
    return path

def calculate_area_stats(mask_array: np.ndarray) -> list:
    """
    Calculate pixel count and percentage for each class in the segmentation mask.
    mask_array: 2D numpy array of class indices (H, W)
    Returns a list of rows for the Gradio Dataframe.
    """
    total = mask_array.size
    rows = []
    for c, name in enumerate(CLASS_NAMES):
        count = int(np.sum(mask_array == c))
        pct = round(100 * count / total, 2)
        rows.append([name, f"{count:,}", f"{pct:.2f}%"])
    return rows

# ── SINGLE IMAGE INFERENCE ────────────────────────────────────────────────────
# @spaces.GPU   # ← uncomment for HuggingFace ZeroGPU deployment
def run_segmentation(input_image: Image.Image):
    """Run model on a single uploaded image."""

    if input_image is None:
        raise gr.Error("Please upload a CT scan image first.")

    if model is None:
        grayscale = input_image.convert("L").convert("RGB")
        overlay = overlay_mask(input_image, grayscale)
        empty_stats = [[name, "N/A", "N/A"] for name in CLASS_NAMES]
        csv_path     = save_csv(empty_stats, "area_stats.csv")
        label_path   = save_image_png(grayscale, "label_grayscale.png")
        color_path   = save_image_png(grayscale, "color_mask.png")
        overlay_path = save_image_png(overlay, "overlay.png")
        return grayscale, grayscale, overlay, empty_stats, csv_path, label_path, color_path, overlay_path, "Demo mode — No model loaded."

    try:
        mask_array = run_patch_inference(model, input_image)
        color_mask, label_img = mask_to_images(mask_array)
        overlay = overlay_mask(input_image, color_mask)
        stats = calculate_area_stats(mask_array)
        csv_path     = save_csv(stats, "area_stats.csv")
        label_path   = save_image_png(label_img, "label_grayscale.png")
        color_path   = save_image_png(color_mask, "color_mask.png")
        overlay_path = save_image_png(overlay, "overlay.png")
        return color_mask, label_img, overlay, stats, csv_path, label_path, color_path, overlay_path, "Done — Segmentation complete."

    except Exception as e:
        raise gr.Error(f"Model inference failed: {str(e)}")

# ── STACKED TIFF INFERENCE ────────────────────────────────────────────────────
# @spaces.GPU   # ← uncomment for HuggingFace ZeroGPU deployment
def run_segmentation_stack(stack_file):
    """
    Run model on every slice of a multi-page TIFF stack.
    Returns the middle slice as a representative visual output,
    and area stats averaged across all slices.
    """

    if stack_file is None:
        raise gr.Error("Please upload a stack image first.")

    try:
        tiff = Image.open(stack_file.name if hasattr(stack_file, 'name') else stack_file)
    except Exception as e:
        raise gr.Error(f"Could not open file: {str(e)}")

    # Count total slices
    n_frames = getattr(tiff, "n_frames", 1)
    print(f"Stack contains {n_frames} slice(s).")

    if model is None:
        print("Demo mode: Processing middle slice only.")
        mid = n_frames // 2
        tiff.seek(mid)
        slice_img = tiff.copy().convert("RGB")
        grayscale = slice_img.convert("L").convert("RGB")
        overlay = overlay_mask(slice_img, grayscale)
        empty_stats = [[name, "N/A", "N/A"] for name in CLASS_NAMES]
        csv_path     = save_volume_csv(empty_stats, "volume_stats_stack.csv")
        perslice_path = os.path.join(tempfile.mkdtemp(), "area_stats_per_slice.csv")
        with open(perslice_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Slice", "Class", "Pixels", "Percentage"])
            for name in CLASS_NAMES:
                writer.writerow([mid + 1, name, "N/A", "N/A"])
        label_path   = save_image_png(grayscale, "label_grayscale.png")
        color_path   = save_image_png(grayscale, "color_mask.png")
        overlay_path = save_image_png(overlay, "overlay.png")
        return grayscale, grayscale, overlay, empty_stats, csv_path, perslice_path, label_path, color_path, overlay_path, f"Demo mode — Showed slice {mid + 1} of {n_frames}, no model loaded."

    # Accumulate pixel counts and collect all mask slices
    total_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    per_slice_data = []
    all_label_imgs  = []
    all_color_masks = []
    all_overlays    = []
    mid_color_mask = None
    mid_label_img  = None
    mid_overlay    = None
    mid_idx = n_frames // 2

    try:
        for i in range(n_frames):
            tiff.seek(i)
            slice_img = tiff.copy()

            mask_array = run_patch_inference(model, slice_img)
            color_mask, label_img = mask_to_images(mask_array)
            ovl = overlay_mask(slice_img, color_mask)
            for c in range(len(CLASS_NAMES)):
                total_counts[c] += int(np.sum(mask_array == c))

            per_slice_data.append((i, mask_array))
            all_label_imgs.append(label_img)
            all_color_masks.append(color_mask)
            all_overlays.append(ovl)

            if i == mid_idx:
                mid_color_mask = color_mask
                mid_label_img  = label_img
                mid_overlay    = ovl

        # Build aggregated stats table
        grand_total = int(total_counts.sum())
        stats = []
        for c, name in enumerate(CLASS_NAMES):
            count = int(total_counts[c])
            pct = round(100 * count / grand_total, 2) if grand_total > 0 else 0.0
            stats.append([name, f"{count:,}", f"{pct:.2f}%"])

        csv_path          = save_volume_csv(stats, "volume_stats_stack.csv")
        perslice_csv_path = save_perslice_csv(per_slice_data, "area_stats_per_slice.csv")
        label_path        = save_tiff_stack(all_label_imgs,  "label_all_slices.tif")
        color_path        = save_tiff_stack(all_color_masks, "color_all_slices.tif")
        overlay_path      = save_tiff_stack(all_overlays,    "overlay_all_slices.tif")
        return mid_color_mask, mid_label_img, mid_overlay, stats, csv_path, perslice_csv_path, label_path, color_path, overlay_path, f"Done — {n_frames} slices processed, showing middle slice ({mid_idx + 1})."

    except Exception as e:
        raise gr.Error(f"Stack inference failed: {str(e)}")

def preview_and_store_stack(f):
    """Open TIFF stack, show middle slice in preview, store original path for inference."""
    if f is None:
        return None, None, "Please upload a TIFF stack file."
    filepath = f.name if hasattr(f, 'name') else f
    try:
        tiff = Image.open(filepath)
        n = getattr(tiff, "n_frames", 1)
        tiff.seek(n // 2)
        preview = tiff.copy().convert("RGB")
        return preview, filepath, f"✅ Stack uploaded ({n} slices) — Click 'Run Segmentation on Stack'."
    except Exception as e:
        return None, None, f"Error reading stack: {str(e)}"

# ── GRADIO INTERFACE ──────────────────────────────────────────────────────────
css = """
/* ── WHITE + GREEN THEME ─────────────────────────────────────────────────── */
body { background: #f7faf7 !important; }
.gradio-container { background: #f7faf7 !important; color: #111 !important; max-width: 100% !important; padding: 20px !important; }
footer { display: none !important; }

/* ── FONT SIZES ──────────────────────────────────────────────────────────── */
body, .gradio-container { font-size: 18px !important; }
h1 { font-size: 2.2rem !important; color: #1b5e20 !important; }
h2 { font-size: 1.9rem !important; color: #2e7d32 !important; }
h3, h4 { font-size: 1.3rem !important; color: #2e7d32 !important; }
p, label, span, td, th, strong, b, em { font-size: 17px !important; color: #111 !important; }
.label-wrap span, label span { font-size: 17px !important; font-weight: 600 !important; }
button { font-size: 17px !important; font-weight: 600 !important; }

/* ── BLOCK / PANEL ───────────────────────────────────────────────────────── */
.block, .gr-box, .gr-panel, .gr-form, .gr-padded, .wrap { background: #ffffff !important; }
.block { overflow: hidden !important; border-radius: 8px !important; }

/* ── TAB BUTTONS (Gradio 6 compatible) ───────────────────────────────────── */
button[role="tab"],
.tab-nav button {
    color: #2e7d32 !important;
    background: #f1f8f1 !important;
    border: 1px solid #c8e6c9 !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}
button[role="tab"][aria-selected="true"],
.tab-nav button.selected {
    background: #2e7d32 !important;
    color: #ffffff !important;
    border-color: #2e7d32 !important;
}
button[role="tab"]:hover,
.tab-nav button:hover {
    background: #c8e6c9 !important;
}

/* ── REMOVE Gradio 6 orange/accent tab indicator line ───────────────────── */
[role="tablist"]::before, [role="tablist"]::after,
.tab-nav::before, .tab-nav::after,
button[role="tab"]::before, button[role="tab"]::after,
button[role="tab"] span::after { display: none !important; content: none !important; }
button[role="tab"] { border-bottom: none !important; box-shadow: none !important; outline: none !important; }
button[role="tab"][aria-selected="true"] { border-bottom: none !important; box-shadow: none !important; }

/* ── OUTER FRAME around main tabs ────────────────────────────────────────── */
#main-tabs {
    border: 2px solid #2e7d32 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #ffffff !important;
}
[role="tablist"], .tab-nav {
    background: #f1f8f1 !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 6px 6px 0 6px !important;
    display: flex !important;
    gap: 4px !important;
}
[role="tabpanel"], .tabitem {
    background: #ffffff !important;
    padding: 12px !important;
}

/* ── RESULT COLUMN — outer frame wraps tab buttons + image ───────────────── */
#result-col, #stack-result-col {
    border: 2px solid #2e7d32 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #ffffff !important;
    padding: 0 !important;
    box-shadow: none !important;
}
/* Tab button row inside result col */
#result-col [role="tablist"],
#stack-result-col [role="tablist"] {
    border-bottom: 2px solid #2e7d32 !important;
    background: #f1f8f1 !important;
    margin: 0 !important;
    padding: 6px 6px 0 6px !important;
    border-radius: 0 !important;
}
/* Tab panel (image area) */
#result-col [role="tabpanel"],
#stack-result-col [role="tabpanel"] {
    border: none !important;
    padding: 8px !important;
    background: #ffffff !important;
}
/* Remove inner block borders to avoid double-frame */
#result-col .block, #stack-result-col .block,
#result-tabs, #stack-result-tabs {
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* ── STATUS BLOCK ────────────────────────────────────────────────────────── */
#status-block { border: 1px solid #a5d6a7 !important; border-radius: 8px !important; padding: 8px 16px !important; background: #f1f8f1 !important; }
#status-block label { font-weight: 700 !important; color: #2e7d32 !important; -webkit-text-fill-color: #2e7d32 !important; }
#status-block textarea { color: #111111 !important; -webkit-text-fill-color: #111111 !important; background: #f1f8f1 !important; font-size: 15px !important; }

/* ── TEXTBOX / INPUTS ────────────────────────────────────────────────────── */
textarea, input[type="text"] { background: #f1f8f1 !important; border-color: #a5d6a7 !important; color: #111111 !important; -webkit-text-fill-color: #111111 !important; }

/* ── BUTTONS ─────────────────────────────────────────────────────────────── */
#run-btn, #stack-run-btn { background: #2e7d32 !important; border-color: #2e7d32 !important; color: #ffffff !important; }
#run-btn:hover, #stack-run-btn:hover { background: #1b5e20 !important; }
button.secondary { background: #f1f8f1 !important; border: 1px solid #a5d6a7 !important; color: #2e7d32 !important; }
button.secondary:hover { background: #c8e6c9 !important; }

/* ── TABLE / DATAFRAME ───────────────────────────────────────────────────── */
table, .table-wrap, .svelte-table,
.gr-dataframe, [data-testid="dataframe"], .dataframe-container { background: #fff !important; }
.dataframe-container { margin-top: 2px !important; }
#area-table, #stack-area-table { margin-top: -32px !important; }

thead, thead tr, thead th, th {
    background: #e8f5e9 !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    font-size: 15px !important;
    border-color: #c8e6c9 !important;
}
[data-testid="dataframe"] thead th,
[data-testid="dataframe"] thead th span,
[data-testid="dataframe"] thead th * {
    background: #e8f5e9 !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}
tbody, tbody tr, tr {
    background: #ffffff !important;
}
tbody tr:nth-child(even), tr:nth-child(even) {
    background: #f7fdf7 !important;
}
td, tbody td, tr td {
    background: transparent !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    border-color: #e8f5e9 !important;
    font-size: 15px !important;
}
/* Gradio 6 cell wrappers */
.cell-wrap, [class*="cell"], [class*="row"] {
    background: transparent !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

/* ── IMAGE BLOCKS — sage green background ───────────────────────────────── */
#img-input, #img-input .wrap, #img-label, #img-label .wrap,
#img-color, #img-color .wrap, #img-overlay, #img-overlay .wrap,
#img-stack-preview, #img-stack-preview .wrap, #img-stack-label, #img-stack-label .wrap,
#img-stack-color, #img-stack-color .wrap, #img-stack-overlay, #img-stack-overlay .wrap,
#stack-file-upload, #stack-file-upload .wrap,
#example-gallery, #example-gallery .wrap, #example-gallery .thumbnail-item,
#example-gallery button, #example-gallery li {
    background-color: #a8bfaa !important;
}

/* Image badge label text — white, larger */
#img-input label, #img-input label *, #img-label label, #img-label label *,
#img-color label, #img-color label *, #img-overlay label, #img-overlay label *,
#img-stack-preview label, #img-stack-preview label *, #img-stack-label label, #img-stack-label label *,
#img-stack-color label, #img-stack-color label *, #img-stack-overlay label, #img-stack-overlay label *,
#stack-file-upload label, #stack-file-upload label * {
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    font-size: 17px !important; font-weight: 600 !important;
}

/* ── ROW WRAPPERS — no border, transparent ───────────────────────────────── */
#main-row, #stack-main-row { border: none !important; padding: 0 !important; background: transparent !important; box-shadow: none !important; }

/* ── STACK FILE UPLOAD — large drop zone ────────────────────────────────── */
#stack-file-upload, #stack-file-upload > div, #stack-file-upload .wrap { min-height: unset !important; height: 70px !important; max-height: 70px !important; }
#stack-file-upload .wrap { flex-direction: row !important; align-items: center !important; justify-content: center !important; gap: 8px !important; }
#stack-file-upload .wrap p { margin: 0 !important; }
#stack-file-upload .wrap svg { width: 20px !important; height: 20px !important; }

/* ── HIDE image input bottom toolbar (icons strip) ───────────────────────── */
#img-input .source-selection,
#img-input [data-testid="source-select"],
#img-input .icon-buttons,
#img-input .toolbar,
#img-input > .block > div > div:last-child { display: none !important; }

/* ── FILE DOWNLOAD BLOCKS ────────────────────────────────────────────────── */
#dl-csv, #dl-label, #dl-color, #dl-overlay,
#stack-dl-csv, #stack-dl-perslice, #stack-dl-label, #stack-dl-color, #stack-dl-overlay {
    background: #ffffff !important;
}
#dl-csv .upload-container, #dl-label .upload-container, #dl-color .upload-container, #dl-overlay .upload-container,
#stack-dl-csv .upload-container, #stack-dl-perslice .upload-container, #stack-dl-label .upload-container,
#stack-dl-color .upload-container, #stack-dl-overlay .upload-container,
#dl-csv [data-testid="file"], #dl-label [data-testid="file"], #dl-color [data-testid="file"], #dl-overlay [data-testid="file"],
#stack-dl-csv [data-testid="file"], #stack-dl-perslice [data-testid="file"], #stack-dl-label [data-testid="file"],
#stack-dl-color [data-testid="file"], #stack-dl-overlay [data-testid="file"],
#dl-csv > div, #dl-label > div, #dl-color > div, #dl-overlay > div,
#stack-dl-csv > div, #stack-dl-perslice > div, #stack-dl-label > div,
#stack-dl-color > div, #stack-dl-overlay > div {
    min-height: unset !important;
    height: 60px !important;
    max-height: 60px !important;
    overflow: hidden !important;
}
/* White text only on the dark label badge */
#dl-csv label, #dl-csv label *, #dl-label label, #dl-label label *,
#dl-color label, #dl-color label *, #dl-overlay label, #dl-overlay label *,
#stack-dl-csv label, #stack-dl-csv label *, #stack-dl-perslice label, #stack-dl-perslice label *,
#stack-dl-label label, #stack-dl-label label *, #stack-dl-color label, #stack-dl-color label *,
#stack-dl-overlay label, #stack-dl-overlay label * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
/* Filename text and links — black */
#dl-csv a, #dl-csv span, #dl-label a, #dl-label span, #dl-color a, #dl-color span, #dl-overlay a, #dl-overlay span,
#stack-dl-csv a, #stack-dl-csv span, #stack-dl-perslice a, #stack-dl-perslice span,
#stack-dl-label a, #stack-dl-label span, #stack-dl-color a, #stack-dl-color span,
#stack-dl-overlay a, #stack-dl-overlay span {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

/* ── IMAGE DISPLAY — scale to fit, no crop ───────────────────────────────── */
#img-input img, #img-label img, #img-color img, #img-overlay img,
#img-stack-preview img, #img-stack-label img, #img-stack-color img, #img-stack-overlay img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}

/* ── EXAMPLE GALLERY CAPTIONS — instant on render, no flash ─────────────── */
#example-gallery [class*="caption"],
#example-gallery figcaption,
#example-gallery .caption-label {
    background: #2e7d32 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    position: absolute !important;
    top: 4px !important;
    left: 4px !important;
    bottom: auto !important;
    right: auto !important;
    border-radius: 4px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 2px 8px !important;
}

/* ── LOADING / PROCESSING STATE ──────────────────────────────────────────── */
/* Keep Gradio's built-in shimmer animation but recolor the border */
.generating, [class*="generating"],
.loading, [class*="loading"],
.pending, [class*="pending"] {
    border-color: #2e7d32 !important;
}

/* Progress bar */
.progress-bar, #progress-bar,
[class*="progress"] { background: #2e7d32 !important; }

/* "processing | 65.2s" timer badge */
.eta-bar, [class*="eta"], .timer, [class*="timer"],
[class*="status"]:not(#status-block) {
    background: #2e7d32 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Spinner circle (SVG) */
svg circle, svg.loading circle, svg.spinner circle {
    stroke: #2e7d32 !important;
}

/* Pending/loading shimmer on buttons */
button.pending, button[disabled],
#run-btn.loading, #stack-run-btn.loading {
    background: #66bb6a !important;
    border-color: #66bb6a !important;
    color: #ffffff !important;
}
"""

with gr.Blocks(title="Leaf CT Scan Segmentation", css=css) as demo:

    gr.HTML("""
        <h1 style='color:#1b5e20; font-size:2.2rem; margin-bottom:6px;'>Leaf CT Scan Segmentation</h1>
        <p style='color:#111; font-size:17px; margin:0 0 4px 0;'>An automatic leaf CT scan segmentation tool built on an Encoder-only Mask Transformer (EoMT) with a DINOv3 ViT-L backbone.</p>
        <p style='color:#111111; font-size:17px; margin:0; -webkit-text-fill-color:#111111;'><strong style='color:#2e7d32; -webkit-text-fill-color:#2e7d32;'>Features:</strong><span style='color:#111111; -webkit-text-fill-color:#111111;'> Upload a single image or a multi-slice TIFF stack to visualize tissue segmentation and export area statistics.</span></p>
    """)

    # Status bar
    status = gr.Textbox(
        label="Status",
        value="Ready — Please upload a CT scan and click Run.",
        interactive=False,
        elem_id="status-block",
    )

    with gr.Tabs(elem_id="main-tabs"):

        # ── TAB 1: SINGLE IMAGE ───────────────────────────────────────────
        with gr.Tab("Single Image"):
            with gr.Row(equal_height=True, elem_id="main-row"):
                with gr.Column():
                    input_image = gr.Image(
                        type="pil",
                        label="Upload CT Scan (.png / .jpg / .tif)",
                        height=500,
                        elem_id="img-input",
                    )

                with gr.Column(elem_id="result-col"):
                    with gr.Tabs(elem_id="result-tabs"):
                        with gr.Tab("Class Label"):
                            output_label = gr.Image(type="pil", label="Class Label (Grayscale)", interactive=False, height=500, elem_id="img-label")
                        with gr.Tab("Color Mask"):
                            output_color = gr.Image(type="pil", label="Color Mask", interactive=False, height=500, elem_id="img-color")
                        with gr.Tab("Overlay"):
                            output_overlay = gr.Image(type="pil", label="Overlay Image( Mask on original)", interactive=False, height=500, elem_id="img-overlay")

            with gr.Row():
                submit_btn = gr.Button("Run Segmentation", variant="primary", elem_id="run-btn")
                clear_btn  = gr.Button("Clear", variant="secondary")

            gr.HTML("""
                <h2 style='margin:10px 0 0 0; color:#2e7d32'>Example Images</h2>
                <p style='margin:0; color:#555; font-size:15px;'>Click an image to load it into the upload area.</p>
            """)
            example_gallery = gr.Gallery(
                value=EXAMPLE_PATHS,
                show_label=False,
                columns=5,
                rows=1,
                height=160,
                min_width=50,
                object_fit="contain",
                allow_preview=False,
                elem_id="example-gallery",
            )

            with gr.Column():
                gr.HTML("<h2 style='margin:0 0 2px 0; color:#2e7d32'>Area Statistics</h2>")
                area_table = gr.Dataframe(
                    headers=["Class", "Pixels", "Percentage"],
                    label="",
                    interactive=False,
                    elem_id="area-table",
                )
                gr.Markdown("## Save Output")
                with gr.Row():
                    dl_csv     = gr.File(label="1. Area Statistics", interactive=False, elem_id="dl-csv", height=80)
                    dl_label   = gr.File(label="2. Label Mask", interactive=False, elem_id="dl-label", height=80)
                    dl_color   = gr.File(label="3. Color Mask", interactive=False, elem_id="dl-color", height=80)
                    dl_overlay = gr.File(label="4. Overlay Image", interactive=False, elem_id="dl-overlay", height=80)

            def load_example_image(evt: gr.SelectData):
                img = Image.open(EXAMPLE_PATHS[evt.index][0])
                return img, "✅ Example loaded — Click 'Run Segmentation' to process."

            example_gallery.select(
                fn=load_example_image,
                outputs=[input_image, status],
            )

            input_image.upload(
                fn=lambda img: (img, "✅ Image uploaded — Click 'Run Segmentation' to process."),
                inputs=[input_image],
                outputs=[input_image, status],
            )
            run_event = submit_btn.click(
                fn=run_segmentation,
                inputs=[input_image],
                outputs=[output_color, output_label, output_overlay, area_table, dl_csv, dl_label, dl_color, dl_overlay, status],
            )
            clear_btn.click(
                fn=lambda: (None, None, None, None, "Ready — Please upload a CT scan and click Run.", None, None, None, None, None),
                outputs=[input_image, output_color, output_label, output_overlay, status, area_table, dl_csv, dl_label, dl_color, dl_overlay],
                cancels=[run_event],
            )

        # ── TAB 2: TIFF STACK ─────────────────────────────────────────────
        with gr.Tab("Stack Image"):
            stack_path_state = gr.State(None)
            with gr.Row(equal_height=True, elem_id="stack-main-row"):
                with gr.Column():
                    stack_file = gr.File(
                        label="Upload CT Stack (.tif / .tiff)",
                        file_types=[".tif", ".tiff"],
                        height=90,
                        elem_id="stack-file-upload",
                    )
                    stack_preview = gr.Image(
                        type="pil",
                        label="Stack Preview (Middle slice)",
                        interactive=False,
                        height=350,
                        elem_id="img-stack-preview",
                    )

                with gr.Column(elem_id="stack-result-col"):
                    with gr.Tabs(elem_id="stack-result-tabs"):
                        with gr.Tab("Class Label"):
                            stack_output_label = gr.Image(type="pil", label="Class Label (Middle slice)", interactive=False, height=500, elem_id="img-stack-label")
                        with gr.Tab("Color Mask"):
                            stack_output_color = gr.Image(type="pil", label="Color Mask (Middle slice)", interactive=False, height=500, elem_id="img-stack-color")
                        with gr.Tab("Overlay"):
                            stack_output_overlay = gr.Image(type="pil", label="Overlay Image (Middle slice)", interactive=False, height=500, elem_id="img-stack-overlay")

            with gr.Row():
                stack_submit_btn = gr.Button("Run Segmentation on Stack", variant="primary", elem_id="stack-run-btn")
                stack_clear_btn  = gr.Button("Clear", variant="secondary")

            with gr.Column():
                gr.HTML("<h2 style='margin:0 0 2px 0; color:#2e7d32'>Volume Statistics (All slices)</h2>")
                stack_area_table = gr.Dataframe(
                    headers=["Class", "Voxels", "Volume %"],
                    label="",
                    interactive=False,
                    elem_id="stack-area-table",
                )
                gr.Markdown("## Save Output")
                with gr.Row():
                    stack_dl_csv          = gr.File(label="1. Area All Slices", interactive=False, elem_id="stack-dl-csv", height=80)
                    stack_dl_perslice_csv = gr.File(label="2. Area Per Slice", interactive=False, elem_id="stack-dl-perslice", height=80)
                    stack_dl_label        = gr.File(label="3. Label All Slices", interactive=False, elem_id="stack-dl-label", height=80)
                    stack_dl_color        = gr.File(label="4. Color All Slices", interactive=False, elem_id="stack-dl-color", height=80)
                    stack_dl_overlay      = gr.File(label="5. Overlay All Slices", interactive=False, elem_id="stack-dl-overlay", height=80)

            stack_file.upload(
                fn=preview_and_store_stack,
                inputs=[stack_file],
                outputs=[stack_preview, stack_path_state, status],
            )
            stack_run_event = stack_submit_btn.click(
                fn=run_segmentation_stack,
                inputs=[stack_path_state],
                outputs=[stack_output_color, stack_output_label, stack_output_overlay, stack_area_table, stack_dl_csv, stack_dl_perslice_csv, stack_dl_label, stack_dl_color, stack_dl_overlay, status],
            )
            stack_clear_btn.click(
                fn=lambda: (None, None, None, None, None, None, None, None, None, None, None, None, "Ready — Please upload a CT stack and click Run."),
                outputs=[stack_file, stack_preview, stack_path_state, stack_output_color, stack_output_label, stack_output_overlay, stack_area_table, stack_dl_csv, stack_dl_perslice_csv, stack_dl_label, stack_dl_color, stack_dl_overlay, status],
                cancels=[stack_run_event],
            )

    gr.Markdown("Note: Processing time depends on image size and server load.")

    demo.load(
        fn=lambda: [],
        inputs=[],
        outputs=[],
        js="""() => {
            function fixGradioSpecific() {
                // .gradio-container
                var container = document.querySelector('.gradio-container');
                if (container) {
                    container.style.setProperty('background', '#f7faf7', 'important');
                    container.style.setProperty('max-width', '100%', 'important');
                    container.style.setProperty('padding', '20px', 'important');
                    container.style.setProperty('color', '#111', 'important');
                }
                // All .block elements — white background, rounded
                // Skip any block that contains (or IS) an image/upload block
                var greenIds = ['img-input','img-label','img-color','img-overlay',
                                'img-stack-preview','img-stack-label','img-stack-color','img-stack-overlay',
                                'stack-file-upload'];
                document.querySelectorAll('.block').forEach(function(el) {
                    var isGreen = greenIds.some(function(id) {
                        return el.id === id || el.querySelector('#' + id);
                    });
                    if (!isGreen) {
                        el.style.setProperty('background', '#ffffff', 'important');
                        el.style.setProperty('border-radius', '8px', 'important');
                        el.style.setProperty('overflow', 'hidden', 'important');
                    }
                });
                // Result col inner blocks — remove double borders
                ['result-col', 'stack-result-col'].forEach(function(id) {
                    var col = document.getElementById(id);
                    if (!col) return;
                    col.querySelectorAll('.block, [class*="tabs"]').forEach(function(el) {
                        el.style.setProperty('border', 'none', 'important');
                        el.style.setProperty('border-radius', '0', 'important');
                        el.style.setProperty('box-shadow', 'none', 'important');
                        el.style.setProperty('background', 'transparent', 'important');
                    });
                });
                // Image block inner .wrap — sage green background
                ['img-input','img-label','img-color','img-overlay',
                 'img-stack-preview','img-stack-label','img-stack-color','img-stack-overlay',
                 'stack-file-upload'].forEach(function(id) {
                    var el = document.getElementById(id);
                    if (!el) return;
                    el.querySelectorAll('div').forEach(function(w) {
                        w.style.setProperty('background-color', '#a8bfaa', 'important');
                    });
                });
                // Stack file upload — compact height
                var su = document.getElementById('stack-file-upload');
                if (su) {
                    [su].concat(Array.from(su.querySelectorAll('> div, .wrap, .upload-container'))).forEach(function(el) {
                        el.style.setProperty('min-height', 'unset', 'important');
                        el.style.setProperty('height', '70px', 'important');
                        el.style.setProperty('max-height', '70px', 'important');
                    });
                    var wrap = su.querySelector('.wrap, .upload-container');
                    if (wrap) {
                        wrap.style.setProperty('flex-direction', 'row', 'important');
                        wrap.style.setProperty('align-items', 'center', 'important');
                        wrap.style.setProperty('justify-content', 'center', 'important');
                        wrap.style.setProperty('gap', '8px', 'important');
                    }
                    su.querySelectorAll('p').forEach(function(p) { p.style.setProperty('margin', '0', 'important'); });
                    su.querySelectorAll('svg').forEach(function(s) { s.style.setProperty('width', '20px', 'important'); s.style.setProperty('height', '20px', 'important'); });
                }
                // Hide image input toolbar
                var imgInput = document.getElementById('img-input');
                if (imgInput) {
                    imgInput.querySelectorAll('.source-selection, .icon-buttons, .toolbar, [data-testid="source-select"]').forEach(function(el) {
                        el.style.setProperty('display', 'none', 'important');
                    });
                }
                // File download blocks — compact height
                ['dl-csv','dl-label','dl-color','dl-overlay','stack-dl-csv','stack-dl-perslice','stack-dl-label','stack-dl-color','stack-dl-overlay'].forEach(function(id) {
                    var el = document.getElementById(id);
                    if (!el) return;
                    el.style.setProperty('background-color', '#ffffff', 'important');
                    el.querySelectorAll('.upload-container, [data-testid="file"], > div').forEach(function(child) {
                        child.style.setProperty('min-height', 'unset', 'important');
                        child.style.setProperty('height', '60px', 'important');
                        child.style.setProperty('max-height', '60px', 'important');
                        child.style.setProperty('overflow', 'hidden', 'important');
                        child.style.setProperty('background-color', '#ffffff', 'important');
                    });
                });
                // Secondary / Clear buttons
                document.querySelectorAll('button.secondary').forEach(function(btn) {
                    btn.style.setProperty('background', '#f1f8f1', 'important');
                    btn.style.setProperty('border', '1px solid #a5d6a7', 'important');
                    btn.style.setProperty('color', '#2e7d32', 'important');
                    btn.style.setProperty('-webkit-text-fill-color', '#2e7d32', 'important');
                });
                // Dataframe / table wrappers
                document.querySelectorAll('.table-wrap, .svelte-table, .gr-dataframe, .dataframe-container, [data-testid="dataframe"]').forEach(function(el) {
                    el.style.setProperty('background', '#fff', 'important');
                });
                // Cell wrappers
                document.querySelectorAll('.cell-wrap').forEach(function(el) {
                    el.style.setProperty('background', 'transparent', 'important');
                    el.style.setProperty('color', '#111111', 'important');
                    el.style.setProperty('-webkit-text-fill-color', '#111111', 'important');
                });
                // Progress / loading indicators
                document.querySelectorAll('.progress-bar, [id="progress-bar"]').forEach(function(el) {
                    el.style.setProperty('background', '#2e7d32', 'important');
                });
                document.querySelectorAll('.eta-bar, .timer').forEach(function(el) {
                    el.style.setProperty('background', '#2e7d32', 'important');
                    el.style.setProperty('color', '#ffffff', 'important');
                    el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                });
                // Always re-apply image block green last (overrides .block white above)
                fixImageLabels();
            }
            function fixBadgeText() {
                document.querySelectorAll('*').forEach(function(el) {
                    if (el.children.length > 3) return;
                    // Skip elements that already have an explicit inline color set
                    var inlineStyle = el.getAttribute('style') || '';
                    if (inlineStyle.indexOf('color') >= 0) return;
                    var rect = el.getBoundingClientRect();
                    if (rect.width < 5 || rect.width > 800 || rect.height < 5 || rect.height > 80) return;
                    var bg = window.getComputedStyle(el).backgroundColor;
                    var m = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    if (!m) return;
                    var brightness = (parseInt(m[1])*299 + parseInt(m[2])*587 + parseInt(m[3])*114) / 1000;
                    if (brightness < 40) {
                        el.style.setProperty('color', '#ffffff', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                        Array.from(el.children).forEach(function(c) {
                            var cs = c.getAttribute('style') || '';
                            if (cs.indexOf('color') >= 0) return;
                            c.style.setProperty('color', '#ffffff', 'important');
                            c.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                        });
                    }
                });
                ['dl-csv','dl-label','dl-color','dl-overlay','stack-dl-csv','stack-dl-perslice','stack-dl-label','stack-dl-color','stack-dl-overlay'].forEach(function(id) {
                    var container = document.getElementById(id);
                    if (!container) return;
                    // Force white background on container and wrappers
                    container.style.setProperty('background-color', '#ffffff', 'important');
                    container.querySelectorAll('.wrap, .upload-container, > div').forEach(function(el) {
                        el.style.setProperty('background-color', '#ffffff', 'important');
                    });
                    // Label text white
                    container.querySelectorAll('label, label *').forEach(function(el) {
                        el.style.setProperty('color', '#ffffff', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                    });
                    // Filename text black
                    container.querySelectorAll('a, [class*="file-name"], [class*="filename"], span:not([class*="label"])').forEach(function(el) {
                        el.style.setProperty('color', '#111111', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#111111', 'important');
                    });
                });
            }
            function fixTableHeaders() {
                document.querySelectorAll('table thead th, table th').forEach(function(el) {
                    el.style.setProperty('background', '#e8f5e9', 'important');
                    el.style.setProperty('color', '#111111', 'important');
                    el.style.setProperty('-webkit-text-fill-color', '#111111', 'important');
                });
                document.querySelectorAll('table thead th *, table th *').forEach(function(el) {
                    el.style.setProperty('background', 'transparent', 'important');
                    el.style.setProperty('color', '#111111', 'important');
                    el.style.setProperty('-webkit-text-fill-color', '#111111', 'important');
                });
            }
            function fixImageLabels() {
                ['img-input','img-label','img-color','img-overlay',
                 'img-stack-preview','img-stack-label','img-stack-color','img-stack-overlay',
                 'stack-file-upload'].forEach(function(id) {
                    var block = document.getElementById(id);
                    if (!block) return;
                    block.style.setProperty('background-color', '#a8bfaa', 'important');
                    block.querySelectorAll('div').forEach(function(el) {
                        el.style.setProperty('background-color', '#a8bfaa', 'important');
                    });
                    block.querySelectorAll('label, label *, [class*="label"] span, [class*="label"] *').forEach(function(el) {
                        el.style.setProperty('color', '#ffffff', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                    });
                });
                // gr.File outer container — same treatment as gr.Image blocks
                var fileUpload = document.getElementById('stack-file-upload');
                if (fileUpload) {
                    var outerBlock = fileUpload.closest('[data-testid="file"]') || fileUpload.parentElement;
                    if (outerBlock) {
                        outerBlock.style.setProperty('background-color', '#a8bfaa', 'important');
                        outerBlock.querySelectorAll('div').forEach(function(el) {
                            el.style.setProperty('background-color', '#a8bfaa', 'important');
                        });
                    }
                }
            }
            function fixDropText() {
                document.querySelectorAll('p, span').forEach(function(el) {
                    var t = (el.innerText || el.textContent || '').trim();
                    if (t === 'Drop Image Here' || t === 'Drop File Here' || t === 'Click to Upload' || t === 'or' || t === '- or -') {
                        el.style.setProperty('color', '#ffffff', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                    }
                });
            }
            function fixExamples() {
                // Hide the auto "Examples" label
                document.querySelectorAll('span, div, p, label').forEach(function(el) {
                    var t = (el.innerText || el.textContent || '').trim();
                    if (t === 'Examples') {
                        el.style.setProperty('display', 'none', 'important');
                    }
                });
                // Fix all buttons that contain only text (example name buttons)
                document.querySelectorAll('button').forEach(function(btn) {
                    var t = (btn.innerText || btn.textContent || '').trim();
                    if (t === 'Lantana' || t === 'Olive' || t === 'Pine' || t === 'Viburnum' || t === 'Wheat') {
                        btn.style.setProperty('color', '#111111', 'important');
                        btn.style.setProperty('-webkit-text-fill-color', '#111111', 'important');
                        btn.style.setProperty('background', '#f1f8f1', 'important');
                        btn.style.setProperty('border', '1px solid #c8e6c9', 'important');
                        btn.querySelectorAll('*').forEach(function(child) {
                            child.style.setProperty('color', '#111111', 'important');
                            child.style.setProperty('-webkit-text-fill-color', '#111111', 'important');
                        });
                    }
                });
            }
            function fixGalleryCaptions() {
                var gallery = document.getElementById('example-gallery');
                if (!gallery) return;
                gallery.style.setProperty('margin-top', '-20px', 'important');
                gallery.style.setProperty('padding-top', '0', 'important');
                var parent = gallery.parentElement;
                if (parent) {
                    parent.style.setProperty('margin-top', '0', 'important');
                    parent.style.setProperty('padding-top', '0', 'important');
                }
                gallery.querySelectorAll('[class*="caption"], figcaption, .caption-label').forEach(function(el) {
                    el.style.setProperty('color', '#ffffff', 'important');
                    el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                    el.style.setProperty('background', '#2e7d32', 'important');
                    el.style.setProperty('position', 'absolute', 'important');
                    el.style.setProperty('top', '4px', 'important');
                    el.style.setProperty('left', '4px', 'important');
                    el.style.setProperty('bottom', 'auto', 'important');
                    el.style.setProperty('right', 'auto', 'important');
                    el.style.setProperty('border-radius', '4px', 'important');
                    el.style.setProperty('font-size', '13px', 'important');
                    el.style.setProperty('font-weight', '600', 'important');
                    el.style.setProperty('padding', '2px 8px', 'important');
                    var parent = el.parentElement;
                    if (parent && window.getComputedStyle(parent).position === 'static') {
                        parent.style.setProperty('position', 'relative', 'important');
                    }
                });
            }
            function fixCorners() {
                ['result-col', 'stack-result-col'].forEach(function(id) {
                    var col = document.getElementById(id);
                    if (!col) return;
                    // Outer frame on the column itself
                    col.style.setProperty('border', '2px solid #2e7d32', 'important');
                    col.style.setProperty('border-radius', '10px', 'important');
                    col.style.setProperty('overflow', 'hidden', 'important');
                    col.style.setProperty('background', '#ffffff', 'important');
                    col.style.setProperty('padding', '0', 'important');
                    // Tab button row — light green background + bottom divider
                    var tabList = col.querySelector('[role="tablist"]');
                    if (tabList) {
                        tabList.style.setProperty('background', '#f1f8f1', 'important');
                        tabList.style.setProperty('border-bottom', '2px solid #2e7d32', 'important');
                        tabList.style.setProperty('border-radius', '0', 'important');
                        tabList.style.setProperty('margin', '0', 'important');
                        tabList.style.setProperty('padding', '6px 6px 0 6px', 'important');
                    }
                    // Remove inner block borders
                    col.querySelectorAll('.block, [role="tabpanel"], [class*="tabs"]').forEach(function(el) {
                        el.style.setProperty('border', 'none', 'important');
                        el.style.setProperty('box-shadow', 'none', 'important');
                        el.style.setProperty('border-radius', '0', 'important');
                    });
                });
                // Clean up main-row wrappers
                ['main-row', 'stack-main-row'].forEach(function(id) {
                    var el = document.getElementById(id);
                    if (!el) return;
                    el.style.setProperty('border', 'none', 'important');
                    el.style.setProperty('background', 'transparent', 'important');
                });
                // Status block: white bg, black normal text everywhere first
                var sb = document.getElementById('status-block');
                if (sb) {
                    sb.style.setProperty('border', 'none', 'important');
                    sb.style.setProperty('box-shadow', 'none', 'important');
                    sb.style.setProperty('background', '#ffffff', 'important');
                    sb.querySelectorAll('*').forEach(function(el) {
                        el.style.setProperty('color', '#111111', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#111111', 'important');
                        el.style.setProperty('font-weight', 'normal', 'important');
                        if (el.tagName !== 'TEXTAREA' && el.tagName !== 'INPUT') {
                            el.style.setProperty('background', '#ffffff', 'important');
                            el.style.setProperty('border', 'none', 'important');
                        }
                    });
                    // Then override label and its text children to green+bold
                    sb.querySelectorAll('label, label span, label strong, label p').forEach(function(el) {
                        el.style.setProperty('color', '#2e7d32', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#2e7d32', 'important');
                        el.style.setProperty('font-weight', 'bold', 'important');
                    });
                }
            }
            setTimeout(fixGradioSpecific, 300);
            setTimeout(fixGradioSpecific, 1000);
            setTimeout(fixGradioSpecific, 3000);
            setTimeout(fixBadgeText, 500);
            setTimeout(fixBadgeText, 1500);
            setTimeout(fixBadgeText, 3000);
            setTimeout(fixImageLabels, 500);
            setTimeout(fixImageLabels, 1500);
            setTimeout(fixImageLabels, 3000);
            setTimeout(fixTableHeaders, 500);
            setTimeout(fixTableHeaders, 1500);
            setTimeout(fixTableHeaders, 3000);
            setTimeout(fixDropText, 500);
            setTimeout(fixDropText, 1500);
            setTimeout(fixExamples, 500);
            setTimeout(fixExamples, 1500);
            setTimeout(fixExamples, 3000);
            setTimeout(fixGalleryCaptions, 500);
            setTimeout(fixGalleryCaptions, 1500);
            setTimeout(fixGalleryCaptions, 3000);
            setTimeout(fixCorners, 500);
            setTimeout(fixCorners, 1500);
            setTimeout(fixCorners, 3000);
            // Re-run fixes when any tab is clicked (catches lazy-rendered tab content)
            setTimeout(function() {
                document.querySelectorAll('.tab-nav button').forEach(function(btn) {
                    btn.addEventListener('click', function() {
                        setTimeout(fixBadgeText, 300);
                        setTimeout(fixBadgeText, 800);
                        setTimeout(fixImageLabels, 300);
                        setTimeout(fixImageLabels, 800);
                        setTimeout(fixDropText, 300);
                        setTimeout(fixDropText, 800);
                        setTimeout(fixCorners, 300);
                        setTimeout(fixTableHeaders, 300);
                        setTimeout(fixTableHeaders, 800);
                    });
                });
            }, 1000);
            // MutationObserver — re-apply tab colors whenever Gradio re-renders
            function fixTabColors() {
                document.querySelectorAll('button[role="tab"]').forEach(function(btn) {
                    if (btn.getAttribute('aria-selected') === 'true') {
                        btn.style.setProperty('background', '#2e7d32', 'important');
                        btn.style.setProperty('color', '#ffffff', 'important');
                        btn.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
                        btn.style.setProperty('border-color', '#2e7d32', 'important');
                    } else {
                        btn.style.setProperty('background', '#f1f8f1', 'important');
                        btn.style.setProperty('color', '#2e7d32', 'important');
                        btn.style.setProperty('-webkit-text-fill-color', '#2e7d32', 'important');
                        btn.style.setProperty('border', '1px solid #c8e6c9', 'important');
                    }
                    btn.style.setProperty('border-bottom', 'none', 'important');
                    btn.style.setProperty('box-shadow', 'none', 'important');
                    btn.style.setProperty('border-radius', '6px 6px 0 0', 'important');
                });
            }
            fixTabColors();
            fixTableHeaders();
            var tabObserver = new MutationObserver(function() { fixTabColors(); });
            tabObserver.observe(document.body, { subtree: true, attributes: true, attributeFilter: ['aria-selected', 'class'] });
            var tableObserver = new MutationObserver(function() { fixGradioSpecific(); fixTableHeaders(); fixExamples(); fixGalleryCaptions(); });
            tableObserver.observe(document.body, { subtree: true, childList: true });
            return [];
        }"""
    )

# ── LAUNCH ────────────────────────────────────────────────────────────────────
demo.launch(
    server_name="0.0.0.0",
    share=False,
    debug=False,
)