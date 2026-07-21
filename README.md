# PSD Mockup Processing Engine & Interactive Editor

An advanced, high-fidelity Python engine for processing multi-layer Photoshop (`.psd` / `.psb`) mockup templates. Featuring an interactive 2D Smart Object design editor, bicubic Bezier mesh warping, raster layer mask extraction, and Photoshop Z-order layer blending.

---

## 🌟 Key Features

* **Interactive 2D Smart Object Design Editor (`gui.py`)**:
  * Real-time 2D canvas to load, scale (slider/scroll wheel), drag, position, center, fit, or fill design images onto Smart Object dimensions.
  * Preserves design transparency and native dimensions without unwanted stretching.
* **Bicubic Bezier Mesh Warping Engine (`warp_engine.py`)**:
  * Reads Photoshop `quiltWarp` and `customEnvelopeWarp` descriptor blocks (`meshPoints`, `quiltSliceX/Y`).
  * Evaluates multi-slice bicubic Bezier patches with piecewise homography remapping for 99%+ visual fidelity on organic or curved surfaces (mugs, t-shirts, apparel).
* **Photoshop Z-Order Layer Compositing (`renderer.py`)**:
  * Processes PSD layer trees strictly in Photoshop visual index order (bottom-to-top).
  * Automatically renders garment folds, textures, shadows, and highlight overlays positioned above Smart Objects.
* **Native Photoshop Blend Modes (`blending.py`)**:
  * Implements `Multiply`, `Overlay`, `Linear Burn`, `Screen`, `Soft Light`, and `Normal` blend functions with layer-level un-multiplied opacity calculations.
* **Layer Mask & Clipping Stencil Support (`psd_parser.py`)**:
  * Applies per-layer raster masks and clipping stencils (`last_base_alpha`) to bound warped designs seamlessly within product contours.

---

## 📁 Package Architecture (`psd-processing-tinker/`)

The core module is modularly organized inside `psd-processing-tinker/`:

```
psd-processing-tinker/
├── __init__.py        # Package exports & public API
├── main.py            # GUI entry point launcher
├── gui.py             # Tkinter GUI & Interactive 2D Design Editor
├── renderer.py        # Z-order PSD multi-layer rendering pipeline
├── warp_engine.py     # Bezier patch evaluation & homography warping engine
├── psd_parser.py      # Photoshop descriptor parsing & tagged block handling
└── blending.py        # Photoshop blend modes & opacity composition
```

### Module Breakdown

| Module | Description |
| :--- | :--- |
| `main.py` | Standalone launcher to start the interactive Tkinter application. |
| `gui.py` | Tkinter user interface featuring the interactive 2D Smart Object canvas, zoom slider, preset placement controls, and 3D preview output window. |
| `renderer.py` | Main orchestrator (`render_full_mockup`) that iterates through layers, applies design warping, processes clipping masks, and blends top-layer overlays. |
| `warp_engine.py` | Contains `warp_and_place` and `fit_design` for high-resolution Bezier patch mesh remapping with edge antialiasing. |
| `psd_parser.py` | Extracts `Trnf` transform quads, local Smart Object dimensions, mesh points, and raster masks from `psd-tools` tagged blocks. |
| `blending.py` | Pure NumPy/Pillow implementations of Photoshop blending modes and opacity math. |

---

## 🚀 Installation & Setup

### Prerequisites
* Python **3.10** or higher
* `tkinter` (usually included with standard Python installations)

### Dependencies
Install required packages:

```bash
pip install psd-tools opencv-python-headless numpy pillow attrs
```

---

## 💻 Usage

### 1. Running the Interactive GUI Application

Launch the desktop interface directly:

```bash
python psd-processing-tinker/main.py
```

#### Workflow Steps in GUI:
1. **Select PSD Template**: Click **Browse PSD...** to load any PSD/PSB file containing Smart Objects (e.g., T-shirt or Mug templates).
2. **Interactive 2D Design Editor**: Click **Select Design Image...** to load your graphics asset. Use mouse drag to position, mouse wheel / slider to scale, or click **Center / Fit / Fill** presets.
3. **Generate Mockup**: Click **GENERATE MOCKUP PREVIEW** to execute full-resolution mesh warping and layer blending. The result is saved automatically to `mockup_output.png`.

---

### 2. Programmatic Python API

You can also import and use the rendering engine directly in Python scripts:

```python
import sys
sys.path.insert(0, 'psd-processing-tinker')

from psd_tools import PSDImage
from PIL import Image
from renderer import render_full_mockup

# Load PSD template
psd = PSDImage.open('T-shirt Female White/T-shirt_White_Front_Female.psd')

# Load custom design image
design_img = Image.open('my_design.png')

# Render full mockup in proper Z-order
output_image = render_full_mockup(
    psd=psd,
    replacement_input=design_img,
    fit_mode='fill',
    use_mask=True
)

# Save high-fidelity mockup output
output_image.save('output_mockup.png')
```

---

## 🛠 Tech Stack

* **Language**: Python 3
* **PSD Parsing**: `psd-tools`
* **Mesh Remapping**: `opencv-python-headless`, `NumPy`
* **Image Processing & Blending**: `Pillow` (PIL)
* **GUI Framework**: `Tkinter`
