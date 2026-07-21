"""
genarate_mockup_tinker.py

Advanced PSD Mockup Generator & Realtime Preview Engine

Combines:
  1. psd-tools descriptor extraction (from analyze_psd_smartobject.py): Reads Trnf, Sz,
     quiltWarp/warp descriptors, meshPoints (Hrzn/Vrtc), quiltSliceX/Y, layer masks, opacities,
     and blend modes directly from tagged blocks (SMART_OBJECT_LAYER_DATA1/2, PLACED_LAYER1/2).
  2. Multi-slice bicubic Bezier patch evaluation & per-cell homography warping with
     high-resolution supersampled remap & anti-aliased edge smoothing.
  3. Multi-layer PSD compositing architecture (from final_multi_layer_smooth_children_no_color.py):
     Processes the layer tree in proper z-order, preserving background layers, clipped layers
     (stencil masks), layer masks, opacities, and Photoshop blend modes.
  4. Interactive Tkinter GUI supporting multi-smart-object selection, replacement image mapping,
     fit modes (fill, fit, stretch, none), and live canvas preview.

Requires: pip install psd-tools opencv-python-headless numpy pillow attrs
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageChops
from psd_tools import PSDImage
from psd_tools.constants import Tag

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import attr as _attr_module
except ImportError:
    _attr_module = None


# ==========================================================================
# Descriptor & Structure Parsing Helpers
# ==========================================================================
def jsonable(obj, depth=0, max_depth=20, _seen=None):
    if _seen is None:
        _seen = set()
    if depth > max_depth:
        return "<max depth>"
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    obj_id = id(obj)
    if obj_id in _seen:
        return "<circular ref>"
    if isinstance(obj, dict):
        _seen.add(obj_id)
        return {(k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)):
                 jsonable(v, depth + 1, max_depth, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        _seen.add(obj_id)
        return [jsonable(v, depth + 1, max_depth, _seen) for v in obj]
    if _attr_module is not None and _attr_module.has(type(obj)):
        _seen.add(obj_id)
        try:
            out = {"__type__": type(obj).__name__}
            for f in _attr_module.fields(type(obj)):
                try:
                    out[f.name] = jsonable(getattr(obj, f.name), depth + 1, max_depth, _seen)
                except Exception as e:
                    out[f.name] = f"<error: {e}>"
            return out
        except Exception:
            pass
    if hasattr(obj, "value") and not callable(getattr(obj, "value")):
        try:
            return jsonable(obj.value, depth + 1, max_depth, _seen)
        except Exception:
            pass
    if hasattr(obj, "__dict__") and vars(obj):
        _seen.add(obj_id)
        return jsonable(vars(obj), depth + 1, max_depth, _seen)
    slots = getattr(obj, "__slots__", None)
    if slots:
        _seen.add(obj_id)
        out = {"__type__": type(obj).__name__}
        for s in slots:
            try:
                out[s] = jsonable(getattr(obj, s), depth + 1, max_depth, _seen)
            except Exception:
                pass
        return out
    if hasattr(obj, "items"):
        try:
            return jsonable(dict(obj.items()), depth + 1, max_depth, _seen)
        except Exception:
            pass
    return str(obj)


def resolve(node):
    if isinstance(node, dict):
        t = node.get("__type__")
        if t in ("Double", "Integer", "UnitFloat"):
            return node["value"]
        if t == "Enumerated":
            return node["enum"]
        if t == "List":
            return [resolve(x) for x in node["_items"]]
        if t == "UnitFloats":
            return node["values"]
        if t in ("Descriptor", "DescriptorBlock", "DescriptorBlock2"):
            return {k: resolve(v) for k, v in node["_items"].items()}
        if t == "ObjectArray":
            inner = node["_items"]
            if len(inner) == 1:
                return resolve(next(iter(inner.values())))
            return {k: resolve(v) for k, v in inner.items()}
        if t and "_items" in node:
            return resolve(node["_items"])
        if t and "value" in node:
            return resolve(node["value"])
        return {k: resolve(v) for k, v in node.items() if k != "__type__"}
    if isinstance(node, list):
        return [resolve(x) for x in node]
    return node


def get_flat_layer_list(container):
    """Recursively collect layers in bottom-to-top execution order."""
    layers = []
    for layer in container:
        if hasattr(layer, 'is_group') and layer.is_group():
            layers.extend(get_flat_layer_list(layer))
        else:
            layers.append(layer)
    return layers


def list_smart_object_layers(psd):
    return [l for l in get_flat_layer_list(psd) if getattr(l, "kind", None) == "smartobject"]


# ==========================================================================
# PSD Smart Object Descriptor Extraction
# ==========================================================================
def extract_smart_object_data(psd, target_layer):
    tagged_blocks = getattr(target_layer._record, "tagged_blocks", {})
    
    # Try multiple possible smart object / placed layer tagged block identifiers
    so_block = None
    for tag_key in [Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2,
                    Tag.PLACED_LAYER2, Tag.PLACED_LAYER1]:
        if tag_key in tagged_blocks:
            so_block = tagged_blocks[tag_key]
            break

    if so_block is None:
        raise ValueError(f"No smart object descriptor block found on layer '{target_layer.name}'.")

    resolved_top = resolve(jsonable(so_block.data))
    resolved = resolved_top.get("data", resolved_top)
    if "Trnf" not in resolved:
        raise KeyError(f"'Trnf' transform quad not found in layer '{target_layer.name}'. Keys: {list(resolved.keys())}")

    trnf = resolved["Trnf"]
    sz = resolved.get("Sz  ") or resolved.get("Sz") or {"Wdth": psd.width, "Hght": psd.height}
    
    quilt = resolved.get("quiltWarp")
    warp_desc = resolved.get("warp")

    # Determine warp dictionary source (quiltWarp takes precedence)
    warp_source = None
    if quilt and isinstance(quilt, dict) and quilt.get("warpStyle") == "warpCustom":
        warp_source = quilt
    elif warp_desc and isinstance(warp_desc, dict) and warp_desc.get("warpStyle") == "warpCustom":
        warp_source = warp_desc

    opacity_val = target_layer.opacity / 255.0 if target_layer.opacity > 1 else target_layer.opacity
    raw_trnf_quad = np.array(trnf, dtype=np.float32).reshape(4, 2)
    local_w, local_h = float(sz.get("Wdth", psd.width)), float(sz.get("Hght", psd.height))

    result = {
        "canvas_size": (psd.width, psd.height),
        "trnf_quad": raw_trnf_quad,
        "local_size": (local_w, local_h),
        "blend_mode": str(target_layer.blend_mode),
        "opacity": opacity_val,
        "layer": target_layer,
    }

    if warp_source:
        bounds = warp_source.get("bounds", {"Left": 0, "Top ": 0, "Rght": local_w, "Btom": local_h})
        env = warp_source.get("customEnvelopeWarp", {})
        mesh = env.get("meshPoints", {})
        
        # Handle mesh points formats (dict with Hrzn/Vrtc or horizontal/vertical lists)
        hrzn = mesh.get("Hrzn") or next((m["values"] for m in mesh if isinstance(m, dict) and m.get("type") == "horizontal"), None)
        vrtc = mesh.get("Vrtc") or next((m["values"] for m in mesh if isinstance(m, dict) and m.get("type") == "vertical"), None)
        
        q_slice_x = env.get("quiltSliceX", [bounds.get("Left", 0), bounds.get("Rght", local_w)])
        q_slice_y = env.get("quiltSliceY", [bounds.get("Top ", 0), bounds.get("Btom", local_h)])

        dst_x_arr = np.array(hrzn, dtype=np.float64) if hrzn is not None else None
        dst_y_arr = np.array(vrtc, dtype=np.float64) if vrtc is not None else None

        result["mesh"] = {
            "rows": int(warp_source.get("deformNumRows", 4)),
            "cols": int(warp_source.get("deformNumCols", 4)),
            "u_order": int(warp_source.get("uOrder", 4)),
            "v_order": int(warp_source.get("vOrder", 4)),
            "bounds": (bounds.get("Left", 0), bounds.get("Top ", 0), bounds.get("Rght", 0), bounds.get("Btom", 0)),
            "dst_x": dst_x_arr,
            "dst_y": dst_y_arr,
            "quilt_slice_x": np.array(q_slice_x, dtype=np.float64),
            "quilt_slice_y": np.array(q_slice_y, dtype=np.float64),
        }

        # Correct trnf_quad placement using deform envelope bounds
        if dst_x_arr is not None and dst_y_arr is not None:
            min_x, max_x = float(dst_x_arr.min()), float(dst_x_arr.max())
            min_y, max_y = float(dst_y_arr.min()), float(dst_y_arr.max())
            if max_x > min_x and max_y > min_y:
                mesh_bounds_quad = np.float32([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
                local_size_quad = np.float32([[0, 0], [local_w, 0], [local_w, local_h], [0, local_h]])
                M_mesh_to_canvas = cv2.getPerspectiveTransform(mesh_bounds_quad, raw_trnf_quad)
                result["trnf_quad"] = cv2.perspectiveTransform(local_size_quad.reshape(-1, 1, 2), M_mesh_to_canvas).reshape(4, 2)
    else:
        result["mesh"] = None

    return result


def extract_layer_mask(layer, canvas_size):
    mask = getattr(layer, "mask", None)
    if mask is None or not getattr(mask, "is_enabled", True):
        return None
    cw, ch = canvas_size
    bg_val = int(getattr(mask, "background_color", 0))
    canvas_mask = np.full((ch, cw), bg_val, dtype=np.uint8)
    
    try:
        mask_img = mask.topil()
    except Exception:
        mask_img = None

    if mask_img is None:
        return canvas_mask

    mx0, my0, mx1, my1 = mask.bbox
    mask_arr = np.array(mask_img.convert("L"))

    cx0, cy0 = max(mx0, 0), max(my0, 0)
    cx1, cy1 = min(mx1, cw), min(my1, ch)
    if cx1 <= cx0 or cy1 <= cy0:
        return canvas_mask

    sx0, sy0 = cx0 - mx0, cy0 - my0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    canvas_mask[cy0:cy1, cx0:cx1] = mask_arr[sy0:sy1, sx0:sx1]
    return canvas_mask


# ==========================================================================
# Image Fitting Utilities
# ==========================================================================
def fit_design(design_img: Image.Image, target_w: float, target_h: float, mode="fill"):
    design_img = design_img.convert("RGBA")
    dw, dh = design_img.size
    target_w_i, target_h_i = int(round(target_w)), int(round(target_h))

    if mode == "none":
        canvas = Image.new("RGBA", (target_w_i, target_h_i), (0, 0, 0, 0))
        crop_w, crop_h = min(dw, target_w_i), min(dh, target_h_i)
        cropped = design_img.crop((0, 0, crop_w, crop_h))
        canvas.paste(cropped, (0, 0), cropped)
        return canvas

    if mode == "stretch":
        return design_img.resize((target_w_i, target_h_i), Image.LANCZOS)

    scale = max(target_w_i / dw, target_h_i / dh) if mode == "fill" else min(target_w_i / dw, target_h_i / dh)
    new_w, new_h = max(1, round(dw * scale)), max(1, round(dh * scale))
    resized = design_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (target_w_i, target_h_i), (0, 0, 0, 0))
    if mode == "fill":
        left = (new_w - target_w_i) // 2
        top = (new_h - target_h_i) // 2
        resized = resized.crop((left, top, left + target_w_i, top + target_h_i))
        canvas.paste(resized, (0, 0))
    else:
        offset = ((target_w_i - new_w) // 2, (target_h_i - new_h) // 2)
        canvas.paste(resized, offset, resized)
    return canvas


# ==========================================================================
# Warp Engine: Multi-slice Bicubic Bezier & Homography Remapping
# ==========================================================================
def _bernstein(i, t):
    return [(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3][i]


def _bezier_patch_eval(u_grid, v_grid, ctrl_x, ctrl_y):
    Bu = np.array([[_bernstein(j, u) for j in range(4)] for u in u_grid])   # (nu, 4)
    Bv = np.array([[_bernstein(i, v) for i in range(4)] for v in v_grid])   # (nv, 4)
    cx = np.array(ctrl_x, dtype=np.float64).reshape(4, 4)
    cy = np.array(ctrl_y, dtype=np.float64).reshape(4, 4)
    dst_x = np.einsum('ia,jb,ab->ij', Bv, Bu, cx)
    dst_y = np.einsum('ia,jb,ab->ij', Bv, Bu, cy)
    return dst_x, dst_y


def _fill_patch_map(map_x, map_y, ctrl_x, ctrl_y, src_x0, src_y0, src_x1, src_y1,
                     canvas_w_s, canvas_h_s, img_w, img_h, subdiv=24):
    grid = np.linspace(0.0, 1.0, subdiv)
    dst_x, dst_y = _bezier_patch_eval(grid, grid, ctrl_x, ctrl_y)

    for i in range(subdiv - 1):
        for j in range(subdiv - 1):
            dst_q = np.array([
                [dst_x[i, j],     dst_y[i, j]],
                [dst_x[i, j + 1], dst_y[i, j + 1]],
                [dst_x[i + 1, j + 1], dst_y[i + 1, j + 1]],
                [dst_x[i + 1, j], dst_y[i + 1, j]],
            ], dtype=np.float32)

            su0 = src_x0 + (j / (subdiv - 1)) * (src_x1 - src_x0)
            su1 = src_x0 + ((j + 1) / (subdiv - 1)) * (src_x1 - src_x0)
            sv0 = src_y0 + (i / (subdiv - 1)) * (src_y1 - src_y0)
            sv1 = src_y0 + ((i + 1) / (subdiv - 1)) * (src_y1 - src_y0)
            src_q = np.array([[su0, sv0], [su1, sv0], [su1, sv1], [su0, sv1]], dtype=np.float32)

            H, _ = cv2.findHomography(dst_q, src_q)
            if H is None:
                continue

            min_c = np.floor(dst_q.min(axis=0)).astype(int)
            max_c = np.ceil(dst_q.max(axis=0)).astype(int)
            min_x, min_y = max(0, min_c[0]), max(0, min_c[1])
            max_x, max_y = min(canvas_w_s, max_c[0]), min(canvas_h_s, max_c[1])
            if max_x <= min_x or max_y <= min_y:
                continue

            xs, ys = np.meshgrid(np.arange(min_x, max_x, dtype=np.float64),
                                  np.arange(min_y, max_y, dtype=np.float64))
            ones = np.ones_like(xs)
            pts = np.stack([xs, ys, ones], axis=-1)
            src_pts = pts @ H.T
            w = src_pts[..., 2]
            valid_w = np.abs(w) > 1e-9
            safe_w = np.where(valid_w, w, 1.0)
            sx = src_pts[..., 0] / safe_w
            sy = src_pts[..., 1] / safe_w
            inside = valid_w & (sx >= 0) & (sx < img_w) & (sy >= 0) & (sy < img_h)

            mx_slice = map_x[min_y:max_y, min_x:max_x]
            my_slice = map_y[min_y:max_y, min_x:max_x]
            mx_slice[inside] = sx[inside]
            my_slice[inside] = sy[inside]


def warp_and_place(local_img: Image.Image, mesh: dict, trnf_quad: np.ndarray,
                    local_size, canvas_size, supersample=2, subdiv=24):
    lw, lh = local_size
    cw, ch = canvas_size
    img_arr = np.array(local_img)
    img_h, img_w = img_arr.shape[:2]

    local_quad = np.float32([[0, 0], [lw, 0], [lw, lh], [0, lh]])
    M = cv2.getPerspectiveTransform(local_quad, trnf_quad.astype(np.float32))

    cw_s, ch_s = int(round(cw * supersample)), int(round(ch * supersample))
    map_x = np.full((ch_s, cw_s), -1, dtype=np.float64)
    map_y = np.full((ch_s, cw_s), -1, dtype=np.float64)

    if mesh is None or mesh.get("dst_x") is None or mesh.get("dst_y") is None:
        # Fallback to perspective placement via inverse homography mapping
        Ms = M.copy()
        Ms[0, :] *= supersample
        Ms[1, :] *= supersample
        Minv = np.linalg.inv(Ms)
        xs, ys = np.meshgrid(np.arange(cw_s, dtype=np.float64), np.arange(ch_s, dtype=np.float64))
        ones = np.ones_like(xs)
        pts = np.stack([xs, ys, ones], axis=-1)
        src_pts = pts @ Minv.T
        w = src_pts[..., 2]
        valid_w = np.abs(w) > 1e-9
        safe_w = np.where(valid_w, w, 1.0)
        sx = src_pts[..., 0] / safe_w
        sy = src_pts[..., 1] / safe_w
        inside = valid_w & (sx >= 0) & (sx < img_w) & (sy >= 0) & (sy < img_h)
        map_x[inside] = sx[inside]
        map_y[inside] = sy[inside]
    else:
        rows, cols = mesh["rows"], mesh["cols"]
        qsx = mesh["quilt_slice_x"]
        qsy = mesh["quilt_slice_y"]
        n_xslices = max(1, len(qsx) - 1)
        n_yslices = max(1, len(qsy) - 1)

        pts_local = np.stack([mesh["dst_x"], mesh["dst_y"]], axis=1).astype(np.float32).reshape(-1, 1, 2)
        pts_canvas = cv2.perspectiveTransform(pts_local, M).reshape(-1, 2) * supersample
        mesh_cx = pts_canvas[:, 0]
        mesh_cy = pts_canvas[:, 1]

        for m in range(n_yslices):
            for k in range(n_xslices):
                row_idx = [3 * m + i for i in range(4)]
                col_idx = [3 * k + j for j in range(4)]
                
                # Boundary safety check for patch control indices
                if max(row_idx) >= rows or max(col_idx) >= cols:
                    continue

                ctrl_x = np.array([mesh_cx[r * cols + c] for r in row_idx for c in col_idx])
                ctrl_y = np.array([mesh_cy[r * cols + c] for r in row_idx for c in col_idx])

                src_x0, src_x1 = qsx[k], qsx[k + 1]
                src_y0, src_y1 = qsy[m], qsy[m + 1]

                _fill_patch_map(map_x, map_y, ctrl_x, ctrl_y,
                                 src_x0, src_y0, src_x1, src_y1,
                                 cw_s, ch_s, img_w, img_h, subdiv=subdiv)

    map_x32 = map_x.astype(np.float32)
    map_y32 = map_y.astype(np.float32)
    warped_hi = cv2.remap(img_arr, map_x32, map_y32, interpolation=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    warped = cv2.resize(warped_hi, (cw, ch), interpolation=cv2.INTER_AREA)

    if warped.shape[2] == 4:
        r, g, b, a = cv2.split(warped)
        a = cv2.GaussianBlur(a, (3, 3), 0)
        warped = cv2.merge([r, g, b, a])

    return Image.fromarray(warped, mode="RGBA")


# ==========================================================================
# Photoshop Blend Modes Implementation
# ==========================================================================
def _normal(base, blend):
    return blend

def _multiply(base, blend):
    return ((base.astype(np.uint16) * blend.astype(np.uint16)) // 255).astype(np.uint8)

def _linear_burn(base, blend):
    return np.clip(base.astype(np.int16) + blend.astype(np.int16) - 255, 0, 255).astype(np.uint8)

def _screen(base, blend):
    return (255 - ((255 - base.astype(np.uint16)) * (255 - blend.astype(np.uint16)) // 255)).astype(np.uint8)

def _overlay(base, blend):
    b = base.astype(np.float32) / 255.0
    f = blend.astype(np.float32) / 255.0
    res = np.where(b < 0.5, 2.0 * b * f, 1.0 - 2.0 * (1.0 - b) * (1.0 - f))
    return np.clip(res * 255.0, 0, 255).astype(np.uint8)

def _soft_light(base, blend):
    b = base.astype(np.float32) / 255.0
    f = blend.astype(np.float32) / 255.0
    res = np.where(f < 0.5, b - (1.0 - 2.0 * f) * b * (1.0 - b),
                   b + (2.0 * f - 1.0) * (np.sqrt(b) - b))
    return np.clip(res * 255.0, 0, 255).astype(np.uint8)

BLEND_FUNCS = {
    "normal": _normal,
    "multiply": _multiply,
    "linear burn": _linear_burn,
    "linearburn": _linear_burn,
    "screen": _screen,
    "overlay": _overlay,
    "soft light": _soft_light,
    "softlight": _soft_light,
}

def normalize_blend_mode(mode_str):
    s = str(mode_str).lower().replace("blendmode.", "").replace("_", " ").replace("-", " ").strip()
    return s

def apply_blend_mode(background: Image.Image, foreground: Image.Image, mode_str: str, opacity: float = 1.0):
    bg = np.array(background.convert("RGBA")).astype(np.uint8)
    fg = np.array(foreground.convert("RGBA")).astype(np.uint8)
    
    norm_mode = normalize_blend_mode(mode_str)
    blend_fn = BLEND_FUNCS.get(norm_mode, _normal)
    
    blended_rgb = blend_fn(bg[..., :3], fg[..., :3]) if blend_fn is not _normal else fg[..., :3]
    
    fg_alpha = (fg[..., 3:4].astype(np.float32) / 255.0) * opacity
    bg_alpha = bg[..., 3:4].astype(np.float32) / 255.0

    out_rgb = bg[..., :3].astype(np.float32) * (1.0 - fg_alpha) + blended_rgb.astype(np.float32) * fg_alpha
    out_alpha = (fg_alpha + bg_alpha * (1.0 - fg_alpha)) * 255.0
    
    out = np.dstack([np.clip(out_rgb, 0, 255).astype(np.uint8), np.clip(out_alpha, 0, 255).astype(np.uint8)])
    return Image.fromarray(out, mode="RGBA")


# ==========================================================================
# Multi-Layer Mockup Orchestrator
# ==========================================================================
def render_full_mockup(psd, replacement_input, fit_mode="fill", use_mask=True,
                        progress_callback=None):
    """
    Renders the full PSD layer tree in proper z-order.
    replacement_input: Can be a single design image (str or Image.Image) applied to ALL
                       smart objects, or a dict mapping layer_name -> image.
    """
    canvas_w, canvas_h = psd.width, psd.height
    accumulated_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    flat_layers = get_flat_layer_list(psd)
    total_layers = len(flat_layers)

    last_base_alpha = None

    # Handle replacement input format (single image or dict)
    if isinstance(replacement_input, dict):
        replacement_map = replacement_input
        default_design = replacement_input.get("__default__")
    else:
        replacement_map = {}
        default_design = replacement_input

    for idx, layer in enumerate(flat_layers):
        if not getattr(layer, "visible", True):
            continue

        layer_name = layer.name
        if progress_callback:
            progress_callback(idx + 1, total_layers, layer_name)

        is_clipped = getattr(layer, "clipping", False)
        blend_mode = str(layer.blend_mode)
        opacity = layer.opacity / 255.0 if layer.opacity > 1 else layer.opacity
        eff_opacity = opacity

        # Check if layer is a smart object with a replacement image assigned
        is_so = getattr(layer, "kind", None) == "smartobject"
        design_img = replacement_map.get(layer_name, default_design) if is_so else None

        if is_so and design_img is not None:
            # 1. Warp user replacement design
            if isinstance(design_img, str):
                design_img = Image.open(design_img)

            so_data = extract_smart_object_data(psd, layer)
            lw, lh = so_data["local_size"]
            local_fitted = fit_design(design_img, lw, lh, mode=fit_mode)

            layer_rendered = warp_and_place(local_fitted, so_data["mesh"], so_data["trnf_quad"],
                                            so_data["local_size"], so_data["canvas_size"])

            # 2. Apply layer's raster mask if enabled
            if use_mask:
                mask_arr = extract_layer_mask(layer, (canvas_w, canvas_h))
                if mask_arr is not None:
                    arr = np.array(layer_rendered)
                    arr[..., 3] = (arr[..., 3].astype(np.float32) * (mask_arr.astype(np.float32) / 255.0)).astype(np.uint8)
                    layer_rendered = Image.fromarray(arr, mode="RGBA")

        else:
            # Render standard PSD layer content (overlays, shadows, background, etc.)
            layer_rendered = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            bbox = layer.bbox
            try:
                topil_img = layer.topil()
                if topil_img is not None and bbox and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
                    if topil_img.mode != "RGBA":
                        topil_img = topil_img.convert("RGBA")
                    layer_rendered.paste(topil_img, (bbox[0], bbox[1]), topil_img)
                    eff_opacity = opacity  # Raw alpha from topil needs layer opacity applied
                else:
                    psd_comp = layer.composite()
                    if psd_comp is not None and bbox and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
                        if psd_comp.mode != "RGBA":
                            psd_comp = psd_comp.convert("RGBA")
                        layer_rendered.paste(psd_comp, (bbox[0], bbox[1]), psd_comp)
                        eff_opacity = 1.0  # Opacity already baked into composite alpha
            except Exception:
                try:
                    psd_comp = layer.composite()
                    if psd_comp is not None and bbox and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
                        if psd_comp.mode != "RGBA":
                            psd_comp = psd_comp.convert("RGBA")
                        layer_rendered.paste(psd_comp, (bbox[0], bbox[1]), psd_comp)
                        eff_opacity = 1.0
                except Exception:
                    pass

        # Handle stencil clipping mask
        if is_clipped and last_base_alpha is not None:
            r, g, b, a = layer_rendered.split()
            new_a = ImageChops.multiply(a, last_base_alpha)
            layer_rendered = Image.merge("RGBA", (r, g, b, new_a))
        elif not is_clipped:
            last_base_alpha = layer_rendered.getchannel("A")

        # Blend layer into master accumulated canvas in strict z-order
        accumulated_canvas = apply_blend_mode(accumulated_canvas, layer_rendered, blend_mode, eff_opacity)

    return accumulated_canvas.convert("RGB")


# ==========================================================================
# Tkinter GUI Application with Interactive 2D Design Editor
# ==========================================================================
class MockupApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSD Smart Object Interactive Mockup Engine")
        self.root.geometry("1100x940")
        self.root.configure(bg="#f3f4f6")

        self.psd = None
        self.psd_path = None
        self.smart_layers = []

        # Smart object canvas dimensions (detected from PSD)
        self.editor_w = 1187.0
        self.editor_h = 475.0
        self.editor_display_scale = 1.0

        # User design image state
        self.user_img = None
        self.user_img_path = None
        self.img_scale = 1.0        # 1.0 = 100% original size
        self.img_offset_x = 0.0     # Position on smart object canvas
        self.img_offset_y = 0.0

        # Dragging state
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_start_off_x = 0.0
        self.drag_start_off_y = 0.0

        # Photo references
        self.editor_photo = None
        self.preview_photo = None

        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#f3f4f6")
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        tk.Label(main_frame, text="PSD Layered Mockup Engine", font=("Arial", 18, "bold"), bg="#f3f4f6").pack(pady=(0, 5))

        # 1. Select PSD File Frame
        ctrl_frame = tk.LabelFrame(main_frame, text=" 1. Select PSD Template ", font=("Arial", 11, "bold"), bg="#f3f4f6", padx=10, pady=6)
        ctrl_frame.pack(fill="x", pady=4)

        tk.Label(ctrl_frame, text="PSD File:", font=("Arial", 10, "bold"), bg="#f3f4f6").grid(row=0, column=0, sticky="w")
        tk.Button(ctrl_frame, text="Browse PSD...", command=self.load_psd, bg="#4f46e5", fg="white", font=("Arial", 9, "bold")).grid(row=0, column=1, padx=10)
        self.psd_status = tk.Label(ctrl_frame, text="No PSD loaded", fg="#ef4444", bg="#f3f4f6", font=("Arial", 10))
        self.psd_status.grid(row=0, column=2, sticky="w")

        self.mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl_frame, text="Apply Layer Raster Masks", variable=self.mask_var, bg="#f3f4f6").grid(row=0, column=3, padx=20, sticky="e")

        # 2. Interactive 2D Smart Object Design Editor Frame
        self.editor_frame = tk.LabelFrame(main_frame, text=" 2. Interactive Smart Object Design Editor ", font=("Arial", 11, "bold"), bg="#f3f4f6", padx=10, pady=8)
        self.editor_frame.pack(fill="x", pady=4)

        # Editor Top Control Bar
        top_bar = tk.Frame(self.editor_frame, bg="#f3f4f6")
        top_bar.pack(fill="x", pady=(0, 6))

        tk.Button(top_bar, text="Select Design Image...", command=self.select_design_image, bg="#2563eb", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=(0, 10))
        self.img_status_lbl = tk.Label(top_bar, text="No design image loaded", fg="#ef4444", bg="#f3f4f6", font=("Arial", 9, "bold"))
        self.img_status_lbl.pack(side="left")

        # Preset positioning buttons
        btn_box = tk.Frame(top_bar, bg="#f3f4f6")
        btn_box.pack(side="right")
        tk.Button(btn_box, text="Center", command=self.center_image, width=7).pack(side="left", padx=2)
        tk.Button(btn_box, text="Fit", command=self.fit_image, width=6).pack(side="left", padx=2)
        tk.Button(btn_box, text="Fill", command=self.fill_image, width=6).pack(side="left", padx=2)
        tk.Button(btn_box, text="Reset", command=self.reset_image, width=6).pack(side="left", padx=2)

        # Scale Control Slider Row
        scale_bar = tk.Frame(self.editor_frame, bg="#f3f4f6")
        scale_bar.pack(fill="x", pady=(0, 6))

        tk.Label(scale_bar, text="Scale (%):", font=("Arial", 9, "bold"), bg="#f3f4f6").pack(side="left", padx=(0, 5))
        self.scale_var = tk.DoubleVar(value=100)
        self.scale_slider = tk.Scale(scale_bar, from_=10, to=300, orient="horizontal", variable=self.scale_var,
                                     command=self.on_scale_slider_change, showvalue=True, length=250, bg="#f3f4f6")
        self.scale_slider.pack(side="left", padx=5)

        self.editor_dim_lbl = tk.Label(scale_bar, text="Canvas: 1187 x 475 px (Mouse drag to move, Scroll to zoom)",
                                       fg="#4b5563", bg="#f3f4f6", font=("Arial", 9))
        self.editor_dim_lbl.pack(side="right", padx=5)

        # 2D Interactive Editor Canvas
        self.editor_canvas = tk.Canvas(self.editor_frame, width=750, height=220, bg="#e5e7eb", highlightthickness=1)
        self.editor_canvas.pack(fill="x", pady=2)

        # Mouse bindings for dragging and scrolling inside 2D Editor
        self.editor_canvas.bind("<ButtonPress-1>", self.on_editor_click)
        self.editor_canvas.bind("<B1-Motion>", self.on_editor_drag)
        self.editor_canvas.bind("<MouseWheel>", self.on_editor_scroll)
        self.editor_canvas.bind("<Button-4>", self.on_editor_scroll)
        self.editor_canvas.bind("<Button-5>", self.on_editor_scroll)

        # 3. Output Mockup Preview & Execution Frame
        preview_frame = tk.LabelFrame(main_frame, text=" 3. Mockup Preview & Output ", font=("Arial", 11, "bold"), bg="#f3f4f6", padx=10, pady=6)
        preview_frame.pack(fill="both", expand=True, pady=4)

        # Generate Button & Progress
        act_bar = tk.Frame(preview_frame, bg="#f3f4f6")
        act_bar.pack(fill="x", pady=(0, 4))

        tk.Button(act_bar, text="GENERATE MOCKUP PREVIEW", command=self.process, bg="#10b981", fg="white",
                  font=("Arial", 11, "bold"), height=1).pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(act_bar, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack(side="right")

        self.progress_label = tk.Label(preview_frame, text="Ready", font=("Arial", 9), bg="#f3f4f6", fg="#4b5563")
        self.progress_label.pack(anchor="w")

        # 3D Final Canvas Preview
        self.preview_canvas = tk.Canvas(preview_frame, width=800, height=320, bg="#d1d5db", highlightthickness=1)
        self.preview_canvas.pack(fill="both", expand=True, pady=2)

    # ==========================================================================
    # PSD & Smart Object Dimension Detection
    # ==========================================================================
    def load_psd(self):
        path = filedialog.askopenfilename(filetypes=[("PSD/PSB Files", "*.psd *.psb")])
        if not path:
            return
        try:
            self.psd = PSDImage.open(path)
            self.psd_path = path
            self.smart_layers = list_smart_object_layers(self.psd)

            if not self.smart_layers:
                messagebox.showwarning("Warning", "No Smart Object layers detected in this PSD file.")
                self.psd_status.config(text="Loaded (0 Smart Objects)", fg="#b45309")
            else:
                # Detect Smart Object canvas dimensions from first smart object layer
                so_data = extract_smart_object_data(self.psd, self.smart_layers[0])
                self.editor_w, self.editor_h = so_data["local_size"]

                so_names = ", ".join([f"'{l.name}'" for l in self.smart_layers])
                self.psd_status.config(
                    text=f"Loaded: {os.path.basename(path)} ({len(self.smart_layers)} SOs: {int(self.editor_w)}x{int(self.editor_h)}px)",
                    fg="#15803d"
                )

            # Re-center loaded design image on new smart object dimensions
            if self.user_img is not None:
                self.center_image()
            else:
                self.update_editor_preview()

            self.update_preview(self.psd.composite())

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PSD file:\n{e}")

    def select_design_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")])
        if not path:
            return
        try:
            self.user_img_path = path
            self.user_img = Image.open(path).convert("RGBA")
            self.img_scale = 1.0  # Default to 100% original scale
            self.scale_var.set(100)
            
            self.center_image()
            self.img_status_lbl.config(text=f"{os.path.basename(path)} ({self.user_img.width}x{self.user_img.height}px)", fg="#15803d")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load design image:\n{e}")

    # ==========================================================================
    # 2D Interactive Editor Controls & Positioning
    # ==========================================================================
    def center_image(self):
        if self.user_img is None:
            return
        scaled_w = self.user_img.width * self.img_scale
        scaled_h = self.user_img.height * self.img_scale
        self.img_offset_x = (self.editor_w - scaled_w) / 2.0
        self.img_offset_y = (self.editor_h - scaled_h) / 2.0
        self.update_editor_preview()

    def fit_image(self):
        if self.user_img is None:
            return
        scale_w = self.editor_w / self.user_img.width
        scale_h = self.editor_h / self.user_img.height
        self.img_scale = min(scale_w, scale_h)
        self.scale_var.set(round(self.img_scale * 100.0))
        self.center_image()

    def fill_image(self):
        if self.user_img is None:
            return
        scale_w = self.editor_w / self.user_img.width
        scale_h = self.editor_h / self.user_img.height
        self.img_scale = max(scale_w, scale_h)
        self.scale_var.set(round(self.img_scale * 100.0))
        self.center_image()

    def reset_image(self):
        if self.user_img is None:
            return
        self.img_scale = 1.0
        self.scale_var.set(100)
        self.center_image()

    def on_scale_slider_change(self, val):
        if self.user_img is None:
            return
        old_scale = self.img_scale
        new_scale = float(val) / 100.0
        if abs(new_scale - old_scale) > 1e-4:
            # Adjust offset to zoom relative to center of current image
            center_x = self.img_offset_x + (self.user_img.width * old_scale) / 2.0
            center_y = self.img_offset_y + (self.user_img.height * old_scale) / 2.0
            self.img_scale = new_scale
            self.img_offset_x = center_x - (self.user_img.width * new_scale) / 2.0
            self.img_offset_y = center_y - (self.user_img.height * new_scale) / 2.0
            self.update_editor_preview()

    def on_editor_click(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_start_off_x = self.img_offset_x
        self.drag_start_off_y = self.img_offset_y

    def on_editor_drag(self, event):
        if self.user_img is None or self.editor_display_scale <= 0:
            return
        dx = (event.x - self.drag_start_x) / self.editor_display_scale
        dy = (event.y - self.drag_start_y) / self.editor_display_scale
        self.img_offset_x = self.drag_start_off_x + dx
        self.img_offset_y = self.drag_start_off_y + dy
        self.update_editor_preview()

    def on_editor_scroll(self, event):
        if self.user_img is None:
            return
        delta = 0
        if hasattr(event, "delta") and event.delta != 0:
            delta = 5 if event.delta > 0 else -5
        elif getattr(event, "num", 0) == 4:
            delta = 5
        elif getattr(event, "num", 0) == 5:
            delta = -5

        if delta != 0:
            curr_val = self.scale_var.get()
            new_val = max(10, min(300, curr_val + delta))
            self.scale_var.set(new_val)
            self.on_scale_slider_change(new_val)

    # ==========================================================================
    # 2D Editor Canvas Rendering
    # ==========================================================================
    def update_editor_preview(self):
        ui_w = self.editor_canvas.winfo_width()
        ui_h = self.editor_canvas.winfo_height()

        if ui_w < 50 or ui_h < 50:
            ui_w, ui_h = 750, 220

        # Calculate scale factor to fit target (editor_w, editor_h) into ui_w x ui_h with padding
        padding = 10
        avail_w, avail_h = max(10, ui_w - 2 * padding), max(10, ui_h - 2 * padding)
        self.editor_display_scale = min(avail_w / self.editor_w, avail_h / self.editor_h)

        disp_w = max(1, int(round(self.editor_w * self.editor_display_scale)))
        disp_h = max(1, int(round(self.editor_h * self.editor_display_scale)))

        # Create transparent checkerboard background for 2D editor
        grid_size = 12
        checker_bg = Image.new("RGBA", (disp_w, disp_h), (240, 240, 240, 255))
        arr_check = np.array(checker_bg)
        for y in range(0, disp_h, grid_size):
            for x in range(0, disp_w, grid_size):
                if ((x // grid_size) + (y // grid_size)) % 2 == 1:
                    arr_check[y:min(y + grid_size, disp_h), x:min(x + grid_size, disp_w)] = [215, 215, 215, 255]
        checker_bg = Image.fromarray(arr_check, mode="RGBA")

        # Paste user design image if loaded
        if self.user_img is not None:
            scaled_w = max(1, int(round(self.user_img.width * self.img_scale * self.editor_display_scale)))
            scaled_h = max(1, int(round(self.user_img.height * self.img_scale * self.editor_display_scale)))
            
            scaled_user = self.user_img.resize((scaled_w, scaled_h), Image.LANCZOS)

            paste_x = int(round(self.img_offset_x * self.editor_display_scale))
            paste_y = int(round(self.img_offset_y * self.editor_display_scale))

            checker_bg.paste(scaled_user, (paste_x, paste_y), scaled_user)

        self.editor_photo = ImageTk.PhotoImage(checker_bg)
        self.editor_canvas.delete("all")

        cx, cy = ui_w // 2, ui_h // 2
        x0, y0 = cx - disp_w // 2, cy - disp_h // 2
        self.editor_canvas.create_image(cx, cy, image=self.editor_photo)

        # Draw red dashed outline around smart object boundary box
        self.editor_canvas.create_rectangle(x0, y0, x0 + disp_w, y0 + disp_h, outline="#0284c7", width=2, dash=(4, 4))

        # Update info text
        img_info = f"{os.path.basename(self.user_img_path)}" if self.user_img_path else "No Image"
        pos_info = f"Pos: ({int(round(self.img_offset_x))}, {int(round(self.img_offset_y))})"
        scale_info = f"Scale: {int(round(self.img_scale * 100))}%"
        self.editor_dim_lbl.config(
            text=f"Smart Object Canvas: {int(self.editor_w)}x{int(self.editor_h)}px  |  Image: {img_info}  |  {pos_info}  |  {scale_info}"
        )

    # ==========================================================================
    # Full-Resolution Composite Generation & Mockup Process
    # ==========================================================================
    def get_composite_design(self):
        """Generates full-resolution RGBA composite image of size (editor_w, editor_h)."""
        target_w_i, target_h_i = int(round(self.editor_w)), int(round(self.editor_h))
        full_canvas = Image.new("RGBA", (target_w_i, target_h_i), (0, 0, 0, 0))  # Transparent background

        if self.user_img is not None:
            scaled_w = max(1, int(round(self.user_img.width * self.img_scale)))
            scaled_h = max(1, int(round(self.user_img.height * self.img_scale)))
            scaled_user = self.user_img.resize((scaled_w, scaled_h), Image.LANCZOS)

            paste_x = int(round(self.img_offset_x))
            paste_y = int(round(self.img_offset_y))

            full_canvas.paste(scaled_user, (paste_x, paste_y), scaled_user)

        return full_canvas

    def process(self):
        if not self.psd:
            messagebox.showerror("Error", "Please load a PSD file first.")
            return

        if self.user_img is None:
            messagebox.showwarning("Warning", "Please select a design image first.")
            return

        def update_progress(current, total, layer_name):
            pct = (current / total) * 100
            self.progress_var.set(pct)
            self.progress_label.config(text=f"Processing layer {current}/{total}: {layer_name}")
            self.root.update_idletasks()

        try:
            self.progress_var.set(0)
            self.progress_label.config(text="Generating composite design and warping mockup...")
            self.root.update_idletasks()

            # Render 2D editor composite image
            composite_design = self.get_composite_design()

            # Warp composite design onto PSD smart objects using mode='none' (direct 1:1 mapping)
            final_img = render_full_mockup(self.psd, composite_design,
                                           fit_mode="none",
                                           use_mask=self.mask_var.get(),
                                           progress_callback=update_progress)

            out_dir = os.path.dirname(self.psd_path) if self.psd_path else "."
            out_path = os.path.join(out_dir, "mockup_output.png")
            final_img.save(out_path, quality=95)

            self.progress_var.set(100)
            self.progress_label.config(text=f"Complete! Saved -> {out_path}")
            self.update_preview(final_img)

        except Exception as e:
            self.progress_label.config(text="Processing Failed")
            messagebox.showerror("Error", f"Mockup generation failed:\n{e}")
            raise

    def update_preview(self, pil_img):
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()

        if canvas_w < 100 or canvas_h < 100:
            canvas_w, canvas_h = 800, 320

        img_w, img_h = pil_img.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_size = (max(1, int(img_w * ratio)), max(1, int(img_h * ratio)))

        preview_img = pil_img.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_photo)


if __name__ == "__main__":
    root = tk.Tk()
    app = MockupApp(root)
    root.mainloop()