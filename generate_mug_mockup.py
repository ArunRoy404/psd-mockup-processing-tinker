"""
generate_mug_mockup.py  (fixed)

Two bugs fixed from the previous version, both confirmed against the
actual PSD descriptor data you extracted:

1. WRONG MESH SOURCE GRID (this was the main cause of the design
   spilling off the right edge of the mug).
   Photoshop's "Custom Warp" (quiltWarp, warpStyle=warpCustom) is NOT a
   uniform grid. uOrder=vOrder=4 means each quilt slice is a cubic
   Bezier patch (4 control points per edge). quiltSliceX had 4 values
   -> 3 UNEVENLY sized slices (the last one ~4x wider than the first
   two). deformNumCols=10 only makes sense as
   (3 slices x 3 new points) + 1 shared start point - not 10 evenly
   spaced columns across the full width. The old code assumed uniform
   spacing, which desynced the source/destination correspondence and
   dragged front-surface pixels into what is actually the unwarped
   "back of the mug, not visible" region (columns 8-9, whose dst_x
   stay flat at ~931 / 1187 - i.e. off the visible mug edge).

2. NO LAYER MASK APPLIED (this is what let the overflow paint straight
   onto the background instead of being clipped).
   The tag list from analyze_psd_smartobject.py included
   LAYER_MASK_AS_GLOBAL_MASK, which only shows up when the layer has an
   actual raster layer mask. That mask - authored directly in canvas
   coordinates - is what Photoshop uses to clip the smart object to the
   mug's real printable silhouette (curvature falloff + handle gap).
   The old script never read layer.mask at all.

Usage is unchanged:
    python generate_mug_mockup.py \\
        --psd Mug/Mug_Thumbnail.psd \\
        --layer "Rectangle 1 copy" \\
        --design my_new_design.png \\
        --out result.png \\
        --fit fill

Requires: pip install psd-tools opencv-python-headless numpy pillow
"""
import argparse
import numpy as np
import cv2
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Tag

try:
    import attr as _attr_module
except ImportError:
    _attr_module = None


# --------------------------------------------------------------------------
# Step 0: generic descriptor -> plain python resolver (unchanged)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Step 1: pull warp/transform/placement/MASK data straight out of the PSD
# --------------------------------------------------------------------------
def extract_smart_object_data(psd_path, layer_name):
    psd = PSDImage.open(psd_path)
    target = None
    for layer in psd.descendants():
        if layer.name == layer_name:
            target = layer
            break
    if target is None:
        raise ValueError(f"Layer '{layer_name}' not found in {psd_path}")

    tagged_blocks = target._record.tagged_blocks
    so_block = tagged_blocks.get(Tag.SMART_OBJECT_LAYER_DATA1)
    if so_block is None:
        raise ValueError(
            f"No SMART_OBJECT_LAYER_DATA1 block on layer '{layer_name}' - "
            f"run analyze_psd_smartobject.py on this file to see what tags "
            f"ARE present and adapt this extractor."
        )
    resolved_top = resolve(jsonable(so_block.data))
    resolved = resolved_top.get("data", resolved_top)
    if "Trnf" not in resolved:
        raise KeyError(
            f"'Trnf' not found. Available keys at this level: {list(resolved.keys())}."
        )

    trnf = resolved["Trnf"]
    sz = resolved["Sz  "]
    quilt = resolved.get("quiltWarp")

    result = {
        "canvas_size": (psd.width, psd.height),
        "trnf_quad": np.array(trnf, dtype=np.float32).reshape(4, 2),
        "local_size": (float(sz["Wdth"]), float(sz["Hght"])),
        "blend_mode": str(target.blend_mode),
        "opacity": target.opacity / 255.0 if target.opacity > 1 else target.opacity,
        "layer": target,
        "psd": psd,
    }

    if quilt and quilt.get("warpStyle") == "warpCustom":
        bounds = quilt["bounds"]
        env = quilt["customEnvelopeWarp"]
        mesh = env["meshPoints"]
        rows = int(quilt["deformNumRows"])
        cols = int(quilt["deformNumCols"])
        result["mesh"] = {
            "rows": rows,
            "cols": cols,
            "u_order": int(quilt["uOrder"]),
            "v_order": int(quilt["vOrder"]),
            "bounds": (bounds["Left"], bounds["Top "], bounds["Rght"], bounds["Btom"]),
            "dst_x": np.array(mesh["Hrzn"], dtype=np.float32),
            "dst_y": np.array(mesh["Vrtc"], dtype=np.float32),
            "quilt_slice_x": np.array(env["quiltSliceX"], dtype=np.float32),
            "quilt_slice_y": np.array(env["quiltSliceY"], dtype=np.float32),
        }
    else:
        result["mesh"] = None

    return result


def extract_layer_mask(layer, canvas_size):
    """Return a canvas-sized (H, W) uint8 array from the layer's own raster
    mask, or None if it has no mask. This is authored directly in full
    canvas coordinates by Photoshop, so it's applied as-is to the final
    placed/warped result - no extra warping needed. This is the piece
    that actually clips the design to the mug's visible printable area
    (curvature falloff on the right side, handle cutout, etc.)."""
    mask = getattr(layer, "mask", None)
    if mask is None:
        return None

    cw, ch = canvas_size
    bg_val = int(getattr(mask, "background_color", 0))
    canvas_mask = np.full((ch, cw), bg_val, dtype=np.uint8)

    mask_img = mask.topil()
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


# --------------------------------------------------------------------------
# Step 2: fit the new design into the smart object's local content box
# --------------------------------------------------------------------------
def fit_design(design_img: Image.Image, target_w: int, target_h: int, mode="fill"):
    design_img = design_img.convert("RGBA")
    dw, dh = design_img.size
    target_w, target_h = int(round(target_w)), int(round(target_h))
    if mode == "stretch":
        return design_img.resize((target_w, target_h), Image.LANCZOS)
    scale = max(target_w / dw, target_h / dh) if mode == "fill" else min(target_w / dw, target_h / dh)
    new_w, new_h = max(1, round(dw * scale)), max(1, round(dh * scale))
    resized = design_img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    if mode == "fill":
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        resized = resized.crop((left, top, left + target_w, top + target_h))
        canvas.paste(resized, (0, 0))
    else:
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        canvas.paste(resized, offset, resized)
    return canvas


# --------------------------------------------------------------------------
# Step 3: apply the custom mesh warp via piecewise-triangle affine warping
# --------------------------------------------------------------------------
def _quilt_rest_positions(slice_coords):
    """Reconstruct the TRUE rest (source/undeformed) control-point positions
    for a uOrder/vOrder=4 (cubic) Photoshop custom-warp mesh.

    slice_coords are the irregular quilt slice boundaries (quiltSliceX or
    quiltSliceY - e.g. 4 values = 3 slices). Within each slice Photoshop
    places control points at parametric t = 1/3, 2/3, 1 (cubic Bezier,
    4 control points per edge), with each slice's t=0 point being the
    SAME as the previous slice's t=1 point. This is NOT uniform spacing
    across the full bounding box - slices can be very different widths
    (confirmed in your data: last X-slice is ~4x wider than the first two).
    """
    coords = [float(slice_coords[0])]
    for k in range(len(slice_coords) - 1):
        a, b = float(slice_coords[k]), float(slice_coords[k + 1])
        for t in (1 / 3, 2 / 3, 1.0):
            coords.append(a + t * (b - a))
    return coords


def _warp_triangle(src, dst, src_tri, dst_tri):
    canvas_h, canvas_w = dst.shape[:2]
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect_raw = cv2.boundingRect(np.float32([dst_tri]))
    sx, sy, sw, sh = src_rect
    rdx, rdy, rdw, rdh = dst_rect_raw
    if sw <= 0 or sh <= 0 or rdw <= 0 or rdh <= 0:
        return
    dx = max(rdx, 0)
    dy = max(rdy, 0)
    dx2 = min(rdx + rdw, canvas_w)
    dy2 = min(rdy + rdh, canvas_h)
    dw_ = dx2 - dx
    dh_ = dy2 - dy
    if dw_ <= 0 or dh_ <= 0:
        return
    src_tri_offset = np.float32([(p[0] - sx, p[1] - sy) for p in src_tri])
    dst_tri_offset_full = np.float32([(p[0] - rdx, p[1] - rdy) for p in dst_tri])
    src_crop = src[sy:sy + sh, sx:sx + sw]
    if src_crop.size == 0:
        return
    warp_mat = cv2.getAffineTransform(src_tri_offset, dst_tri_offset_full)
    warped_full = cv2.warpAffine(
        src_crop, warp_mat, (rdw, rdh),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    mask_full = np.zeros((rdh, rdw), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, np.int32(np.round(dst_tri_offset_full)), 255, cv2.LINE_AA)
    crop_left = dx - rdx
    crop_top = dy - rdy
    warped = warped_full[crop_top:crop_top + dh_, crop_left:crop_left + dw_]
    mask = mask_full[crop_top:crop_top + dh_, crop_left:crop_left + dw_]
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    dst_region = dst[dy:dy + dh_, dx:dx + dw_].astype(np.float32)
    warped_f = warped.astype(np.float32)
    blended = dst_region * (1 - mask_f) + warped_f * mask_f
    dst[dy:dy + dh_, dx:dx + dw_] = blended.astype(np.uint8)


def apply_mesh_warp(local_img: Image.Image, mesh: dict) -> Image.Image:
    rows, cols = mesh["rows"], mesh["cols"]

    if mesh["u_order"] != 4 or mesh["v_order"] != 4:
        raise NotImplementedError(
            f"uOrder/vOrder = {mesh['u_order']}/{mesh['v_order']} - this script only "
            f"handles the standard cubic (order=4) custom warp. Share the descriptor "
            f"and we'll extend _quilt_rest_positions for this case."
        )

    xs = _quilt_rest_positions(mesh["quilt_slice_x"])
    ys = _quilt_rest_positions(mesh["quilt_slice_y"])
    if len(xs) != cols or len(ys) != rows:
        raise ValueError(
            f"Quilt slice reconstruction produced {len(xs)} cols / {len(ys)} rows, "
            f"but the PSD declares deformNumCols={cols} / deformNumRows={rows}. "
            f"The slice count doesn't match a simple cubic subdivision - share the "
            f"raw quiltSliceX/quiltSliceY/deformNumRows/deformNumCols values."
        )

    src_pts = np.zeros((rows * cols, 2), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            src_pts[r * cols + c] = [xs[c], ys[r]]

    dst_pts = np.stack([mesh["dst_x"], mesh["dst_y"]], axis=1).astype(np.float32)

    src_arr = np.array(local_img)
    out_arr = np.zeros_like(src_arr)

    for r in range(rows - 1):
        for c in range(cols - 1):
            i00 = r * cols + c
            i10 = r * cols + (c + 1)
            i01 = (r + 1) * cols + c
            i11 = (r + 1) * cols + (c + 1)
            for tri in [(i00, i10, i01), (i10, i11, i01)]:
                src_tri = src_pts[list(tri)]
                dst_tri = dst_pts[list(tri)]
                _warp_triangle(src_arr, out_arr, src_tri, dst_tri)

    return Image.fromarray(out_arr, mode="RGBA")


# --------------------------------------------------------------------------
# Step 4: place the (warped) local image onto the canvas using Trnf
# --------------------------------------------------------------------------
def place_on_canvas(local_img: Image.Image, trnf_quad: np.ndarray,
                     local_size, canvas_size) -> Image.Image:
    lw, lh = local_size
    src_quad = np.float32([[0, 0], [lw, 0], [lw, lh], [0, lh]])
    dst_quad = trnf_quad.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    local_arr = np.array(local_img)
    cw, ch = canvas_size
    warped = cv2.warpPerspective(
        local_arr, M, (cw, ch),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    return Image.fromarray(warped, mode="RGBA")


# --------------------------------------------------------------------------
# Step 5: blend the placed layer onto the background using the real blend mode
# --------------------------------------------------------------------------
def _linear_burn(base, blend):
    return np.clip(base.astype(np.int16) + blend.astype(np.int16) - 255, 0, 255).astype(np.uint8)


def _multiply(base, blend):
    return ((base.astype(np.uint16) * blend.astype(np.uint16)) // 255).astype(np.uint8)


def _screen(base, blend):
    return (255 - ((255 - base.astype(np.uint16)) * (255 - blend.astype(np.uint16)) // 255)).astype(np.uint8)


def _normal(base, blend):
    return blend


BLEND_FUNCS = {
    "BlendMode.NORMAL": _normal,
    "BlendMode.MULTIPLY": _multiply,
    "BlendMode.LINEAR_BURN": _linear_burn,
    "BlendMode.SCREEN": _screen,
}


def composite_layer(background: Image.Image, placed_layer: Image.Image,
                     blend_mode: str, opacity: float = 1.0) -> Image.Image:
    bg = np.array(background.convert("RGBA")).astype(np.uint8)
    fg = np.array(placed_layer.convert("RGBA")).astype(np.uint8)
    blend_fn = BLEND_FUNCS.get(blend_mode, _normal)
    if blend_fn is not _normal:
        blended_rgb = blend_fn(bg[..., :3], fg[..., :3])
    else:
        blended_rgb = fg[..., :3]
    alpha = (fg[..., 3:4].astype(np.float32) / 255.0) * opacity
    out_rgb = bg[..., :3].astype(np.float32) * (1 - alpha) + blended_rgb.astype(np.float32) * alpha
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out = np.dstack([out_rgb, np.full(bg.shape[:2], 255, dtype=np.uint8)])
    return Image.fromarray(out, mode="RGBA")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def generate_mockup(psd_path, layer_name, design_path, output_path, fit_mode="fill",
                     use_mask=True, debug=False):
    import os
    data = extract_smart_object_data(psd_path, layer_name)
    design = Image.open(design_path)
    lw, lh = data["local_size"]
    local_img = fit_design(design, lw, lh, mode=fit_mode)

    if data["mesh"] is not None:
        print(f"Applying custom mesh warp ({data['mesh']['rows']}x{data['mesh']['cols']} grid, "
              f"quilt-slice-corrected source positions)...")
        local_img = apply_mesh_warp(local_img, data["mesh"])
    else:
        print("No mesh warp found on this layer - skipping to plain placement.")

    if debug:
        os.makedirs("analysis_out", exist_ok=True)
        local_img.save("analysis_out/debug_1_warped_local.png")
        print("  [debug] wrote analysis_out/debug_1_warped_local.png "
              "(this is the mesh-warped design BEFORE placement - the curve, if any, "
              "should already be visible here as a deformed rectangle)")

    print("Placing onto canvas via perspective transform (Trnf quad)...")
    placed = place_on_canvas(local_img, data["trnf_quad"], data["local_size"], data["canvas_size"])

    if debug:
        placed.save("analysis_out/debug_2_placed_no_mask.png")
        print("  [debug] wrote analysis_out/debug_2_placed_no_mask.png "
              "(placed on canvas, mask NOT yet applied)")

    layer_mask = extract_layer_mask(data["layer"], data["canvas_size"])
    if debug and layer_mask is not None:
        Image.fromarray(layer_mask, mode="L").save("analysis_out/debug_3_raw_mask.png")
        print("  [debug] wrote analysis_out/debug_3_raw_mask.png "
              "(the raw layer mask exactly as extracted - look at whether its edges "
              "are curvy or a plain rectangle)")

    if use_mask and layer_mask is not None:
        print("Applying layer's raster mask (clips to real mug print area)...")
        placed_arr = np.array(placed)
        placed_arr[..., 3] = (
            placed_arr[..., 3].astype(np.float32) * (layer_mask.astype(np.float32) / 255.0)
        ).astype(np.uint8)
        placed = Image.fromarray(placed_arr, mode="RGBA")
    elif not use_mask:
        print("Skipping mask application (--no-mask).")
    else:
        print("WARNING: layer has no raster mask.")

    data["layer"].visible = False
    background = data["psd"].composite()
    data["layer"].visible = True

    print(f"Compositing with blend_mode={data['blend_mode']} opacity={data['opacity']:.2f}...")
    final = composite_layer(background, placed, data["blend_mode"], data["opacity"])
    final.convert("RGB").save(output_path, quality=95)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psd", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fit", choices=["fill", "fit", "stretch"], default="fill")
    parser.add_argument("--no-mask", action="store_true", help="skip applying the layer's raster mask")
    parser.add_argument("--debug", action="store_true", help="dump intermediate stages to analysis_out/")
    args = parser.parse_args()
    generate_mockup(args.psd, args.layer, args.design, args.out, args.fit,
                     use_mask=not args.no_mask, debug=args.debug)