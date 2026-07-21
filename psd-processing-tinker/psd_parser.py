"""
PSD Parser: Descriptor resolution, tagged block parsing, and layer tree utilities.
"""

import numpy as np
import cv2

try:
    import attr as _attr_module
except ImportError:
    _attr_module = None


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
        if hasattr(layer, "is_group") and layer.is_group():
            layers.extend(get_flat_layer_list(layer))
        else:
            layers.append(layer)
    return layers


def list_smart_object_layers(psd):
    return [
        l
        for l in get_flat_layer_list(psd)
        if getattr(l, "kind", None) == "smartobject"
    ]


from psd_tools.constants import Tag

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
