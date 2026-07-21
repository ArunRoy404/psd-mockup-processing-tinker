"""
analyze_psd_smartobject.py

Purpose: figure out EXACTLY how a smart object's "wrap" is implemented in a
PSD (displacement map / mesh warp / simple perspective / smart filter stack),
before we write any warping code.

Usage:
    python analyze_psd_smartobject.py path/to/mockup.psd

Output:
    - Printed layer tree with bounding boxes / blend modes
    - For each smart object layer: embedded content extracted to ./analysis_out/
    - Raw warp/placed-layer descriptor dumped to ./analysis_out/<layer>_descriptor.json
    - Full composite render -> analysis_out/composite_full.png
    - Composite with each smart object hidden -> analysis_out/composite_without_<layer>.png
      (shows you the shadow/highlight/shading layers that sit on top of the design)

Requires: pip install psd-tools pillow numpy
"""

import sys
import os
import json
from pathlib import Path

from psd_tools import PSDImage
from psd_tools.constants import Tag


try:
    import attr as _attr_module
except ImportError:
    _attr_module = None


def jsonable(obj, depth=0, max_depth=14, _seen=None):
    """Best-effort conversion of psd-tools descriptor objects into plain
    Python types so we can json.dumps() them and actually read them.
    psd-tools uses the `attrs` library heavily for things like
    PlacedLayerData / SmartObjectLayerData, so we introspect those
    directly rather than relying on dict-like duck typing."""
    if _seen is None:
        _seen = set()

    if depth > max_depth:
        return "<max depth>"

    # Primitives
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

    # dict-like descriptors (psd_tools' Descriptor is an OrderedDict subclass)
    if isinstance(obj, dict):
        _seen.add(obj_id)
        out = {}
        for k, v in obj.items():
            key = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)
            out[key] = jsonable(v, depth + 1, max_depth, _seen)
        return out

    # list-like
    if isinstance(obj, (list, tuple)):
        _seen.add(obj_id)
        return [jsonable(v, depth + 1, max_depth, _seen) for v in obj]

    # attrs-decorated classes (most psd-tools structures, including
    # PlacedLayerData / SmartObjectLayerData / Transform / Warp)
    if _attr_module is not None and _attr_module.has(type(obj)):
        _seen.add(obj_id)
        try:
            field_names = [f.name for f in _attr_module.fields(type(obj))]
            out = {}
            for name in field_names:
                try:
                    out[name] = jsonable(getattr(obj, name), depth + 1, max_depth, _seen)
                except Exception as e:
                    out[name] = f"<error reading field: {e}>"
            return {"__type__": type(obj).__name__, **out}
        except Exception:
            pass  # fall through to other strategies

    # enums / anything with a plain `.value`
    if hasattr(obj, "value") and not callable(getattr(obj, "value")):
        try:
            return jsonable(obj.value, depth + 1, max_depth, _seen)
        except Exception:
            pass

    # generic objects with a __dict__ (plain classes)
    if hasattr(obj, "__dict__") and vars(obj):
        _seen.add(obj_id)
        return jsonable(vars(obj), depth + 1, max_depth, _seen)

    # objects using __slots__ instead of __dict__
    slots = getattr(obj, "__slots__", None)
    if slots:
        _seen.add(obj_id)
        out = {}
        for slot in slots:
            try:
                out[slot] = jsonable(getattr(obj, slot), depth + 1, max_depth, _seen)
            except Exception:
                pass
        if out:
            return {"__type__": type(obj).__name__, **out}

    # dict-like via .items()
    if hasattr(obj, "items"):
        try:
            return jsonable(dict(obj.items()), depth + 1, max_depth, _seen)
        except Exception:
            pass

    # last resort: string repr (still useful for plain enums/values)
    return str(obj)


def dump_placed_layer_descriptor(layer, out_dir: Path):
    """Enumerate every tagged block ACTUALLY present on this layer (don't
    guess names) and dump each one's data so we can see what's really
    stored, including the placed-layer transform quad and any warp info."""
    found_any = False
    tagged_blocks = getattr(layer._record, "tagged_blocks", None)
    if tagged_blocks is None:
        print(f"    (no tagged_blocks attribute at all on layer record for {layer.name})")
        return found_any

    # tagged_blocks keys are already-resolved Tag enum members (or raw
    # signature bytes for unrecognized ones) - just iterate what's there.
    try:
        items = list(tagged_blocks.items())
    except AttributeError:
        # some psd-tools versions expose it as a list of (key, block) tuples directly
        items = list(tagged_blocks)

    print(f"    tagged blocks present: {[str(k) for k, _ in items]}")

    for key, block in items:
        found_any = True
        tag_label = getattr(key, "name", None) or str(key)
        try:
            data = block.data
        except Exception as e:
            data = f"<could not read block.data: {e}>"

        out_path = out_dir / f"{layer.name}_{tag_label}.json".replace("/", "_")
        try:
            with open(out_path, "w") as f:
                json.dump(jsonable(data), f, indent=2, default=str)
            print(f"    -> wrote {out_path}")
        except Exception as e:
            print(f"    (failed to serialize {tag_label}: {e}; raw repr below)")
            print(f"       {repr(data)[:500]}")

    if not found_any:
        print("    (tagged_blocks exists but is empty for this layer)")
    return found_any


def describe_layer(layer, indent=0):
    pad = "  " * indent
    kind = getattr(layer, "kind", "?")
    bbox = layer.bbox if hasattr(layer, "bbox") else None
    print(f"{pad}- '{layer.name}' kind={kind} visible={layer.visible} "
          f"blend={getattr(layer, 'blend_mode', None)} bbox={bbox}")

    if kind == "smartobject":
        so = layer.smart_object
        print(f"{pad}    smart_object: filename={so.filename!r} "
              f"unique_id={so.unique_id} size={so.size if hasattr(so,'size') else '?'}")

    if hasattr(layer, "__iter__") and kind == "group":
        for child in layer:
            describe_layer(child, indent + 1)


def main(psd_path):
    out_dir = Path("analysis_out")
    out_dir.mkdir(exist_ok=True)

    psd = PSDImage.open(psd_path)
    print(f"Opened {psd_path}  size={psd.size}  color_mode={psd.color_mode}\n")

    print("=== LAYER TREE ===")
    for layer in psd:
        describe_layer(layer)
    print()

    # Full composite for reference
    full = psd.composite()
    full.save(out_dir / "composite_full.png")
    print(f"Wrote {out_dir/'composite_full.png'}")

    # Walk all layers (flat) to find smart objects
    def all_layers(layer):
        yield layer
        if hasattr(layer, "__iter__"):
            for child in layer:
                yield from all_layers(child)

    smart_layers = [l for l in psd.descendants() if getattr(l, "kind", None) == "smartobject"] \
        if hasattr(psd, "descendants") else \
        [l for top in psd for l in all_layers(top) if getattr(l, "kind", None) == "smartobject"]

    if not smart_layers:
        print("No smart object layers found via psd-tools 'kind' detection. "
              "The file may need psd-tools' raw layer records instead — "
              "share the layer tree output above and we'll adjust.")
        return

    print(f"\n=== FOUND {len(smart_layers)} SMART OBJECT LAYER(S) ===")
    for layer in smart_layers:
        print(f"\n--- {layer.name} ---")

        # 1. extract embedded (flat, unwrapped) source content
        try:
            so = layer.smart_object
            data = so.data
            ext = (so.filename or "").split(".")[-1].lower()
            if ext not in ("png", "jpg", "jpeg", "psd", "psb", "tif", "tiff"):
                ext = "bin"
            raw_path = out_dir / f"{layer.name}_embedded_source.{ext}"
            with open(raw_path, "wb") as f:
                f.write(data)
            print(f"    Extracted embedded content -> {raw_path}")
        except Exception as e:
            print(f"    Could not extract embedded content: {e}")

        # 2. dump placed-layer / warp descriptor
        print("    Scanning for placed-layer / warp / filter descriptors...")
        dump_placed_layer_descriptor(layer, out_dir)

        # 3. composite with this smart object hidden, to reveal shading layers
        layer.visible = False
        without = psd.composite()
        without_path = out_dir / f"composite_without_{layer.name}.png"
        without.save(without_path)
        layer.visible = True
        print(f"    Wrote {without_path} (compare against composite_full.png "
              f"to see shadow/highlight layers)")

    print("\n=== NEXT STEP ===")
    print("Open the *_descriptor.json files. Look for:")
    print("  - a 'warp' key with 'warpStyle' != 'warpNone'  -> parametric/mesh warp")
    print("  - 'customEnvelopeWarp' with a 'meshPoints' array -> custom mesh warp")
    print("  - anything mentioning 'Displace' in FILTER_EFFECTS blocks -> displacement map filter")
    print("  - if nothing informative shows up at all and the layer only has a plain")
    print("    Transform, it's likely a simple perspective warp on the smart object's corners.")
    print("Share what you find (or just the JSON files) and we'll pick the exact reproduction method.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_psd_smartobject.py path/to/mockup.psd")
        sys.exit(1)
    main(sys.argv[1])