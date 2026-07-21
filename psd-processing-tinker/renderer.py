"""
Renderer: Multi-layer PSD mockup rendering orchestrator.
"""

import numpy as np
from PIL import Image, ImageChops

try:
    from .psd_parser import (
        get_flat_layer_list,
        extract_smart_object_data,
        extract_layer_mask,
    )
    from .warp_engine import warp_and_place, fit_design
    from .blending import apply_blend_mode
except ImportError:
    from psd_parser import (
        get_flat_layer_list,
        extract_smart_object_data,
        extract_layer_mask,
    )
    from warp_engine import warp_and_place, fit_design
    from blending import apply_blend_mode


def render_full_mockup(
    psd,
    replacement_input,
    fit_mode="fill",
    use_mask=True,
    progress_callback=None,
):
    """Renders the full PSD layer tree in proper z-order with Smart Object design replacements."""
    canvas_w, canvas_h = psd.width, psd.height
    accumulated_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    flat_layers = get_flat_layer_list(psd)
    total_layers = len(flat_layers)

    last_base_alpha = None

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

        is_so = getattr(layer, "kind", None) == "smartobject"
        design_img = (
            replacement_map.get(layer_name, default_design) if is_so else None
        )

        if is_so and design_img is not None:
            if isinstance(design_img, str):
                design_img = Image.open(design_img)

            so_data = extract_smart_object_data(psd, layer)
            lw, lh = so_data["local_size"]
            local_fitted = fit_design(design_img, lw, lh, mode=fit_mode)

            layer_rendered = warp_and_place(
                local_fitted,
                so_data["mesh"],
                so_data["trnf_quad"],
                so_data["local_size"],
                so_data["canvas_size"],
            )

            if use_mask:
                mask_arr = extract_layer_mask(layer, (canvas_w, canvas_h))
                if mask_arr is not None:
                    arr = np.array(layer_rendered)
                    arr[..., 3] = (
                        arr[..., 3].astype(np.float32)
                        * (mask_arr.astype(np.float32) / 255.0)
                    ).astype(np.uint8)
                    layer_rendered = Image.fromarray(arr, mode="RGBA")

        else:
            layer_rendered = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            bbox = layer.bbox
            try:
                topil_img = layer.topil()
                if (
                    topil_img is not None
                    and bbox
                    and (bbox[2] > bbox[0])
                    and (bbox[3] > bbox[1])
                ):
                    if topil_img.mode != "RGBA":
                        topil_img = topil_img.convert("RGBA")
                    layer_rendered.paste(topil_img, (bbox[0], bbox[1]), topil_img)
                    eff_opacity = opacity
                else:
                    psd_comp = layer.composite()
                    if (
                        psd_comp is not None
                        and bbox
                        and (bbox[2] > bbox[0])
                        and (bbox[3] > bbox[1])
                    ):
                        if psd_comp.mode != "RGBA":
                            psd_comp = psd_comp.convert("RGBA")
                        layer_rendered.paste(
                            psd_comp, (bbox[0], bbox[1]), psd_comp
                        )
                        eff_opacity = 1.0
            except Exception:
                try:
                    psd_comp = layer.composite()
                    if (
                        psd_comp is not None
                        and bbox
                        and (bbox[2] > bbox[0])
                        and (bbox[3] > bbox[1])
                    ):
                        if psd_comp.mode != "RGBA":
                            psd_comp = psd_comp.convert("RGBA")
                        layer_rendered.paste(
                            psd_comp, (bbox[0], bbox[1]), psd_comp
                        )
                        eff_opacity = 1.0
                except Exception:
                    pass

        if is_clipped and last_base_alpha is not None:
            r, g, b, a = layer_rendered.split()
            new_a = ImageChops.multiply(a, last_base_alpha)
            layer_rendered = Image.merge("RGBA", (r, g, b, new_a))
        elif not is_clipped:
            last_base_alpha = layer_rendered.getchannel("A")

        accumulated_canvas = apply_blend_mode(
            accumulated_canvas, layer_rendered, blend_mode, eff_opacity
        )

    return accumulated_canvas.convert("RGB")
