"""
Photoshop Blend Modes: Implementation of Photoshop layer blend functions.
"""

import numpy as np
from PIL import Image


def _normal(base, blend):
    return blend


def _multiply(base, blend):
    return ((base.astype(np.uint16) * blend.astype(np.uint16)) // 255).astype(
        np.uint8
    )


def _linear_burn(base, blend):
    return np.clip(
        base.astype(np.int16) + blend.astype(np.int16) - 255, 0, 255
    ).astype(np.uint8)


def _screen(base, blend):
    return (
        255
        - (
            (255 - base.astype(np.uint16))
            * (255 - blend.astype(np.uint16))
            // 255
        )
    ).astype(np.uint8)


def _overlay(base, blend):
    b = base.astype(np.float32) / 255.0
    f = blend.astype(np.float32) / 255.0
    res = np.where(b < 0.5, 2.0 * b * f, 1.0 - 2.0 * (1.0 - b) * (1.0 - f))
    return np.clip(res * 255.0, 0, 255).astype(np.uint8)


def _soft_light(base, blend):
    b = base.astype(np.float32) / 255.0
    f = blend.astype(np.float32) / 255.0
    res = np.where(
        f < 0.5,
        b - (1.0 - 2.0 * f) * b * (1.0 - b),
        b + (2.0 * f - 1.0) * (np.sqrt(b) - b),
    )
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
    s = (
        str(mode_str)
        .lower()
        .replace("blendmode.", "")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )
    return s


def apply_blend_mode(
    background: Image.Image,
    foreground: Image.Image,
    mode_str: str,
    opacity: float = 1.0,
):
    bg = np.array(background.convert("RGBA")).astype(np.uint8)
    fg = np.array(foreground.convert("RGBA")).astype(np.uint8)

    norm_mode = normalize_blend_mode(mode_str)
    blend_fn = BLEND_FUNCS.get(norm_mode, _normal)

    blended_rgb = (
        blend_fn(bg[..., :3], fg[..., :3])
        if blend_fn is not _normal
        else fg[..., :3]
    )

    fg_alpha = (fg[..., 3:4].astype(np.float32) / 255.0) * opacity
    bg_alpha = bg[..., 3:4].astype(np.float32) / 255.0

    out_rgb = (
        bg[..., :3].astype(np.float32) * (1.0 - fg_alpha)
        + blended_rgb.astype(np.float32) * fg_alpha
    )
    out_alpha = (fg_alpha + bg_alpha * (1.0 - fg_alpha)) * 255.0

    out = np.dstack(
        [
            np.clip(out_rgb, 0, 255).astype(np.uint8),
            np.clip(out_alpha, 0, 255).astype(np.uint8),
        ]
    )
    return Image.fromarray(out, mode="RGBA")
