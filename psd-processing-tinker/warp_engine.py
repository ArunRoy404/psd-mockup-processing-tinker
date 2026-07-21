"""
Warp Engine: Multi-slice Bicubic Bezier patch evaluation & homography remapping.
"""

import numpy as np
import cv2
from PIL import Image


def fit_design(
    design_img: Image.Image, target_w: float, target_h: float, mode="fill"
):
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

    scale_w = target_w_i / float(dw)
    scale_h = target_h_i / float(dh)
    scale = (
        max(scale_w, scale_h)
        if mode == "fill"
        else (min(scale_w, scale_h) if mode == "fit" else 1.0)
    )

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


def _bernstein(i, t):
    return [(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3][i]


def _bezier_patch_eval(u_grid, v_grid, ctrl_x, ctrl_y):
    Bu = np.array([[_bernstein(j, u) for j in range(4)] for u in u_grid])
    Bv = np.array([[_bernstein(i, v) for i in range(4)] for v in v_grid])
    cx = np.array(ctrl_x, dtype=np.float64).reshape(4, 4)
    cy = np.array(ctrl_y, dtype=np.float64).reshape(4, 4)
    dst_x = np.einsum("ia,jb,ab->ij", Bv, Bu, cx)
    dst_y = np.einsum("ia,jb,ab->ij", Bv, Bu, cy)
    return dst_x, dst_y


def _fill_patch_map(
    map_x,
    map_y,
    ctrl_x,
    ctrl_y,
    src_x0,
    src_y0,
    src_x1,
    src_y1,
    canvas_w_s,
    canvas_h_s,
    img_w,
    img_h,
    subdiv=24,
):
    grid = np.linspace(0.0, 1.0, subdiv)
    dst_x, dst_y = _bezier_patch_eval(grid, grid, ctrl_x, ctrl_y)

    for i in range(subdiv - 1):
        for j in range(subdiv - 1):
            dst_q = np.array(
                [
                    [dst_x[i, j], dst_y[i, j]],
                    [dst_x[i, j + 1], dst_y[i, j + 1]],
                    [dst_x[i + 1, j + 1], dst_y[i + 1, j + 1]],
                    [dst_x[i + 1, j], dst_y[i + 1, j]],
                ],
                dtype=np.float32,
            )

            su0 = src_x0 + (j / (subdiv - 1)) * (src_x1 - src_x0)
            su1 = src_x0 + ((j + 1) / (subdiv - 1)) * (src_x1 - src_x0)
            sv0 = src_y0 + (i / (subdiv - 1)) * (src_y1 - src_y0)
            sv1 = src_y0 + ((i + 1) / (subdiv - 1)) * (src_y1 - src_y0)

            src_q = np.array(
                [[su0, sv0], [su1, sv0], [su1, sv1], [su0, sv1]],
                dtype=np.float32,
            )

            H, _ = cv2.findHomography(dst_q, src_q)
            if H is None:
                continue

            min_x = max(0, int(np.floor(dst_q[:, 0].min())))
            max_x = min(canvas_w_s - 1, int(np.ceil(dst_q[:, 0].max())))
            min_y = max(0, int(np.floor(dst_q[:, 1].min())))
            max_y = min(canvas_h_s - 1, int(np.ceil(dst_q[:, 1].max())))

            if min_x > max_x or min_y > max_y:
                continue

            grid_x, grid_y = np.meshgrid(
                np.arange(min_x, max_x + 1, dtype=np.float32),
                np.arange(min_y, max_y + 1, dtype=np.float32),
            )
            pts_dst = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

            pts_poly = dst_q.reshape(-1, 1, 2)
            inside_mask = np.array(
                [
                    cv2.pointPolygonTest(
                        pts_poly, (float(p[0]), float(p[1])), False
                    )
                    >= 0
                    for p in pts_dst
                ]
            )

            if not np.any(inside_mask):
                continue

            valid_dst = pts_dst[inside_mask]
            ones = np.ones((valid_dst.shape[0], 1), dtype=np.float32)
            valid_h = np.hstack([valid_dst, ones])

            src_h = valid_h @ H.T
            valid_w = np.abs(src_h[:, 2]) > 1e-9
            src_h = src_h[valid_w]
            valid_dst = valid_dst[valid_w]

            sx = src_h[:, 0] / src_h[:, 2]
            sy = src_h[:, 1] / src_h[:, 2]

            valid_bounds = (
                (sx >= 0) & (sx < img_w - 1) & (sy >= 0) & (sy < img_h - 1)
            )
            vx = valid_dst[valid_bounds, 0].astype(int)
            vy = valid_dst[valid_bounds, 1].astype(int)

            map_x[vy, vx] = sx[valid_bounds]
            map_y[vy, vx] = sy[valid_bounds]


def warp_and_place(
    design_img: Image.Image,
    mesh: dict,
    trnf_quad: np.ndarray,
    local_size: tuple,
    canvas_size: tuple,
    supersample: int = 2,
    subdiv: int = 16,
) -> Image.Image:
    """Warps design replacement image onto target PSD canvas using Bezier mesh mapping."""
    cw, ch = int(round(canvas_size[0])), int(round(canvas_size[1]))
    cw_s, ch_s = cw * supersample, ch * supersample

    img_arr = np.array(design_img.convert("RGBA"))
    img_h, img_w = img_arr.shape[:2]

    lw, lh = local_size
    src_rect = np.array([[0, 0], [lw, 0], [lw, lh], [0, lh]], dtype=np.float32)
    dst_quad = trnf_quad.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_rect, dst_quad)

    map_x = np.full((ch_s, cw_s), -1.0, dtype=np.float32)
    map_y = np.full((ch_s, cw_s), -1.0, dtype=np.float32)

    if (
        mesh is None
        or mesh.get("dst_x") is None
        or mesh.get("dst_y") is None
    ):
        # Fallback to perspective placement via inverse homography mapping
        Ms = M.copy()
        Ms[0, :] *= supersample
        Ms[1, :] *= supersample
        Minv = np.linalg.inv(Ms)
        xs, ys = np.meshgrid(
            np.arange(cw_s, dtype=np.float64),
            np.arange(ch_s, dtype=np.float64),
        )
        ones = np.ones_like(xs)
        pts = np.stack([xs, ys, ones], axis=-1)
        src_pts = pts @ Minv.T
        w = src_pts[..., 2]
        valid_w = np.abs(w) > 1e-9
        safe_w = np.where(valid_w, w, 1.0)
        sx = src_pts[..., 0] / safe_w
        sy = src_pts[..., 1] / safe_w
        inside = (
            valid_w & (sx >= 0) & (sx < img_w) & (sy >= 0) & (sy < img_h)
        )
        map_x[inside] = sx[inside]
        map_y[inside] = sy[inside]
    else:
        rows, cols = mesh["rows"], mesh["cols"]
        qsx = mesh["quilt_slice_x"]
        qsy = mesh["quilt_slice_y"]
        n_xslices = max(1, len(qsx) - 1)
        n_yslices = max(1, len(qsy) - 1)

        pts_local = (
            np.stack([mesh["dst_x"], mesh["dst_y"]], axis=1)
            .astype(np.float32)
            .reshape(-1, 1, 2)
        )
        pts_canvas = (
            cv2.perspectiveTransform(pts_local, M).reshape(-1, 2) * supersample
        )
        mesh_cx = pts_canvas[:, 0]
        mesh_cy = pts_canvas[:, 1]

        for m in range(n_yslices):
            for k in range(n_xslices):
                row_idx = [3 * m + i for i in range(4)]
                col_idx = [3 * k + j for j in range(4)]

                if max(row_idx) >= rows or max(col_idx) >= cols:
                    continue

                ctrl_x = np.array(
                    [mesh_cx[r * cols + c] for r in row_idx for c in col_idx]
                )
                ctrl_y = np.array(
                    [mesh_cy[r * cols + c] for r in row_idx for c in col_idx]
                )

                src_x0, src_x1 = qsx[k], qsx[k + 1]
                src_y0, src_y1 = qsy[m], qsy[m + 1]

                _fill_patch_map(
                    map_x,
                    map_y,
                    ctrl_x,
                    ctrl_y,
                    src_x0,
                    src_y0,
                    src_x1,
                    src_y1,
                    cw_s,
                    ch_s,
                    img_w,
                    img_h,
                    subdiv=subdiv,
                )

    map_x32 = map_x.astype(np.float32)
    map_y32 = map_y.astype(np.float32)
    warped_hi = cv2.remap(
        img_arr,
        map_x32,
        map_y32,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    warped = cv2.resize(warped_hi, (cw, ch), interpolation=cv2.INTER_AREA)

    if warped.shape[2] == 4:
        r, g, b, a = cv2.split(warped)
        a = cv2.GaussianBlur(a, (3, 3), 0)
        warped = cv2.merge([r, g, b, a])

    return Image.fromarray(warped, mode="RGBA")
