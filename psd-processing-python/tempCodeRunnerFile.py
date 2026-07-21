import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageChops
from psd_tools import PSDImage

class PsdWarpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSD Multi-Layer Warp Engine")
        self.root.geometry("1000x850") # Slightly taller for progress bar
        
        self.psd_obj = None
        self.psd_json = None
        self.user_img = None
        self.preview_photo = None
        
        self.status_labels = {}
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#f3f4f6")
        main_frame.pack(fill='both', expand=True)

        tk.Label(main_frame, text="Layered Mockup Generator", font=('Arial', 20, 'bold'), bg="#f3f4f6").pack(pady=10)

        # Control Panel
        ctrl = tk.Frame(main_frame, bg="#f3f4f6")
        ctrl.pack(pady=10)

        self.create_row(ctrl, "PSD File:", self.load_psd, 0)
        self.create_row(ctrl, "Mesh JSON:", self.load_json, 1)
        self.create_row(ctrl, "User Image:", self.load_image, 2)

        # Progress Section
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=(20, 0), padx=50, fill='x')
        
        self.progress_label = tk.Label(main_frame, text="Ready", font=('Arial', 10), bg="#f3f4f6", fg="#4b5563")
        self.progress_label.pack(pady=5)

        tk.Button(main_frame, text="GENERATE MOCKUP", command=self.process_warp, 
                  bg="#6366f1", fg="white", font=('Arial', 12, 'bold'), height=2).pack(pady=10, padx=50, fill='x')

        self.preview_canvas = tk.Canvas(main_frame, width=800, height=450, bg="#d1d5db")
        self.preview_canvas.pack(pady=10)

    def create_row(self, parent, label, cmd, row):
        tk.Label(parent, text=label, width=15, anchor="w", bg="#f3f4f6").grid(row=row, column=0)
        tk.Button(parent, text="Browse", command=cmd, width=10).grid(row=row, column=1, padx=5, pady=5)
        status = tk.Label(parent, text="Missing", fg="red", bg="#f3f4f6")
        status.grid(row=row, column=2, padx=5)
        self.status_labels[label] = status

    def load_psd(self):
        path = filedialog.askopenfilename(filetypes=[("PSD", "*.psd *.psb")])
        if path:
            self.psd_obj = PSDImage.open(path)
            # Hide all smart objects in the PSD to avoid showing placeholder colors
            self.hide_smart_objects(self.psd_obj)
            self.status_labels["PSD File:"].config(text="Loaded", fg="green")
            self.update_preview(self.psd_obj.composite())

    def hide_smart_objects(self, container):
        """Recursively hide layers that are Smart Objects."""
        for layer in container:
            # Hide if it's a smart object
            if hasattr(layer, 'kind') and layer.kind == 'smartobject':
                layer.visible = False
            # Recurse into groups
            if hasattr(layer, '__iter__') and not hasattr(layer, 'kind'): # Groups don't have kind usually, or kind is 'group'
                self.hide_smart_objects(layer)
            elif hasattr(layer, 'is_group') and layer.is_group():
                self.hide_smart_objects(layer)
            elif hasattr(layer, 'kind') and layer.kind == 'group':
                self.hide_smart_objects(layer)

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'r') as f:
                self.psd_json = json.load(f)  # now psd_json IS the array directly
            self.status_labels["Mesh JSON:"].config(text="Loaded", fg="green")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg")])
        if path:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.user_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self.status_labels["User Image:"].config(text="Loaded", fg="green")

    def bezier_calc(self, u, v, h_pts, v_pts):
        def B(i, t): return [(1-t)**3, 3*t*(1-t)**2, 3*t**2*(1-t), t**3][i]
        x, y = 0, 0
        for i in range(4):
            for j in range(4):
                coeff = B(i, v) * B(j, u)
                x += h_pts[i*4+j] * coeff
                y += v_pts[i*4+j] * coeff
        return x, y

    def get_psd_layer_by_name(self, name):
        """Recursively search for a layer by name in the PSD tree."""
        def search(layer_container):
            for layer in layer_container:
                if layer.name == name:
                    return layer.composite()
                if hasattr(layer, '__iter__'): # Recursive search into groups
                    res = search(layer)
                    if res: return res
            return None
        return search(self.psd_obj)

    def process_warp(self):
        if not all([self.psd_obj, self.psd_json is not None, self.user_img is not None]):
            messagebox.showerror("Error", "All files must be loaded!")
            return

        canvas_w, canvas_h = self.psd_obj.width, self.psd_obj.height
        final_image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        # Robust handling for both JSON formats (dict with 'children' or direct list)
        if isinstance(self.psd_json, dict):
            layers_data = self.psd_json.get('children', [])
        else:
            layers_data = self.psd_json if isinstance(self.psd_json, list) else []
            
        total_layers = len(layers_data)

        idx = 0
        while idx < total_layers:
            layer_data = layers_data[idx]
            if layer_data.get('hidden', False):
                # Skip this layer AND any clipped layers attached to it
                idx += 1
                while idx < total_layers and layers_data[idx].get('clipping', False):
                    idx += 1
                continue

            # 1. Update Progress
            name = layer_data.get('name', 'Unknown')
            percent = (idx / total_layers) * 100
            self.progress_var.set(percent)
            self.progress_label.config(text=f"Processing layer {idx+1}/{total_layers}: {name}")
            self.root.update_idletasks()

            # 2. Render this layer as a potential clipping base
            blend_mode = layer_data.get('blendMode', 'normal')
            base_img = self.render_layer(layer_data, canvas_w, canvas_h)
            final_image = self.apply_blend_mode(final_image, base_img, blend_mode)

            # 3. Handle any consecutive layers clipped to this one
            next_idx = idx + 1
            while next_idx < total_layers and layers_data[next_idx].get('clipping', False):
                clipped_layer_data = layers_data[next_idx]
                
                # Update Progress for clipped layer too
                c_name = clipped_layer_data.get('name', 'Unknown')
                c_blend = clipped_layer_data.get('blendMode', 'normal')
                c_percent = (next_idx / total_layers) * 100
                self.progress_var.set(c_percent)
                self.progress_label.config(text=f"Processing clipped layer {next_idx+1}/{total_layers}: {c_name}")
                self.root.update_idletasks()

                if not clipped_layer_data.get('hidden', False):
                    # Render the clipped layer
                    clipped_img = self.render_layer(clipped_layer_data, canvas_w, canvas_h)
                    
                    # Apply the base layer's alpha mask (stencil effect)
                    base_alpha = base_img.getchannel('A')
                    r, g, b, a = clipped_img.split()
                    new_a = ImageChops.multiply(a, base_alpha)
                    masked_clipped_img = Image.merge("RGBA", (r, g, b, new_a))
                    
                    # Apply blend mode for clipped layer
                    final_image = self.apply_blend_mode(final_image, masked_clipped_img, c_blend)
                
                next_idx += 1
            
            # Skip the clipped layers in the main loop
            idx = next_idx

        # Final Wrap up
        self.progress_var.set(100)
        self.progress_label.config(text="Complete!")
        self.update_preview(final_image)
        final_image.save("final_mockup.png")

    def render_layer(self, layer_data, canvas_w, canvas_h):
        """Helper to render a single layer into a canvas-sized RGBA image."""
        name = layer_data.get('name', 'Unknown')
        opacity = layer_data.get('opacity', 1.0)
        
        # 1. Warp/Smart Object Layer
        if 'placedLayer' in layer_data:
            warped_arr = self.run_warp_math(canvas_w, canvas_h, layer_data)
            layer_img = Image.fromarray(warped_arr)
        
        # 2. Standard Layer
        else:
            psd_layer_img = self.get_psd_layer_by_name(name)
            if psd_layer_img:
                layer_img = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
                mask = psd_layer_img if psd_layer_img.mode == 'RGBA' else None
                layer_img.paste(psd_layer_img, (layer_data['left'], layer_data['top']), mask)
            else:
                layer_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        # Apply Mask if present in JSON
        if 'mask' in layer_data and not layer_data['mask'].get('disabled', False):
            m = layer_data['mask']
            # Create a simple rectangular mask from bounds
            mask_box = Image.new("L", (canvas_w, canvas_h), 0)
            # The mask bounds in JSON are absolute canvas coordinates
            l, t, r, b = m['left'], m['top'], m['right'], m['bottom']
            mask_box.paste(255, (l, t, r, b))
            
            # Combine with existing alpha
            r, g, b, a = layer_img.split()
            new_a = ImageChops.multiply(a, mask_box)
            layer_img = Image.merge("RGBA", (r, g, b, new_a))

        # Apply Opacity
        if opacity < 1.0:
            layer_img = self.apply_opacity(layer_img, opacity)
            
        return layer_img

    def apply_blend_mode(self, background, foreground, mode):
        """Applies Photoshop-style blend modes using numpy for performance."""
        if mode == 'normal' or mode not in ['multiply', 'linear burn']:
            background.alpha_composite(foreground)
            return background

        # Convert to numpy arrays for channel-wise math
        # background and foreground are PIL RGBA images
        back = np.array(background).astype(np.float32)
        front = np.array(foreground).astype(np.float32)

        # Normalize alpha to 0-1
        alpha_f = front[:, :, 3:4] / 255.0
        
        # Separate channels
        rgb_b = back[:, :, :3]
        rgb_f = front[:, :, :3]

        if mode == 'multiply':
            blend_rgb = (rgb_f * rgb_b) / 255.0
        elif mode == 'linear burn':
            blend_rgb = np.clip(rgb_f + rgb_b - 255.0, 0, 255)
        else:
            blend_rgb = rgb_f # Should not happen

        # Alpha Compositing Formula: 
        # Result = Front_Alpha * Blended_Color + (1 - Front_Alpha) * Back_Color
        res_rgb = (alpha_f * blend_rgb) + ((1 - alpha_f) * rgb_b)
        
        # Final Alpha: standard alpha composite formula
        alpha_b = back[:, :, 3:4] / 255.0
        res_alpha = (alpha_f + alpha_b * (1 - alpha_f)) * 255.0

        res_arr = np.concatenate([res_rgb, res_alpha], axis=2).astype(np.uint8)
        return Image.fromarray(res_arr)

    def run_warp_math(self, canvas_w, canvas_h, layer_data):
        scale = 2
        sw, sh = canvas_w * scale, canvas_h * scale
        img_h, img_w = self.user_img.shape[:2]
        
        placed_layer = layer_data.get('placedLayer', {})
        warp_data = placed_layer.get('warp', {})
        
        # 1. Setup Coordinates (Mesh vs simple transform)
        if warp_data.get('style') == 'custom' and 'customEnvelopeWarp' in warp_data:
            # COMPLEX MESH WARP
            mesh = warp_data['customEnvelopeWarp']['meshPoints']
            h_pts = [v * scale for v in next(m['values'] for m in mesh if m['type'] == 'horizontal')]
            v_pts = [v * scale for v in next(m['values'] for m in mesh if m['type'] == 'vertical')]
            
            subdiv = 60 
            grid_u = np.linspace(0, 1, subdiv)
            grid_v = np.linspace(0, 1, subdiv)
            dst_pts = np.zeros((subdiv, subdiv, 2), dtype=np.float32)
            for i, v in enumerate(grid_v):
                for j, u in enumerate(grid_u):
                    dst_pts[i, j] = self.bezier_calc(u, v, h_pts, v_pts)
        else:
            # SIMPLE PERSPECTIVE (No Mesh)
            subdiv = 2
            dst_pts = np.zeros((2, 2, 2), dtype=np.float32)
            transform = placed_layer.get('transform', [])
            if len(transform) == 8:
                # transform is [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
                dst_pts[0, 0] = [transform[0] * scale, transform[1] * scale]
                dst_pts[0, 1] = [transform[2] * scale, transform[3] * scale]
                dst_pts[1, 1] = [transform[4] * scale, transform[5] * scale]
                dst_pts[1, 0] = [transform[6] * scale, transform[7] * scale]
            else:
                # Fallback to layer bounds
                l, t, r, b = layer_data['left'], layer_data['top'], layer_data['right'], layer_data['bottom']
                dst_pts[0, 0] = [l * scale, t * scale]
                dst_pts[0, 1] = [r * scale, t * scale]
                dst_pts[1, 1] = [r * scale, b * scale]
                dst_pts[1, 0] = [l * scale, b * scale]

        # 2. Create high-res map
        map_x = np.full((sh, sw), -1, dtype=np.float32)
        map_y = np.full((sh, sw), -1, dtype=np.float32)
        
        for i in range(subdiv - 1):
            for j in range(subdiv - 1):
                # Scale source quad to user image dimensions
                su0, su1 = j / (subdiv - 1), (j + 1) / (subdiv - 1)
                sv0, sv1 = i / (subdiv - 1), (i + 1) / (subdiv - 1)
                
                src_q = np.array([
                    [su0 * img_w, sv0 * img_h], [su1 * img_w, sv0 * img_h], 
                    [su1 * img_w, sv1 * img_h], [su0 * img_w, sv1 * img_h]
                ], dtype=np.float32)
                
                dst_q = np.array([
                    dst_pts[i, j], dst_pts[i, j+1], 
                    dst_pts[i+1, j+1], dst_pts[i+1, j]
                ], dtype=np.float32)

                H, _ = cv2.findHomography(dst_q, src_q)
                if H is None: continue

                min_c = np.floor(np.min(dst_q, axis=0)).astype(int)
                max_c = np.ceil(np.max(dst_q, axis=0)).astype(int)
                
                min_x, min_y = max(0, min_c[0]), max(0, min_c[1])
                max_x, max_y = min(sw, max_c[0]), min(sh, max_c[1])
                
                for py in range(min_y, max_y):
                    for px in range(min_x, max_x):
                        v_src = H @ np.array([px, py, 1.0])
                        if v_src[2] == 0: continue
                        v_src /= v_src[2]
                        if 0 <= v_src[0] < img_w and 0 <= v_src[1] < img_h:
                            map_x[py, px], map_y[py, px] = v_src[0], v_src[1]

        # 5. High-Quality Remap at 2x size
        warped_high_res = cv2.remap(self.user_img, map_x, map_y, 
                                   interpolation=cv2.INTER_LANCZOS4, 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # 6. Downscale back to original size (This acts as the ultimate anti-aliasing)
        final_warp = cv2.resize(warped_high_res, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

        # 7. Final Polish: Soften the alpha edges slightly to blend with the PSD
        # This removes the "cut-out" look. Channel order is RGBA from self.user_img
        r, g, b, a = cv2.split(final_warp)
        a = cv2.GaussianBlur(a, (3, 3), 0)
        final_warp = cv2.merge([r, g, b, a])

        return final_warp
    def apply_opacity(self, pil_img, opacity):
        """Applies opacity (0.0 to 1.0) to a PIL image."""
        if opacity >= 1.0:
            return pil_img
        
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
            
        r, g, b, a = pil_img.split()
        # Use a lookup table for faster calculation
        lut = [int(i * opacity) for i in range(256)]
        a = a.point(lut)
        return Image.merge("RGBA", (r, g, b, a))

    def update_preview(self, pil_img):
        img_w, img_h = pil_img.size
        ratio = min(800/img_w, 450/img_h)
        new_size = (int(img_w*ratio), int(img_h*ratio))
        preview_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(400, 225, image=self.preview_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = PsdWarpApp(root)
    root.mainloop()