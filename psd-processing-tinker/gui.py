"""
GUI: Interactive Tkinter Application with 2D Design Editor Canvas.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from psd_tools import PSDImage
import numpy as np

try:
    from .psd_parser import list_smart_object_layers, extract_smart_object_data
    from .renderer import render_full_mockup
except ImportError:
    from psd_parser import list_smart_object_layers, extract_smart_object_data
    from renderer import render_full_mockup


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
        self.img_scale = 1.0  # 1.0 = 100% original size
        self.img_offset_x = 0.0  # Position on smart object canvas
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

        tk.Label(
            main_frame,
            text="PSD Layered Mockup Engine",
            font=("Arial", 18, "bold"),
            bg="#f3f4f6",
        ).pack(pady=(0, 5))

        # 1. Select PSD File Frame
        ctrl_frame = tk.LabelFrame(
            main_frame,
            text=" 1. Select PSD Template ",
            font=("Arial", 11, "bold"),
            bg="#f3f4f6",
            padx=10,
            pady=6,
        )
        ctrl_frame.pack(fill="x", pady=4)

        tk.Label(
            ctrl_frame, text="PSD File:", font=("Arial", 10, "bold"), bg="#f3f4f6"
        ).grid(row=0, column=0, sticky="w")
        tk.Button(
            ctrl_frame,
            text="Browse PSD...",
            command=self.load_psd,
            bg="#4f46e5",
            fg="white",
            font=("Arial", 9, "bold"),
        ).grid(row=0, column=1, padx=10)
        self.psd_status = tk.Label(
            ctrl_frame,
            text="No PSD loaded",
            fg="#ef4444",
            bg="#f3f4f6",
            font=("Arial", 10),
        )
        self.psd_status.grid(row=0, column=2, sticky="w")

        self.mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            ctrl_frame,
            text="Apply Layer Raster Masks",
            variable=self.mask_var,
            bg="#f3f4f6",
        ).grid(row=0, column=3, padx=20, sticky="e")

        # 2. Interactive 2D Smart Object Design Editor Frame
        self.editor_frame = tk.LabelFrame(
            main_frame,
            text=" 2. Interactive Smart Object Design Editor ",
            font=("Arial", 11, "bold"),
            bg="#f3f4f6",
            padx=10,
            pady=8,
        )
        self.editor_frame.pack(fill="x", pady=4)

        # Editor Top Control Bar
        top_bar = tk.Frame(self.editor_frame, bg="#f3f4f6")
        top_bar.pack(fill="x", pady=(0, 6))

        tk.Button(
            top_bar,
            text="Select Design Image...",
            command=self.select_design_image,
            bg="#2563eb",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=(0, 10))
        self.img_status_lbl = tk.Label(
            top_bar,
            text="No design image loaded",
            fg="#ef4444",
            bg="#f3f4f6",
            font=("Arial", 9, "bold"),
        )
        self.img_status_lbl.pack(side="left")

        # Preset positioning buttons
        btn_box = tk.Frame(top_bar, bg="#f3f4f6")
        btn_box.pack(side="right")
        tk.Button(btn_box, text="Center", command=self.center_image, width=7).pack(
            side="left", padx=2
        )
        tk.Button(btn_box, text="Fit", command=self.fit_image, width=6).pack(
            side="left", padx=2
        )
        tk.Button(btn_box, text="Fill", command=self.fill_image, width=6).pack(
            side="left", padx=2
        )
        tk.Button(btn_box, text="Reset", command=self.reset_image, width=6).pack(
            side="left", padx=2
        )

        # Scale Control Slider Row
        scale_bar = tk.Frame(self.editor_frame, bg="#f3f4f6")
        scale_bar.pack(fill="x", pady=(0, 6))

        tk.Label(
            scale_bar,
            text="Scale (%):",
            font=("Arial", 9, "bold"),
            bg="#f3f4f6",
        ).pack(side="left", padx=(0, 5))
        self.scale_var = tk.DoubleVar(value=100)
        self.scale_slider = tk.Scale(
            scale_bar,
            from_=10,
            to=300,
            orient="horizontal",
            variable=self.scale_var,
            command=self.on_scale_slider_change,
            showvalue=True,
            length=250,
            bg="#f3f4f6",
        )
        self.scale_slider.pack(side="left", padx=5)

        self.editor_dim_lbl = tk.Label(
            scale_bar,
            text="Canvas: 1187 x 475 px (Mouse drag to move, Scroll to zoom)",
            fg="#4b5563",
            bg="#f3f4f6",
            font=("Arial", 9),
        )
        self.editor_dim_lbl.pack(side="right", padx=5)

        # 2D Interactive Editor Canvas
        self.editor_canvas = tk.Canvas(
            self.editor_frame, width=750, height=220, bg="#e5e7eb", highlightthickness=1
        )
        self.editor_canvas.pack(fill="x", pady=2)

        # Mouse bindings for dragging and scrolling inside 2D Editor
        self.editor_canvas.bind("<ButtonPress-1>", self.on_editor_click)
        self.editor_canvas.bind("<B1-Motion>", self.on_editor_drag)
        self.editor_canvas.bind("<MouseWheel>", self.on_editor_scroll)
        self.editor_canvas.bind("<Button-4>", self.on_editor_scroll)
        self.editor_canvas.bind("<Button-5>", self.on_editor_scroll)

        # 3. Output Mockup Preview & Execution Frame
        preview_frame = tk.LabelFrame(
            main_frame,
            text=" 3. Mockup Preview & Output ",
            font=("Arial", 11, "bold"),
            bg="#f3f4f6",
            padx=10,
            pady=6,
        )
        preview_frame.pack(fill="both", expand=True, pady=4)

        # Generate Button & Progress
        act_bar = tk.Frame(preview_frame, bg="#f3f4f6")
        act_bar.pack(fill="x", pady=(0, 4))

        tk.Button(
            act_bar,
            text="GENERATE MOCKUP PREVIEW",
            command=self.process,
            bg="#10b981",
            fg="white",
            font=("Arial", 11, "bold"),
            height=1,
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            act_bar, variable=self.progress_var, maximum=100, length=200
        )
        self.progress_bar.pack(side="right")

        self.progress_label = tk.Label(
            preview_frame,
            text="Ready",
            font=("Arial", 9),
            bg="#f3f4f6",
            fg="#4b5563",
        )
        self.progress_label.pack(anchor="w")

        # 3D Final Canvas Preview
        self.preview_canvas = tk.Canvas(
            preview_frame, width=800, height=320, bg="#d1d5db", highlightthickness=1
        )
        self.preview_canvas.pack(fill="both", expand=True, pady=2)

    def load_psd(self):
        path = filedialog.askopenfilename(
            filetypes=[("PSD/PSB Files", "*.psd *.psb")]
        )
        if not path:
            return
        try:
            self.psd = PSDImage.open(path)
            self.psd_path = path
            self.smart_layers = list_smart_object_layers(self.psd)

            if not self.smart_layers:
                messagebox.showwarning(
                    "Warning", "No Smart Object layers detected in this PSD file."
                )
                self.psd_status.config(
                    text="Loaded (0 Smart Objects)", fg="#b45309"
                )
            else:
                so_data = extract_smart_object_data(self.psd, self.smart_layers[0])
                self.editor_w, self.editor_h = so_data["local_size"]
                self.psd_status.config(
                    text=f"Loaded: {os.path.basename(path)} ({len(self.smart_layers)} SOs: {int(self.editor_w)}x{int(self.editor_h)}px)",
                    fg="#15803d",
                )

            if self.user_img is not None:
                self.center_image()
            else:
                self.update_editor_preview()

            self.update_preview(self.psd.composite())

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PSD file:\n{e}")

    def select_design_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")
            ]
        )
        if not path:
            return
        try:
            self.user_img_path = path
            self.user_img = Image.open(path).convert("RGBA")
            self.img_scale = 1.0
            self.scale_var.set(100)

            self.center_image()
            self.img_status_lbl.config(
                text=f"{os.path.basename(path)} ({self.user_img.width}x{self.user_img.height}px)",
                fg="#15803d",
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load design image:\n{e}")

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

    def update_editor_preview(self):
        ui_w = self.editor_canvas.winfo_width()
        ui_h = self.editor_canvas.winfo_height()

        if ui_w < 50 or ui_h < 50:
            ui_w, ui_h = 750, 220

        padding = 10
        avail_w, avail_h = max(10, ui_w - 2 * padding), max(10, ui_h - 2 * padding)
        self.editor_display_scale = min(
            avail_w / self.editor_w, avail_h / self.editor_h
        )

        disp_w = max(1, int(round(self.editor_w * self.editor_display_scale)))
        disp_h = max(1, int(round(self.editor_h * self.editor_display_scale)))

        grid_size = 12
        checker_bg = Image.new("RGBA", (disp_w, disp_h), (240, 240, 240, 255))
        arr_check = np.array(checker_bg)
        for y in range(0, disp_h, grid_size):
            for x in range(0, disp_w, grid_size):
                if ((x // grid_size) + (y // grid_size)) % 2 == 1:
                    arr_check[
                        y : min(y + grid_size, disp_h),
                        x : min(x + grid_size, disp_w),
                    ] = [215, 215, 215, 255]
        checker_bg = Image.fromarray(arr_check, mode="RGBA")

        if self.user_img is not None:
            scaled_w = max(
                1,
                int(
                    round(
                        self.user_img.width
                        * self.img_scale
                        * self.editor_display_scale
                    )
                ),
            )
            scaled_h = max(
                1,
                int(
                    round(
                        self.user_img.height
                        * self.img_scale
                        * self.editor_display_scale
                    )
                ),
            )
            scaled_user = self.user_img.resize(
                (scaled_w, scaled_h), Image.LANCZOS
            )

            paste_x = int(round(self.img_offset_x * self.editor_display_scale))
            paste_y = int(round(self.img_offset_y * self.editor_display_scale))

            checker_bg.paste(scaled_user, (paste_x, paste_y), scaled_user)

        self.editor_photo = ImageTk.PhotoImage(checker_bg)
        self.editor_canvas.delete("all")

        cx, cy = ui_w // 2, ui_h // 2
        x0, y0 = cx - disp_w // 2, cy - disp_h // 2
        self.editor_canvas.create_image(cx, cy, image=self.editor_photo)

        self.editor_canvas.create_rectangle(
            x0, y0, x0 + disp_w, y0 + disp_h, outline="#0284c7", width=2, dash=(4, 4)
        )

        img_info = (
            f"{os.path.basename(self.user_img_path)}"
            if self.user_img_path
            else "No Image"
        )
        pos_info = f"Pos: ({int(round(self.img_offset_x))}, {int(round(self.img_offset_y))})"
        scale_info = f"Scale: {int(round(self.img_scale * 100))}%"
        self.editor_dim_lbl.config(
            text=f"Smart Object Canvas: {int(self.editor_w)}x{int(self.editor_h)}px  |  Image: {img_info}  |  {pos_info}  |  {scale_info}"
        )

    def get_composite_design(self):
        """Generates full-resolution RGBA composite image of size (editor_w, editor_h)."""
        target_w_i, target_h_i = int(round(self.editor_w)), int(round(self.editor_h))
        full_canvas = Image.new("RGBA", (target_w_i, target_h_i), (0, 0, 0, 0))

        if self.user_img is not None:
            scaled_w = max(1, int(round(self.user_img.width * self.img_scale)))
            scaled_h = max(1, int(round(self.user_img.height * self.img_scale)))
            scaled_user = self.user_img.resize(
                (scaled_w, scaled_h), Image.LANCZOS
            )

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
            self.progress_label.config(
                text=f"Processing layer {current}/{total}: {layer_name}"
            )
            self.root.update_idletasks()

        try:
            self.progress_var.set(0)
            self.progress_label.config(
                text="Generating composite design and warping mockup..."
            )
            self.root.update_idletasks()

            composite_design = self.get_composite_design()

            final_img = render_full_mockup(
                self.psd,
                composite_design,
                fit_mode="none",
                use_mask=self.mask_var.get(),
                progress_callback=update_progress,
            )

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

        preview_img = pil_img.convert("RGB").resize(
            new_size, Image.Resampling.LANCZOS
        )
        self.preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self.preview_photo
        )
