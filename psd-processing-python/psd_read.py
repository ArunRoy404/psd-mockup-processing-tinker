import tkinter as tk
from tkinter import filedialog, messagebox
from psd_tools import PSDImage

def open_and_show_psd():
    # 1. Initialize hidden Tkinter root for the file dialog
    root = tk.Tk()
    root.withdraw()

    # 2. Select the PSD file
    file_path = filedialog.askopenfilename(
        title="Select PSD File",
        filetypes=[("Photoshop files", "*.psd *.psb")]
    )

    if not file_path:
        return

    try:
        print(f"Loading {file_path}...")
        # 3. Load the PSD
        psd = PSDImage.open(file_path)
        
        # 4. Convert the composite (merged) image to a PIL object
        # This represents exactly what you see when you save a PSD with 'Maximize Compatibility'
        merged_image = psd.composite()
        
        # 5. Show the image using the default system viewer
        merged_image.show()
        
        print("Success! Image displayed.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open PSD: {e}")
    finally:
        root.destroy()

if __name__ == "__main__":
    open_and_show_psd()