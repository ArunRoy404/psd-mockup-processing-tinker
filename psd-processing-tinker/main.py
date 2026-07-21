"""
Main entry point for PSD Smart Object Interactive Mockup Application.
"""

import os
import sys

# Ensure current directory is in Python path for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import tkinter as tk
from gui import MockupApp


def main():
    root = tk.Tk()
    app = MockupApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
