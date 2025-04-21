import nibabel as nib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from pathlib import Path
import sys
import os

class BrainViewer:
    def __init__(self, default_opacity: float = 0.7):
        self.data = None
        self.filename = None
        self.affine = None
        self.header = None
        self.current_coords = None
        self.brightness = 0
        self.contrast = 1
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Main window
        self.root = tk.Tk()
        self.root.title("Brain Viewer")
        self.root.geometry("1200x800")
        
        # Create frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.viz_frame = ttk.Frame(self.root)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_controls()
        self.create_visualization_area()
        
        # Initialize cursor positions
        self.cursor_positions = {
            'axial': [0, 0],
            'sagittal': [0, 0],
            'coronal': [0, 0]
        }
        
        self.mouse_pressed = False
        self.mouse_over_axes = None
        
        # Add view state tracking
        self.view_states = {
            'axial': {'xlim': None, 'ylim': None},
            'sagittal': {'xlim': None, 'ylim': None},
            'coronal': {'xlim': None, 'ylim': None}
        }
        
        # Store references to plot elements
        self.image_plots = {'axial': None, 'sagittal': None, 'coronal': None}
        self.crosshair_lines = {
            'axial': {'vline': None, 'hline': None},
            'sagittal': {'vline': None, 'hline': None},
            'coronal': {'vline': None, 'hline': None}
        }
        self.temp_lines = {
            'axial': {'vline': None, 'hline': None},
            'sagittal': {'vline': None, 'hline': None},
            'coronal': {'vline': None, 'hline': None}
        }

    def create_controls(self):
        # File selection
        ttk.Label(self.control_frame, text="File Selection:").pack(anchor="w", pady=(0, 5))
        ttk.Button(self.control_frame, text="Select File", command=self.load_nifti).pack(fill="x", pady=(0, 10))
        
        # Slice controls with both slider and entry
        self.slice_controls = {}
        self.slice_entries = {}
        
        for plane in ['Axial', 'Sagittal', 'Coronal']:
            frame = ttk.Frame(self.control_frame)
            frame.pack(fill="x", pady=(0, 10))
            
            ttk.Label(frame, text=f"{plane} Slice:").pack(side=tk.LEFT)
            
            entry_var = tk.StringVar(value="0")
            entry = ttk.Entry(frame, textvariable=entry_var, width=8)
            entry.pack(side=tk.RIGHT)
            
            var = tk.IntVar(value=0)
            scale = ttk.Scale(
                self.control_frame,
                from_=0,
                to=100,
                orient="horizontal",
                variable=var,
                command=lambda v, p=plane.lower(): self.on_slider_change(v, p)
            )
            scale.pack(fill="x", pady=(0, 10))
            
            self.slice_controls[plane.lower()] = var
            self.slice_entries[plane.lower()] = entry_var
            
            entry.bind('<Return>', lambda e, p=plane.lower(): self.on_entry_change(p))
            entry.bind('<FocusOut>', lambda e, p=plane.lower(): self.on_entry_change(p))
        
        # Brightness and contrast controls
        ttk.Label(self.control_frame, text="Brightness:").pack(anchor="w")
        self.brightness_var = tk.DoubleVar(value=0)
        ttk.Scale(
            self.control_frame,
            from_=100,
            to=-100,
            orient="horizontal",
            variable=self.brightness_var,
            command=self.update_display
        ).pack(fill="x", pady=(0, 10))
        
        ttk.Label(self.control_frame, text="Contrast:").pack(anchor="w")
        self.contrast_var = tk.DoubleVar(value=0)
        ttk.Scale(
            self.control_frame,
            from_=100,
            to=-100,
            orient="horizontal",
            variable=self.contrast_var,
            command=self.update_display
        ).pack(fill="x", pady=(0, 10))

    def create_visualization_area(self):
        self.fig = Figure(figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        
        # Initialize subplots
        self.axes = {
            'axial': self.fig.add_subplot(131, title='Axial View'),
            'sagittal': self.fig.add_subplot(132, title='Sagittal View'),
            'coronal': self.fig.add_subplot(133, title='Coronal View')
        }
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('axes_enter_event', self.on_axes_enter)
        self.canvas.mpl_connect('axes_leave_event', self.on_axes_leave)
        
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self.viz_frame)
        toolbar.update()
        
        self.fig.tight_layout()

    def initialize_slice_positions(self):
        if self.data is None:
            return
            
        # Set each slider to middle of its range
        for i, (plane, var) in enumerate(self.slice_controls.items()):
            dim_index = {'sagittal': 0, 'coronal': 1, 'axial': 2}[plane]
            max_val = self.data.shape[dim_index] - 1
            middle_val = max_val // 2
            
            var.set(middle_val)
            self.slice_entries[plane].set(str(middle_val))
            
            # Configure slider range
            for child in self.control_frame.winfo_children():
                if isinstance(child, ttk.Scale) and child.cget('variable') == str(var):
                    child.configure(from_=0, to=max_val)
                    break

    def get_slice_display_params(self):
        brightness = self.brightness_var.get() / 100
        contrast = (self.contrast_var.get() + 100) / 100
        
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        data_range = data_max - data_min
        
        return {
            'vmin': data_min + brightness * data_range,
            'vmax': data_max * contrast
        }

    def store_view_states(self):
        """Store the current view limits for all axes"""
        for view_name, ax in self.axes.items():
            self.view_states[view_name]['xlim'] = ax.get_xlim()
            self.view_states[view_name]['ylim'] = ax.get_ylim()

    def restore_view_states(self):
        """Restore the stored view limits for all axes"""
        for view_name, ax in self.axes.items():
            if self.view_states[view_name]['xlim'] is not None:
                ax.set_xlim(self.view_states[view_name]['xlim'])
                ax.set_ylim(self.view_states[view_name]['ylim'])

    def initialize_plots(self):
        """Initialize the image plots and crosshair lines"""
        if self.data is None:
            return
            
        display_params = self.get_slice_display_params()
        x = self.slice_controls['sagittal'].get()
        y = self.slice_controls['coronal'].get()
        z = self.slice_controls['axial'].get()
        
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        # Initialize plots for each view
        slices = {
            'axial': np.flip(np.rot90(self.data[:, :, z]), axis=1),
            'sagittal': np.flip(np.rot90(self.data[x, :, :]), axis=1),
            'coronal': np.flip(np.rot90(self.data[:, y, :]), axis=1)
        }
        
        for view_name, ax in self.axes.items():
            # Create image plot
            self.image_plots[view_name] = ax.imshow(slices[view_name], cmap='gray', **display_params)
            ax.set_aspect('equal')
            
            # Create permanent crosshair lines
            self.crosshair_lines[view_name]['vline'] = ax.axvline(x=0, color='r', alpha=0.5)
            self.crosshair_lines[view_name]['hline'] = ax.axhline(y=0, color='r', alpha=0.5)
            
            # Create temporary (hover) crosshair lines
            self.temp_lines[view_name]['vline'] = ax.axvline(x=0, color='b', alpha=0.3, linestyle='--', visible=False)
            self.temp_lines[view_name]['hline'] = ax.axhline(y=0, color='b', alpha=0.3, linestyle='--', visible=False)
            
            slice_indices = {
                'a': z,
                's': x,
                'c': y
            }
            ax.set_title(f'{view_name.capitalize()} ({view_name[0]}={slice_indices[view_name[0]]})')
        
        self.update_crosshairs()
        self.fig.tight_layout()
        self.canvas.draw()

    def update_crosshairs(self):
        """Update crosshair positions without redrawing images"""
        if self.data is None:
            return
            
        x = self.slice_controls['sagittal'].get()
        y = self.slice_controls['coronal'].get()
        z = self.slice_controls['axial'].get()
        
        # Update crosshair positions for each view
        # Axial view
        self.crosshair_lines['axial']['vline'].set_xdata([x, x])
        self.crosshair_lines['axial']['hline'].set_ydata([self.data.shape[1] - y - 1] * 2)
        
        # Sagittal view
        self.crosshair_lines['sagittal']['vline'].set_xdata([self.data.shape[1] - y - 1] * 2)
        self.crosshair_lines['sagittal']['hline'].set_ydata([z, z])
        
        # Coronal view
        self.crosshair_lines['coronal']['vline'].set_xdata([x, x])
        self.crosshair_lines['coronal']['hline'].set_ydata([z, z])
        
        # Update titles
        for view_name, ax in self.axes.items():
            slice_indices = {
                'a': z,
                's': x,
                'c': y
            }
            ax.set_title(f'{view_name.capitalize()} ({view_name[0]}={slice_indices[view_name[0]]})')
        
        self.canvas.draw()

    def update_image_data(self):
        """Update only the image data when brightness/contrast changes"""
        if self.data is None:
            return
            
        display_params = self.get_slice_display_params()
        x = self.slice_controls['sagittal'].get()
        y = self.slice_controls['coronal'].get()
        z = self.slice_controls['axial'].get()
        
        slices = {
            'axial': np.flip(np.rot90(self.data[:, :, z]), axis=1),
            'sagittal': np.flip(np.rot90(self.data[x, :, :]), axis=1),
            'coronal': np.flip(np.rot90(self.data[:, y, :]), axis=1)
        }
        
        for view_name, img_plot in self.image_plots.items():
            img_plot.set_data(slices[view_name])
            img_plot.set_clim(**display_params)
        
        self.canvas.draw()

    def update_temp_crosshairs(self, event):
        """Update temporary crosshairs for hover effect"""
        if event.inaxes is None or self.data is None:
            return
            
        view_name = next((k for k, v in self.axes.items() if v == event.inaxes), None)
        if view_name is None:
            return
            
        # Hide all temporary crosshairs first
        for lines in self.temp_lines.values():
            lines['vline'].set_visible(False)
            lines['hline'].set_visible(False)
        
        # Show and update temporary crosshairs for current view
        self.temp_lines[view_name]['vline'].set_visible(True)
        self.temp_lines[view_name]['hline'].set_visible(True)
        self.temp_lines[view_name]['vline'].set_xdata([event.xdata, event.xdata])
        self.temp_lines[view_name]['hline'].set_ydata([event.ydata, event.ydata])
        
        self.canvas.draw()

    def load_nifti(self):
        try:
            filename = filedialog.askopenfilename(
                title="Select NIfTI file",
                filetypes=[("NIfTI files", ".nii;.nii.gz")]
            )
            
            if not filename:
                return
                
            self.filename = filename
            self.logger.info(f"Loading file: {os.path.basename(self.filename)}")
            
            nifti_img = nib.load(self.filename)
            self.data = nifti_img.get_fdata()
            self.affine = nifti_img.affine
            self.header = nifti_img.header
            
            self.initialize_slice_positions()
            self.initialize_plots()
            messagebox.showinfo("Success", "File loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading file: {str(e)}")
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def on_slider_change(self, value, plane):
        self.slice_entries[plane].set(str(int(float(value))))
        self.update_image_data()
        self.update_crosshairs()

    def on_entry_change(self, plane):
        try:
            value = int(self.slice_entries[plane].get())
            if self.data is not None:
                dim_index = {'sagittal': 0, 'coronal': 1, 'axial': 2}[plane]
                max_val = self.data.shape[dim_index] - 1
                value = max(0, min(value, max_val))
            self.slice_controls[plane].set(value)
            self.slice_entries[plane].set(str(value))
            self.update_image_data()
            self.update_crosshairs()
        except ValueError:
            self.slice_entries[plane].set(str(self.slice_controls[plane].get()))

    def on_axes_enter(self, event):
        self.mouse_over_axes = event.inaxes

    def on_axes_leave(self, event):
        self.mouse_over_axes = None
        self.update_image_data()
        self.update_crosshairs()

    def on_motion(self, event):
        if event.inaxes is not None and self.data is not None:
            if self.mouse_pressed:
                self.update_cursor_position(event)
            else:
                self.update_temp_crosshairs(event)

    def on_click(self, event):
        if event.inaxes is not None and self.data is not None:
            self.mouse_pressed = True
            self.update_cursor_position(event)

    def on_release(self, event):
        self.mouse_pressed = False

    def update_display(self, _=None):
        self.update_image_data()
        self.update_crosshairs()

    def update_cursor_position(self, event):
        if not event.inaxes or self.data is None:
            return
        
        view_name = next((k for k, v in self.axes.items() if v == event.inaxes), None)
        if view_name is None:
            return
        
        x = int(np.clip(event.xdata, 0, self.data.shape[0] - 1))
        y = int(np.clip(event.ydata, 0, self.data.shape[1] - 1))
        
        if view_name == 'axial':
            self.slice_controls['sagittal'].set(x)
            coronal_y = self.data.shape[1] - y - 1
            self.slice_controls['coronal'].set(coronal_y)
            self.slice_entries['sagittal'].set(str(x))
            self.slice_entries['coronal'].set(str(coronal_y))
        elif view_name == 'sagittal':
            coronal_x = self.data.shape[1] - x - 1
            self.slice_controls['coronal'].set(coronal_x)
            self.slice_controls['axial'].set(y)
            self.slice_entries['coronal'].set(str(coronal_x))
            self.slice_entries['axial'].set(str(y))
        elif view_name == 'coronal':
            self.slice_controls['sagittal'].set(x)
            self.slice_controls['axial'].set(y)
            self.slice_entries['sagittal'].set(str(x))
            self.slice_entries['axial'].set(str(y))
        
        self.update_image_data()
        self.update_crosshairs()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    viewer = BrainViewer()
    viewer.run()
