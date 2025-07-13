#!/usr/bin/env python3
"""
CSV Time Series Plotter
A simple offline GUI application to upload CSV files and plot columns against time.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import os

# Try to import tkinterdnd2 for drag and drop
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

class CSVPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Time Series Plotter")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)
        
        # Center the window
        self.center_window()
        
        # Initialize variables
        self.df = None
        self.current_file = None
        self.current_view = "landing"  # landing, plotting
        
        # Enable drag and drop if available
        if HAS_DND:
            self.setup_drag_drop()
        
        # Create the GUI
        self.create_landing_page()
        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = 800
        height = 600
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        if HAS_DND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
            self.root.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.root.dnd_bind('<<DragLeave>>', self.on_drag_leave)
        
    def on_drop(self, event):
        """Handle dropped files"""
        if event.data:
            file_path = event.data.strip('{}')
            if file_path.lower().endswith('.csv'):
                self.load_csv_file(file_path)
            else:
                messagebox.showerror("Invalid File", "Please drop a CSV file (.csv extension)")
                self.reset_drop_zone_style()
    
    def on_drag_enter(self, event):
        """Visual feedback when dragging over the window"""
        if hasattr(self, 'drop_zone') and self.current_view == "landing":
            self.drop_zone.config(bg="#90EE90")  # Light green
            # Update all child labels
            for child in self.drop_zone.winfo_children():
                if isinstance(child, tk.Label):
                    # Special handling for the upload icon
                    text = child.cget('text')
                    if text == "📤":
                        child.config(bg="#90EE90", text="🎯", fg="#000000")
                    else:
                        child.config(bg="#90EE90", fg="#000000")
    
    def on_drag_leave(self, event):
        """Reset visual feedback when drag leaves"""
        if self.current_view == "landing":
            self.reset_drop_zone_style()
    
    def reset_drop_zone_style(self):
        """Reset drop zone to default style"""
        if hasattr(self, 'drop_zone') and self.current_view == "landing":
            self.drop_zone.config(bg="#e8e8e8")
            # Reset all child widgets to original colors
            for child in self.drop_zone.winfo_children():
                if isinstance(child, tk.Label):
                    text = child.cget('text')
                    if text == "🎯":  # Reset the drag icon
                        child.config(bg="#e8e8e8", fg="#000000", text="📤")
                    else:
                        child.config(bg="#e8e8e8", fg="#000000")
    
    def create_landing_page(self):
        """Create the main landing page with prominent drag and drop"""
        # Clear any existing content
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main container with visible border for debugging
        main_frame = tk.Frame(self.root, bg="#ffffff", relief="solid", borderwidth=2)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title section with visible background
        title_frame = tk.Frame(main_frame, bg="#f0f0f0", relief="raised", borderwidth=1)
        title_frame.pack(fill=tk.X, pady=(10, 20), padx=10)
        
        title_label = tk.Label(
            title_frame,
            text="CSV Time Series Plotter",
            font=("Helvetica", 22, "bold"),
            bg="#f0f0f0",
            fg="#000000",
            padx=20,
            pady=10
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Upload your CSV file to create beautiful time series plots",
            font=("Helvetica", 11),
            bg="#f0f0f0",
            fg="#000000",
            padx=10,
            pady=5
        )
        subtitle_label.pack()
        
        # Drag and drop zone (main feature) with bright colors for visibility
        drop_frame = tk.Frame(main_frame, bg="#ffffff", relief="sunken", borderwidth=2)
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Create a more prominent visual drop zone
        if HAS_DND:
            main_text = "DRAG & DROP YOUR CSV FILE HERE"
            sub_text = "or click anywhere in this area to browse"
        else:
            main_text = "CLICK HERE TO SELECT CSV FILE"
            sub_text = "Browse for files using the button below"
        
        self.drop_zone = tk.Frame(
            drop_frame,
            bg="#e8e8e8",
            relief="groove",
            borderwidth=5
        )
        self.drop_zone.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Upload icon (large visual element)
        icon_label = tk.Label(
            self.drop_zone,
            text="📤",
            font=("Helvetica", 40, "bold"),
            bg="#e8e8e8",
            fg="#000000"
        )
        icon_label.pack(pady=(30, 15))
        
        # Main instruction text
        main_label = tk.Label(
            self.drop_zone,
            text=main_text,
            font=("Helvetica", 16, "bold"),
            bg="#e8e8e8",
            fg="#000000",
            wraplength=400
        )
        main_label.pack(pady=(0, 10))
        
        # Sub instruction text
        sub_label = tk.Label(
            self.drop_zone,
            text=sub_text,
            font=("Helvetica", 12),
            bg="#e8e8e8",
            fg="#000000",
            wraplength=400
        )
        sub_label.pack(pady=(5, 30))
        
        # Make the entire drop zone clickable with simple hover effects
        def on_enter(e):
            if self.current_view == "landing":
                self.drop_zone.config(bg="#d0d0d0")
                for child in self.drop_zone.winfo_children():
                    if isinstance(child, tk.Label):
                        child.config(bg="#d0d0d0")
        
        def on_leave(e):
            if self.current_view == "landing":
                self.drop_zone.config(bg="#e8e8e8")
                for child in self.drop_zone.winfo_children():
                    if isinstance(child, tk.Label):
                        child.config(bg="#e8e8e8")
        
        for widget in [self.drop_zone, icon_label, main_label, sub_label]:
            widget.bind("<Button-1>", lambda e: self.browse_file())
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.config(cursor="hand2")
        
        # Button section with visible background
        button_frame = tk.Frame(main_frame, bg="#f8f8f8", relief="raised", borderwidth=1)
        button_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Style the button
        style = ttk.Style()
        style.configure("Large.TButton", font=("Helvetica", 12, "bold"), padding=12)
        
        browse_btn = ttk.Button(
            button_frame,
            text="📁 Browse Files",
            command=self.browse_file,
            style="Large.TButton"
        )
        browse_btn.pack(pady=15)
        
        # Alternative text
        alt_text = tk.Label(
            button_frame,
            text="Click the gray area above or use this button",
            font=("Helvetica", 10),
            fg="#000000",
            bg="#f8f8f8"
        )
        alt_text.pack(pady=(0, 15))
        
        # Info section with visible border
        info_frame = tk.Frame(main_frame, bg="#eeeeee", relief="solid", borderwidth=1)
        info_frame.pack(fill=tk.X, pady=(5, 10), padx=10)
        
        info_text = "Supported formats: CSV files (.csv)\n"
        if HAS_DND:
            info_text += "💡 You can drag files directly from Finder"
        else:
            info_text += "💡 Install tkinterdnd2 for drag & drop support"
            
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=("Helvetica", 9),
            bg="#eeeeee",
            fg="#000000",
            justify=tk.CENTER,
            pady=10
        )
        info_label.pack()
        
        self.current_view = "landing"
        
    def create_plotting_interface(self):
        """Create the plotting interface after file is loaded"""
        # Clear existing content
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main frame
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header with file info and back button
        header_frame = tk.Frame(main_frame, bg="white")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Back button
        back_btn = ttk.Button(
            header_frame,
            text="← Back to Upload",
            command=self.create_landing_page
        )
        back_btn.pack(side=tk.LEFT)
        
        # File info
        filename = os.path.basename(self.current_file) if self.current_file else "Unknown file"
        file_info = tk.Label(
            header_frame,
            text=f"📊 {filename} | {len(self.df)} rows × {len(self.df.columns)} columns",
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#333333"
        )
        file_info.pack(side=tk.RIGHT)
        
        # Controls frame
        controls_frame = tk.LabelFrame(main_frame, text="Plot Configuration", bg="white", font=("Arial", 10, "bold"))
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Column selection
        col_frame = tk.Frame(controls_frame, bg="white")
        col_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Time column
        tk.Label(col_frame, text="Time Column (X-axis):", font=("Arial", 10), bg="white").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.time_combo = ttk.Combobox(col_frame, values=list(self.df.columns), width=25)
        self.time_combo.grid(row=0, column=1, padx=(0, 20))
        
        # Data column  
        tk.Label(col_frame, text="Data Column (Y-axis):", font=("Arial", 10), bg="white").grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.data_combo = ttk.Combobox(col_frame, values=list(self.df.columns), width=25)
        self.data_combo.grid(row=0, column=3)
        
        # Auto-select columns
        self.auto_select_columns()
        
        # Plot button
        plot_btn = ttk.Button(
            controls_frame,
            text="📈 Generate Plot",
            command=self.plot_data,
            style="Large.TButton"
        )
        plot_btn.pack(pady=(0, 15))
        
        # Plot area
        self.plot_frame = tk.LabelFrame(main_frame, text="Plot", bg="white", font=("Arial", 10, "bold"))
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Select columns and click 'Generate Plot'", fontsize=14)
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.current_view = "plotting"
        
    def auto_select_columns(self):
        """Auto-select appropriate columns"""
        columns = list(self.df.columns)
        
        # Try to auto-select time column
        time_keywords = ['time', 'timestamp', 'date', 'datetime', 'Time', 'Timestamp', 'Date', 'DateTime']
        for keyword in time_keywords:
            for col in columns:
                if keyword.lower() in col.lower():
                    self.time_combo.set(col)
                    break
            if self.time_combo.get():
                break
        
        # If no time column found, select first column
        if not self.time_combo.get() and columns:
            self.time_combo.set(columns[0])
        
        # Select second column as data column by default
        if len(columns) > 1:
            selected_time = self.time_combo.get()
            for col in columns:
                if col != selected_time:
                    self.data_combo.set(col)
                    break
        elif columns:
            self.data_combo.set(columns[0])
        
    def browse_file(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_csv_file(file_path)
    
    def load_csv_file(self, file_path):
        """Load CSV file and switch to plotting interface"""
        try:
            # Read the CSV file
            self.df = pd.read_csv(file_path)
            self.current_file = file_path
            
            # Switch to plotting interface
            self.create_plotting_interface()
            
            # Show success message
            filename = os.path.basename(file_path)
            messagebox.showinfo(
                "File Loaded Successfully!", 
                f"📊 {filename}\n\n"
                f"Rows: {len(self.df):,}\n"
                f"Columns: {len(self.df.columns)}\n\n"
                f"Ready to create your plot!"
            )
            
        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to load CSV file:\n\n{str(e)}")
            if self.current_view == "landing":
                self.reset_drop_zone_style()
                
    def plot_data(self):
        """Generate and display the plot"""
        if self.df is None:
            messagebox.showerror("Error", "No CSV file loaded")
            return
            
        time_col = self.time_combo.get()
        data_col = self.data_combo.get()
        
        if not time_col or not data_col:
            messagebox.showerror("Error", "Please select both time and data columns")
            return
            
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Get data
            time_data = self.df[time_col]
            y_data = self.df[data_col]
            
            # Try to parse time data
            x_data = self.parse_time_data(time_data)
            
            # Create the plot
            self.ax.plot(x_data, y_data, linewidth=2, color='#1976d2', alpha=0.8)
            self.ax.set_title(f"{data_col} vs {time_col}", fontsize=14, pad=15)
            self.ax.set_xlabel(time_col, fontsize=12)
            self.ax.set_ylabel(data_col, fontsize=12)
            self.ax.grid(True, alpha=0.3)
            
            # Format x-axis if it's datetime
            if pd.api.types.is_datetime64_any_dtype(x_data):
                self.fig.autofmt_xdate()
            
            # Adjust layout and refresh
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Plotting Error", f"Failed to create plot:\n\n{str(e)}")
            
    def parse_time_data(self, time_data):
        """Try to parse time data into appropriate format"""
        # If already numeric, return as is
        if pd.api.types.is_numeric_dtype(time_data):
            return time_data
            
        # Try to parse as datetime
        try:
            return pd.to_datetime(time_data)
        except:
            pass
            
        # If datetime parsing fails, try to convert to numeric
        try:
            return pd.to_numeric(time_data)
        except:
            pass
            
        # If all else fails, use index
        return range(len(time_data))

def main():
    try:
        # Initialize root window with drag and drop support if available
        if HAS_DND:
            root = TkinterDnD.Tk()
        else:
            root = tk.Tk()
        
        # Ensure window appears and is brought to front
        root.lift()
        root.attributes('-topmost', True)
        root.after(100, lambda: root.attributes('-topmost', False))
        
        app = CSVPlotter(root)
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 