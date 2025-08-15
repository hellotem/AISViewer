#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 AIS Trajectory Viewer
- Light background
- Option to show basemap
- Highlight selected data point in a different color
- Show 4-column sortable table instead of simple list
- Robust exception handling
- Display point info on mouse hover instead of click
- Added basemap overlay functionality
- Size-adaptive vessel icon for simulation
"""

import sys
import math
import os
import requests
import io
from dataclasses import dataclass
from typing import Dict, List, Optional
from PIL import Image

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFormLayout, QMessageBox, QSplitter, QGroupBox, QCheckBox,
    QHeaderView
)
from PyQt5.QtGui import QCursor

os.environ['QT_XCB_GL_INTEGRATION'] = 'none'

# Custom QTableWidgetItem that sorts numerically
class NumericTableWidgetItem(QTableWidgetItem):
    def __init__(self, text, numeric_value):
        super().__init__(text)
        self.numeric_value = numeric_value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self.numeric_value < other.numeric_value
        return super().__lt__(other)

import pyqtgraph as pg
import pandas as pd
import numpy as np

# Utilities
def _norm_col(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())

COLUMN_ALIASES = {
    'time': {'timestamp', 'time', 'datetime', 'basedatetime', 'datetimestamp'},
    'lat': {'lat', 'latitude', 'y'},
    'lon': {'lon', 'long', 'longitude', 'x'},
    'sog': {'sog', 'speed', 'speedoverground'},
    'cog': {'cog', 'course', 'courseoverground', 'heading'},
    'status': {'status', 'navstatus', 'nav_status'},
    'mmsi': {'mmsi', 'imo', 'vesselid', 'vessel_id', 'shipid'},
}

def map_columns(df: pd.DataFrame) -> Dict[str, str]:
    canon = {}
    norm_map = {_norm_col(c): c for c in df.columns}
    for target, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            key = _norm_col(a)
            if key in norm_map:
                canon[target] = norm_map[key]
                break
    missing = [k for k in ['time', 'lat', 'lon'] if k not in canon]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return canon

def parse_time_series(series: pd.Series) -> pd.Series:
    try:
        s = pd.to_datetime(series, errors='coerce', utc=False, infer_datetime_format=True)
        if s.isna().all():
            s = pd.to_datetime(series.astype(float), unit='s', errors='coerce', utc=False)
    except Exception:
        s = pd.Series([pd.NaT]*len(series))
    return s

def haversine_km(lat1, lon1, lat2, lon2):
    try:
        R = 6371.0088
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
    except Exception:
        return np.zeros_like(lat1)

def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile numbers"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """Convert tile numbers to lat/lon"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

class TileLoader(QThread):
    """Background thread to load map tiles"""
    tile_loaded = pyqtSignal(int, int, int, object)  # x, y, zoom, image_array
    
    def __init__(self):
        super().__init__()
        self.tiles_to_load = []
        self.running = True
        self._lock = False  # Simple lock to prevent issues
        
    def add_tile(self, x, y, zoom):
        if not self._lock:
            self.tiles_to_load.append((x, y, zoom))
        
    def run(self):
        self._lock = True
        try:
            while self.running and self.tiles_to_load:
                if not self.tiles_to_load:
                    break
                    
                x, y, zoom = self.tiles_to_load.pop(0)
                
                if not self.running:  # Check if we should stop
                    break
                    
                try:
                    # Use OpenStreetMap tiles
                    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                    headers = {'User-Agent': 'AIS Trajectory Viewer'}
                    
                    response = requests.get(url, headers=headers, timeout=10)  # Increased timeout
                    if response.status_code == 200:
                        img = Image.open(io.BytesIO(response.content))
                        img_array = np.array(img)
                        if self.running:  # Only emit if still running
                            self.tile_loaded.emit(x, y, zoom, img_array)
                except Exception as e:
                    print(f"Failed to load tile {x},{y},{zoom}: {e}")
                    
        except Exception as e:
            print(f"TileLoader error: {e}")
        finally:
            self._lock = False
                
    def stop(self):
        self.running = False
        self.tiles_to_load.clear()

@dataclass
class Trajectory:
    path: str
    df: pd.DataFrame
    columns: Dict[str, str]

    @property
    def mmsi(self) -> Optional[str]:
        col = self.columns.get('mmsi')
        if col and col in self.df.columns:
            vals = self.df[col].dropna().unique()
            if len(vals) == 1:
                return str(vals[0])
            elif len(vals) > 1:
                return f"{str(vals[0])} (+{len(vals)-1} more)"
        return None

    def stats(self) -> Dict[str, str]:
        try:
            tcol = self.columns['time']
            lat = self.columns['lat']
            lon = self.columns['lon']
            n = len(self.df)
            tmin = self.df[tcol].min()
            tmax = self.df[tcol].max()
            dur = (tmax - tmin) if pd.notna(tmax) and pd.notna(tmin) else pd.NaT
            dlat = self.df[lat].to_numpy()
            dlon = self.df[lon].to_numpy()
            dist_km = np.nansum(haversine_km(dlat[:-1], dlon[:-1], dlat[1:], dlon[1:])) if n >= 2 else 0.0
            return {
                'Vessel ID': self.mmsi or '—',
                'Points': f"{n}",
                'Time start': f"{tmin}",
                'Time end': f"{tmax}",
                'Duration': (str(dur) if pd.notna(dur) else '—'),
                'Total distance': f"{dist_km:.3f} km",
            }
        except Exception:
            return {
                'Vessel ID': 'â€"',
                'Points': '0',
                'Time start': 'â€"',
                'Time end': 'â€"',
                'Duration': 'â€"',
                'Total distance': '0 km',
            }

    def table_data(self) -> Dict[str, str]:
        """Get data for table display"""
        try:
            tcol = self.columns['time']
            lat = self.columns['lat']
            lon = self.columns['lon']
            n = len(self.df)
            tmin = self.df[tcol].min()
            tmax = self.df[tcol].max()
            dur = (tmax - tmin) if pd.notna(tmax) and pd.notna(tmin) else pd.NaT
            dlat = self.df[lat].to_numpy()
            dlon = self.df[lon].to_numpy()
            dist_km = np.nansum(haversine_km(dlat[:-1], dlon[:-1], dlat[1:], dlon[1:])) if n >= 2 else 0.0
            
            # Format duration nicely and store numeric value for sorting
            if pd.notna(dur):
                total_seconds = int(dur.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                duration_str = f"{hours}h {minutes}m"
                duration_numeric = total_seconds  # Store seconds for sorting
            else:
                duration_str = "—"
                duration_numeric = 0
                
            return {
                'filename': os.path.basename(self.path),
                'points': n,
                'distance': dist_km,
                'duration': duration_str,
                'duration_numeric': duration_numeric
            }
        except Exception:
            return {
                'filename': os.path.basename(self.path),
                'points': 0,
                'distance': 0.0,
                'duration': "—",
                'duration_numeric': 0
            }

class AISViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIS Trajectory Viewer")
        self.resize(1200, 800)

        splitter = QSplitter()
        self.setCentralWidget(splitter)

        left = QWidget()
        left.setMaximumWidth(400)
        left_layout = QVBoxLayout(left)

        btn_row = QHBoxLayout()
        self.btn_import = QPushButton("Open Files…")
        btn_row.addWidget(self.btn_import)
        self.btn_import_folder = QPushButton("Open Folder…")
        btn_row.addWidget(self.btn_import_folder)
        self.btn_delete_selected = QPushButton("Delete Selected")
        self.btn_delete_selected.clicked.connect(self.delete_selected_rows)
        btn_row.addWidget(self.btn_delete_selected)

        btn_row2 = QHBoxLayout()
        self.btn_simulate = QPushButton("▶ Simulate Sailing")
        self.btn_simulate.setCheckable(True)
        self.btn_simulate.clicked.connect(self.on_simulate_toggle)
        btn_row2.addWidget(self.btn_simulate)
        self.btn_measure = QPushButton("Measure Distance")
        btn_row2.addWidget(self.btn_measure)
        self.btn_reset = QPushButton("Reset View")
        btn_row2.addWidget(self.btn_reset)


        self.btn_import_folder.clicked.connect(self.on_import_folder)

        self.btn_measure.setCheckable(True)
        self.measure_start = None          # QPointF
        self.measure_line   = None         # pg.PlotDataItem
        self.measure_text   = None         # pg.TextItem

        left_layout.addLayout(btn_row)
        left_layout.addLayout(btn_row2)

        # Add basemap checkbox
        basemap_layout = QHBoxLayout()
        self.simulate_speed_checkbox = QCheckBox("Simulate with SOG")
        basemap_layout.addWidget(self.simulate_speed_checkbox)       
        self.basemap_checkbox = QCheckBox("Show Basemap")
        self.basemap_checkbox.toggled.connect(self.on_basemap_toggled)
        basemap_layout.addWidget(self.basemap_checkbox)
        basemap_layout.addStretch()  # Push checkbox to left
        left_layout.addLayout(basemap_layout)

        # Create table widget instead of list
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels(['File Name', 'Points', 'Distance (km)', 'Duration'])
        
        # Enable sorting
        self.file_table.setSortingEnabled(True)
        
        # Remove grid lines
        self.file_table.setShowGrid(False)
        
        # Set column widths - make them user-resizable
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # File name - user resizable
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Points - user resizable
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Distance - user resizable
        header.setSectionResizeMode(3, QHeaderView.Interactive)  # Duration - user resizable
        
        # Set initial column widths
        self.file_table.setColumnWidth(0, 200)  # File name
        self.file_table.setColumnWidth(1, 80)   # Points
        self.file_table.setColumnWidth(2, 100)  # Distance
        self.file_table.setColumnWidth(3, 80)   # Duration
        
        # Set selection behavior
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.ExtendedSelection)  # was SingleSelection
        
        left_layout.addWidget(self.file_table)

        stats_group = QGroupBox("Trajectory Info")
        self.stats_form = QFormLayout()
        stats_group.setLayout(self.stats_form)
        left_layout.addWidget(stats_group)

        point_group = QGroupBox("Point Details")
        self.point_form = QFormLayout()
        point_group.setLayout(self.point_form)
        left_layout.addWidget(point_group)

        splitter.addWidget(left)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot = self.plot_widget.getPlotItem()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = None
        self.scatter = None
        self.selected_point = None
        
        # Basemap related attributes
        self.basemap_items = []  # Store basemap image items
        self.tile_loader = None
        self.current_trajectory = None
        
        # Simulation related attributes
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.simulation_index = 0
        self.simulation_active = False
        self.vessel_icon = None
        self.dynamic_curve = None
        self.simulation_speed_multiplier = 70 # Speed up simulation
        
        splitter.addWidget(self.plot_widget)

        self.trajectories: List[Trajectory] = []

        self.btn_import.clicked.connect(self.on_import)
        self.file_table.itemSelectionChanged.connect(self.on_select_file)
        self.btn_reset.clicked.connect(self.on_reset_view)
        # self.btn_zoom_in.clicked.connect(lambda: self.plot.getViewBox().scaleBy((0.8, 0.8)))
        # self.btn_zoom_out.clicked.connect(lambda: self.plot.getViewBox().scaleBy((1.25, 1.25)))

        pg.setConfigOptions(antialias=True)

        # Enable hover events
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)

    def create_vessel_icon(self, lat, lon, cog=None):
        """Create a vessel icon as a scatter plot item (like AIS points)"""
        # Create a larger, more visible scatter point for the vessel
        vessel_size = 15  # Fixed pixel size, stable across zoom levels
        
        # Choose symbol and color based on whether we have course data
        if cog is not None and pd.notna(cog):
            # Use triangle symbol and rotate it to show heading
            symbol = 'o'  # Triangle symbol
            brush = pg.mkBrush(255, 0, 0, 220)  # Red with transparency
        else:
            # Use circle if no heading data
            symbol = 'o'
            brush = pg.mkBrush(255, 100, 0, 220)  # Orange with transparency
            
        pen = pg.mkPen(width=2, color='darkred')
        
        return {
            'pos': (lon, lat),
            'size': vessel_size,
            'symbol': symbol,
            'brush': brush,
            'pen': pen,
            'data': {'vessel': True, 'cog': cog}
        }

    def on_simulate_toggle(self, checked):
        """Handle simulation toggle button"""
        # Disable if multiple files are selected
        selected_rows = sorted(set(idx.row() for idx in self.file_table.selectedIndexes()))
        if len(selected_rows) != 1:
            QMessageBox.warning(self, "Simulation Disabled", "Please select exactly one trajectory to simulate.")
            self.btn_simulate.setChecked(False)
            return

        if checked:
            self.start_simulation()
        else:
            self.stop_simulation()

    # def on_simulate_toggle(self, checked):
    #     """Handle simulation toggle button"""
    #     if checked:
    #         self.start_simulation()
    #     else:
    #         self.stop_simulation()

    def start_simulation(self):
        """Start the vessel sailing simulation"""
        if not self.current_trajectory:
            self.btn_simulate.setChecked(False)
            QMessageBox.warning(self, "No Trajectory", "Please select a trajectory first.")
            return

        # First, set the view range based on trajectory extent to prevent flashing
        self.set_view_to_trajectory_extent()

        self.simulation_active = True
        self.simulation_index = 0
        self.btn_simulate.setText("⏸ Stop Simulation")
        
        # Clear existing trajectory display
        if self.curve:
            self.plot.removeItem(self.curve)
            self.curve = None
        if self.scatter:
            self.plot.removeItem(self.scatter)
            self.scatter = None
        
        # Create dynamic curve for trajectory being drawn (start with empty data)
        self.dynamic_curve = pg.PlotCurveItem(pen=pg.mkPen(width=2, color='b'))
        self.dynamic_curve.setZValue(1000)
        self.plot.addItem(self.dynamic_curve)
        
        # Create vessel icon as a size-adaptive thin triangle
        # vessel_shape = self.create_vessel_icon(0,0)
        # self.vessel_icon = pg.PlotCurveItem()
        # self.vessel_icon.setZValue(3000)
        # self.plot.addItem(self.vessel_icon)
        self.vessel_icon = pg.ScatterPlotItem()
        self.vessel_icon.setZValue(3000)
        self.plot.addItem(self.vessel_icon)
        
        # Start the simulation timer
        self.update_simulation()  # First update immediately
        self.simulation_timer.start(50)  # Update every 50ms

    def set_view_to_trajectory_extent(self):
        """Set the plot view range to fit the entire trajectory with padding"""
        if not self.current_trajectory:
            return
            
        lat_col = self.current_trajectory.columns['lat']
        lon_col = self.current_trajectory.columns['lon']
        lats = self.current_trajectory.df[lat_col].to_numpy()
        lons = self.current_trajectory.df[lon_col].to_numpy()
        
        # Get bounds
        min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
        min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)
        
        # Add padding (10% of range)
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        if lat_range == 0:
            lat_range = 0.001  # Minimum range for single point
        if lon_range == 0:
            lon_range = 0.001
            
        lat_pad = lat_range * 0.1
        lon_pad = lon_range * 0.1
        
        # Set the view range and disable auto-range to prevent flashing
        self.plot.vb.disableAutoRange()
        self.plot.vb.setRange(
            xRange=[min_lon - lon_pad, max_lon + lon_pad],
            yRange=[min_lat - lat_pad, max_lat + lat_pad],
            padding=0
        )

    def stop_simulation(self):
        """Stop the vessel sailing simulation"""
        self.simulation_active = False
        self.simulation_timer.stop()
        self.btn_simulate.setText("▶ Simulate Sailing")
        self.btn_simulate.setChecked(False)
        
        # Remove simulation elements
        if self.vessel_icon:
            self.plot.removeItem(self.vessel_icon)
            self.vessel_icon = None
        if self.dynamic_curve:
            self.plot.removeItem(self.dynamic_curve)
            self.dynamic_curve = None
        
        # Re-enable auto-range and restore full trajectory display
        self.plot.vb.enableAutoRange()
        if self.current_trajectory:
            self.render_trajectory(self.current_trajectory)

    def update_simulation(self):
        """Update the simulation animation"""
        if not self.simulation_active or not self.current_trajectory:
            return
            
        df = self.current_trajectory.df
        if self.simulation_index >= len(df):
            # Animation complete
            self.stop_simulation()
            return
            
        lat_col = self.current_trajectory.columns['lat']
        lon_col = self.current_trajectory.columns['lon']
        time_col = self.current_trajectory.columns['time']
        cog_col = self.current_trajectory.columns.get('cog')
        
        # Get current position
        current_row = df.iloc[self.simulation_index]
        current_lat = current_row[lat_col]
        current_lon = current_row[lon_col]
        current_cog = current_row[cog_col] if cog_col else None
        
        # Update vessel position using scatter plot approach
        if self.vessel_icon:
            vessel_spot = self.create_vessel_icon(current_lat, current_lon, current_cog)
            
            # Clear previous vessel position and add new one
            self.vessel_icon.clear()
            self.vessel_icon.addPoints([vessel_spot])
        
        # Update dynamic trajectory (draw path up to current position)
        if self.dynamic_curve:
            path_lats = df[lat_col].iloc[:self.simulation_index + 1].to_numpy()
            path_lons = df[lon_col].iloc[:self.simulation_index + 1].to_numpy()
            self.dynamic_curve.setData(path_lons, path_lats)
        # Calculate time-based progression or use fixed increment

        if self.simulation_index < len(df) - 1:
            current_time = current_row[time_col]
            next_time = df.iloc[self.simulation_index + 1][time_col]

            if self.simulate_speed_checkbox.isChecked():
                # Simulate speed using SOG (knots) → convert to m/s
                sog_col = self.current_trajectory.columns.get('sog')
                if sog_col and sog_col in df.columns:
                    current_sog = current_row[sog_col]  # knots
                    if pd.notna(current_sog):
                        speed_mps = current_sog * 0.514444  # knots → m/s
                        time_diff_seconds = (next_time - current_time).total_seconds() if pd.notna(current_time) and pd.notna(next_time) else 1
                        # Steps based on distance traveled; slower vessels take smaller steps
                        steps = max(1, int(time_diff_seconds * (speed_mps / self.simulation_speed_multiplier)))
                    else:
                        steps = 1
                else:
                    steps = 1
            else:
                # Constant speed mode (original behavior)
                if pd.notna(current_time) and pd.notna(next_time):
                    time_diff_seconds = (next_time - current_time).total_seconds()
                    steps = max(1, int(time_diff_seconds / self.simulation_speed_multiplier))
                else:
                    steps = 1
        else:
            steps = 1

        # # Calculate time-based progression or use fixed increment
        # if self.simulation_index < len(df) - 1:
        #     # Try to use time difference for realistic speed
        #     current_time = current_row[time_col]
        #     next_time = df.iloc[self.simulation_index + 1][time_col]
            
        #     if pd.notna(current_time) and pd.notna(next_time):
        #         time_diff_seconds = (next_time - current_time).total_seconds()
        #         # Speed up by multiplier, minimum 1 step
        #         steps = max(1, int(time_diff_seconds / self.simulation_speed_multiplier))
        #     else:
        #         steps = 1
        # else:
        #     steps = 1
            
        self.simulation_index += steps

    def on_basemap_toggled(self, checked):
        """Handle basemap checkbox toggle"""
        if checked:
            self.load_basemap()
        else:
            self.clear_basemap()

    def clear_basemap(self):
        """Remove all basemap tiles from the plot"""
        try:
            for item in self.basemap_items:
                if item in self.plot.items:  # Check if item is still in the plot
                    self.plot.removeItem(item)
            self.basemap_items.clear()
            
            if self.tile_loader:
                self.tile_loader.stop()
                if self.tile_loader.isRunning():
                    self.tile_loader.wait(1000)  # Wait max 1 second
                self.tile_loader = None
        except Exception as e:
            print(f"Error clearing basemap: {e}")
            self.basemap_items.clear()  # Clear the list anyway

    def load_basemap(self):
        """Load basemap tiles for current view"""
        if not self.current_trajectory:
            return
            
        # Clear existing basemap
        self.clear_basemap()
        
        # Get trajectory bounds
        lat_col = self.current_trajectory.columns['lat']
        lon_col = self.current_trajectory.columns['lon']
        lats = self.current_trajectory.df[lat_col].to_numpy()
        lons = self.current_trajectory.df[lon_col].to_numpy()
        
        min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
        min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)
        
        # Add some padding
        lat_pad = (max_lat - min_lat) * 0.2  # Increased padding
        lon_pad = (max_lon - min_lon) * 0.2
        min_lat -= lat_pad
        max_lat += lat_pad
        min_lon -= lon_pad
        max_lon += lon_pad
        
        # Determine appropriate zoom level (simple heuristic)
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        max_range = max(lat_range, lon_range)
        
        # Adjust zoom levels for better tile coverage
        if max_range > 20:
            zoom = 5
        elif max_range > 10:
            zoom = 6
        elif max_range > 5:
            zoom = 7
        elif max_range > 2:
            zoom = 8
        elif max_range > 1:
            zoom = 9
        elif max_range > 0.5:
            zoom = 10
        elif max_range > 0.25:
            zoom = 11
        else:
            zoom = 12
            
        # Limit maximum number of tiles to prevent overloading
        max_tiles_per_axis = 6
        
        # Get tile bounds
        min_x, max_y = deg2num(min_lat, min_lon, zoom)
        max_x, min_y = deg2num(max_lat, max_lon, zoom)
        
        # Ensure we don't request too many tiles
        if (max_x - min_x) > max_tiles_per_axis:
            center_x = (min_x + max_x) // 2
            min_x = max(0, center_x - max_tiles_per_axis // 2)
            max_x = min_x + max_tiles_per_axis
            
        if (max_y - min_y) > max_tiles_per_axis:
            center_y = (min_y + max_y) // 2
            min_y = max(0, center_y - max_tiles_per_axis // 2)
            max_y = min_y + max_tiles_per_axis
        
        print(f"Loading basemap: zoom={zoom}, tiles=({min_x}-{max_x}, {min_y}-{max_y})")
        
        # Start tile loader
        self.tile_loader = TileLoader()
        self.tile_loader.tile_loaded.connect(self.on_tile_loaded)
        
        # Queue tiles for loading
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                self.tile_loader.add_tile(x, y, zoom)
                
        self.tile_loader.start()

    def on_tile_loaded(self, x, y, zoom, img_array):
        """Handle loaded tile"""
        if img_array is None:
            return
            
        # Get tile bounds in lat/lon
        lat_north, lon_west = num2deg(x, y, zoom)
        lat_south, lon_east = num2deg(x + 1, y + 1, zoom)
        
        # Flip the image vertically to correct orientation
        # PyQtGraph expects images with (0,0) at bottom-left, but PIL gives top-left
        img_flipped = np.flipud(img_array)
        
        # Create image item
        img_item = pg.ImageItem(img_flipped)
        
        # Set the image position and scale correctly
        # PyQtGraph ImageItem expects: setRect(x, y, width, height)
        # where (x,y) is the bottom-left corner
        img_item.setRect(lon_west, lat_south, lon_east - lon_west, lat_north - lat_south)
        
        # Add to plot with low z-value so it appears behind trajectories
        img_item.setZValue(-1000)
        self.plot.addItem(img_item)
        self.basemap_items.append(img_item)

    def add_trajectory_to_table(self, traj: Trajectory):
        data = traj.table_data()
        row = self.file_table.rowCount()
        self.file_table.insertRow(row)

        # Column 0: filename (store the trajectory for easy retrieval)
        name_item = QTableWidgetItem(data['filename'])
        name_item.setData(Qt.UserRole, traj)
        self.file_table.setItem(row, 0, name_item)

        # Col 1: points (numeric sort)
        self.file_table.setItem(
            row, 1, NumericTableWidgetItem(str(int(data['points'])), int(data['points']))
        )

        # Col 2: distance (numeric sort, show 3 decimals)
        self.file_table.setItem(
            row, 2, NumericTableWidgetItem(f"{float(data['distance']):.3f}", float(data['distance']))
        )

        # Col 3: duration (numeric sort by seconds, show "Hh Mm")
        secs = int(data.get('duration_numeric', 0))
        self.file_table.setItem(
            row, 3, NumericTableWidgetItem(data['duration'], secs)
        )

        # Optional: auto-fit columns to content
        self.file_table.resizeColumnsToContents()
    def delete_selected_rows(self):
        selected_rows = sorted(set(idx.row() for idx in self.file_table.selectedIndexes()), reverse=True)
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select one or more rows to delete.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the selected {len(selected_rows)} file(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(QCursor(Qt.WaitCurosor))
        for row in selected_rows:
            item = self.file_table.item(row, 0)
            if item:
                traj = item.data(Qt.UserRole)
                if traj in self.trajectories:
                    self.trajectories.remove(traj)
            self.file_table.removeRow(row)

        # Clear trajectory view if deleted current
        if self.current_trajectory and self.current_trajectory not in self.trajectories:
            self.current_trajectory = None
            self.plot.clear()
            self.clear_form_layout(self.stats_form)
            self.clear_form_layout(self.point_form)
        QApplication.restoreOverrideCursor()

    def on_import_folder(self):
        """Import every CSV file found in the selected folder."""
        try:
            folder = QFileDialog.getExistingDirectory(
                self, 
                "Select Folder with CSV files",
                "",  # Default directory
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            if not folder:
                return

            # Check if folder exists and is accessible
            if not os.path.exists(folder) or not os.path.isdir(folder):
                QMessageBox.warning(self, "Invalid Folder", "Selected folder does not exist or is not accessible.")
                return

            try:
                csv_files = [os.path.join(folder, f) for f in os.listdir(folder)
                            if f.lower().endswith('.csv') and os.path.isfile(os.path.join(folder, f))]
            except PermissionError:
                QMessageBox.warning(self, "Permission Error", "Cannot access the selected folder. Permission denied.")
                return
            except Exception as e:
                QMessageBox.warning(self, "Folder Error", f"Error reading folder: {str(e)}")
                return

            if not csv_files:
                QMessageBox.information(self, "No CSV Files", "No CSV files found in the selected folder.")
                return

            # Set waiting cursor for long operation
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            successful_loads = 0
            failed_loads = 0
            
            try:
                for path in csv_files:
                    try:
                        self._load_single_file(path)
                        successful_loads += 1
                    except Exception as e:
                        failed_loads += 1
                        print(f"Failed to load {path}: {e}")  # Log to console instead of showing dialog for each failure
                        
            finally:
                QApplication.restoreOverrideCursor()
                
            # Show summary message
            if successful_loads > 0:
                message = f"Successfully loaded {successful_loads} files."
                if failed_loads > 0:
                    message += f" Failed to load {failed_loads} files."
                QMessageBox.information(self, "Import Complete", message)
            else:
                QMessageBox.warning(self, "Import Failed", f"Failed to load all {failed_loads} files.")
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {str(e)}")

    def _load_single_file(self, path):
        """Load a single CSV file - separated for reuse"""
        if not os.path.exists(path):
            raise ValueError("File does not exist")
            
        if os.path.getsize(path) == 0:
            raise ValueError("File is empty")
            
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV has no data rows")
            
        cols = map_columns(df)
        df[cols['time']] = parse_time_series(df[cols['time']])
        df[cols['lat']] = pd.to_numeric(df[cols['lat']], errors='coerce')
        df[cols['lon']] = pd.to_numeric(df[cols['lon']], errors='coerce')
        df = df.dropna(subset=[cols['time'], cols['lat'], cols['lon']]).reset_index(drop=True)
        
        if df.empty:
            raise ValueError("All required columns have NaN values after cleaning")
            
        df = df.sort_values(cols['time']).reset_index(drop=True)
        traj = Trajectory(path=path, df=df, columns=cols)
        self.trajectories.append(traj)
        self.add_trajectory_to_table(traj)

    def _toggle_measure_mode(self, checked):
        """Turn measuring mode on/off."""
        if not checked:
            self._clear_measure_overlay()
            self.measure_start = None

    def _clear_measure_overlay(self):
        """Remove line + text from the plot."""
        if self.measure_line is not None:
            self.plot.removeItem(self.measure_line)
            self.measure_line = None
        if self.measure_text is not None:
            self.plot.removeItem(self.measure_text)
            self.measure_text = None

    def _on_mouse_click(self, ev):
        """Left-clicks while measuring define start/end points."""
        if not self.btn_measure.isChecked() or ev.button() != Qt.LeftButton:
            return

        pos = self.plot.vb.mapSceneToView(ev.scenePos())

        if self.measure_start is None:               # first click → start
            self.measure_start = pos
            self._clear_measure_overlay()

            # Store current view range to prevent auto-ranging
            current_range = self.plot.vb.viewRange()
            
            # tiny non-zero line prevents auto-range collapse
            eps = 1e-4
            self.measure_line = pg.PlotDataItem(
                [pos.x(), pos.x() + eps], [pos.y(), pos.y() + eps],
                pen=pg.mkPen('m', width=2, style=Qt.DashLine))
            
            # Disable auto-range temporarily
            self.plot.vb.disableAutoRange()
            self.plot.addItem(self.measure_line)
            # Restore the view range to prevent shrinking
            self.plot.vb.setRange(xRange=current_range[0], yRange=current_range[1], padding=0)

            self.measure_text = pg.TextItem(
                anchor=(0, 1), color='m', fill=pg.mkBrush(255, 255, 255, 200))
            self.plot.addItem(self.measure_text)

            # live update while dragging
            self.plot_widget.scene().sigMouseMoved.connect(self._update_measure_drag)

        else:                                        # second click → finish
            self.plot_widget.scene().sigMouseMoved.disconnect(self._update_measure_drag)
            self._update_measure_line(pos)
            self.measure_start = None                # ready for next pair

    def _update_measure_drag(self, scene_pos):
        """Update the dashed line while the mouse moves."""
        if self.measure_start is None:
            return
        cur = self.plot.vb.mapSceneToView(scene_pos)
        self._update_measure_line(cur)

    def _update_measure_line(self, end_pt):
        """Draw line from start to end_pt and label the distance."""
        if self.measure_start is None:
            return

        x0, y0 = self.measure_start.x(), self.measure_start.y()
        x1, y1 = end_pt.x(), end_pt.y()

        self.measure_line.setData([x0, x1], [y0, y1])

        d_km = haversine_km(y0, x0, y1, x1)
        self.measure_text.setText(f"{d_km:.3f} km")
        self.measure_text.setPos(x1, y1)

    def on_import(self):
        """Import selected CSV files"""
        try:
            paths, _ = QFileDialog.getOpenFileNames(
                self, 
                "Select AIS CSV files", 
                '',  # Default directory
                "CSV Files (*.csv);;All Files (*)"
            )
            if not paths:
                return
            
            # Set waiting cursor for long operation
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            successful_loads = 0
            failed_loads = 0
            
            try:
                for path in paths:
                    try:
                        self._load_single_file(path)
                        successful_loads += 1
                    except Exception as e:
                        failed_loads += 1
                        print(f"Failed to load {path}: {e}")  # Log to console instead of showing dialog for each failure
                        
            finally:
                QApplication.restoreOverrideCursor()
                
            # Show summary message
            if successful_loads > 0:
                message = f"Successfully loaded {successful_loads} files."
                if failed_loads > 0:
                    message += f" Failed to load {failed_loads} files."
                QMessageBox.information(self, "Import Complete", message)
            else:
                QMessageBox.warning(self, "Import Failed", f"Failed to load all {failed_loads} files.")
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {str(e)}")

    def on_select_file(self):
        scroll_value = self.file_table.verticalScrollBar().value()
        selected_rows = sorted(set(idx.row() for idx in self.file_table.selectedIndexes()))
        if not selected_rows:
            return

        # Clear previous trajectory elements (but keep basemap if enabled)
        self.plot.clear()
        self.scatter = None
        self.selected_point = None
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # If basemap enabled, reload after clearing
        if self.basemap_checkbox.isChecked():
            self.load_basemap()

        if len(selected_rows) == 1:
            # ----- Single selection mode -----
            row = selected_rows[0]
            item0 = self.file_table.item(row, 0)
            traj = item0.data(Qt.UserRole) if item0 else None
            if not traj:
                return

            self.current_trajectory = traj
            self.render_trajectory(traj)   # original method
            self.populate_stats(traj)

        else:
            # ----- Multiple selection mode -----
            legend = self.plot.addLegend()
            colors = [
                (255, 0, 0), (0, 128, 255), (0, 200, 0),
                (255, 165, 0), (128, 0, 255), (255, 20, 147),
                (0, 255, 255), (128, 128, 0), (255, 105, 180)
            ]
            color_index = 0
            all_lats, all_lons = [], []
            all_spots = []

            trajectories_to_plot = []
            for row in selected_rows:
                item0 = self.file_table.item(row, 0)
                traj = item0.data(Qt.UserRole) if item0 else None
                if traj:
                    trajectories_to_plot.append(traj)

            self.current_trajectory = trajectories_to_plot[0] if trajectories_to_plot else None

            for traj in trajectories_to_plot:
                lat_col = traj.columns['lat']
                lon_col = traj.columns['lon']
                lats = traj.df[lat_col].to_numpy()
                lons = traj.df[lon_col].to_numpy()

                color = colors[color_index % len(colors)]
                color_index += 1

                # Line curve
                curve = pg.PlotCurveItem(x=lons, y=lats, pen=pg.mkPen(width=2, color=color))
                self.plot.addItem(curve)
                legend.addItem(curve, os.path.basename(traj.path))

                # Scatter points
                spots = []
                for i, row_data in traj.df.iterrows():
                    data = {'index': int(i)}
                    for k, col in traj.columns.items():
                        data[k] = row_data[col]
                    spots.append({
                        'pos': (row_data[lon_col], row_data[lat_col]),
                        'data': data,
                        'size': 5,
                        'brush': pg.mkBrush(*color, 150)
                    })
                all_spots.extend(spots)

                all_lats.extend(lats)
                all_lons.extend(lons)

            # One scatter plot for all points
            self.scatter = pg.ScatterPlotItem()
            self.scatter.addPoints(all_spots)
            self.scatter.setZValue(2000)
            self.plot.addItem(self.scatter)

            # Fit view
            min_lat, max_lat = np.nanmin(all_lats), np.nanmax(all_lats)
            min_lon, max_lon = np.nanmin(all_lons), np.nanmax(all_lons)
            self.plot.vb.setRange(
                xRange=[min_lon, max_lon],
                yRange=[min_lat, max_lat],
                padding=0.1
            )

            # Clear stats form in multi-select mode
            self.clear_form_layout(self.stats_form)
        self.file_table.verticalScrollBar().setValue(scroll_value)
        if len(self.file_table.selectedIndexes()) > 1 and self.simulation_active:
            self.stop_simulation()
    # def on_select_file(self):
    #     selected_rows = sorted(set(idx.row() for idx in self.file_table.selectedIndexes()))
    #     if not selected_rows:
    #         return

    #     # Clear previous plot (but keep basemap if enabled)
    #     self.plot.clear()
    #     self.plot.showGrid(x=True, y=True, alpha=0.3)

    #     if self.basemap_checkbox.isChecked():
    #         self.load_basemap()

    #     # Prepare legend
    #     legend = self.plot.addLegend()

    #     colors = [
    #         (255, 0, 0), (0, 128, 255), (0, 200, 0),
    #         (255, 165, 0), (128, 0, 255), (255, 20, 147),
    #         (0, 255, 255), (128, 128, 0), (255, 105, 180)
    #     ]
    #     color_index = 0

    #     trajectories_to_plot = []

    #     for row in selected_rows:
    #         item0 = self.file_table.item(row, 0)
    #         traj = item0.data(Qt.UserRole) if item0 else None
    #         if traj:
    #             trajectories_to_plot.append(traj)

    #     # Store only the first trajectory as current for simulation purposes
    #     self.current_trajectory = trajectories_to_plot[0] if trajectories_to_plot else None

    #     for i, traj in enumerate(trajectories_to_plot):
    #         lat = traj.columns['lat']
    #         lon = traj.columns['lon']
    #         lats = traj.df[lat].to_numpy()
    #         lons = traj.df[lon].to_numpy()

    #         color = colors[color_index % len(colors)]
    #         color_index += 1

    #         curve = pg.PlotCurveItem(
    #             x=lons, y=lats,
    #             pen=pg.mkPen(width=2, color=color)
    #         )
    #         self.plot.addItem(curve)
    #         legend.addItem(curve, os.path.basename(traj.path))

    #     # Set view to fit all selected trajectories
    #     all_lats = np.concatenate([t.df[t.columns['lat']].to_numpy() for t in trajectories_to_plot])
    #     all_lons = np.concatenate([t.df[t.columns['lon']].to_numpy() for t in trajectories_to_plot])
    #     min_lat, max_lat = np.nanmin(all_lats), np.nanmax(all_lats)
    #     min_lon, max_lon = np.nanmin(all_lons), np.nanmax(all_lons)
    #     self.plot.vb.setRange(
    #         xRange=[min_lon, max_lon],
    #         yRange=[min_lat, max_lat],
    #         padding=0.1
    #     )

    #     # If exactly one trajectory is selected → populate stats
    #     if len(trajectories_to_plot) == 1:
    #         self.populate_stats(trajectories_to_plot[0])
    #     else:
    #         self.clear_form_layout(self.stats_form)

    # def on_select_file(self):
    #     """Handle file selection from table"""
    #     current_row = self.file_table.currentRow()
    #     if current_row < 0:
    #         return

    #     # First try getting the trajectory directly from the UserRole
    #     item0 = self.file_table.item(current_row, 0)
    #     traj = item0.data(Qt.UserRole) if item0 is not None else None

    #     # Fallback for older rows (no UserRole) – find by file name
    #     if traj is None:
    #         filename = item0.text() if item0 else None
    #         for t in self.trajectories:
    #             if os.path.basename(t.path) == filename:
    #                 traj = t
    #                 break

    #     # Still nothing? Abort.
    #     if traj is None:
    #         return

    #     # Store current trajectory and plot
    #     self.current_trajectory = traj
    #     self.render_trajectory(traj)
    #     self.populate_stats(traj)
        
    #     # Reload basemap if it's enabled
    #     if self.basemap_checkbox.isChecked():
    #         self.load_basemap()

    def render_trajectory(self, traj: Trajectory):
        # Set waiting cursor for potentially long rendering operation
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            # Clear trajectory elements but keep basemap
            if self.curve:
                self.plot.removeItem(self.curve)
            if self.scatter:
                self.plot.removeItem(self.scatter)
                
            lat = traj.columns['lat']
            lon = traj.columns['lon']
            lats = traj.df[lat].to_numpy()
            lons = traj.df[lon].to_numpy()
            
            self.curve = pg.PlotCurveItem(x=lons, y=lats, pen=pg.mkPen(width=2, color='b'))
            self.curve.setZValue(1000)  # Ensure trajectory appears above basemap
            self.plot.addItem(self.curve)
            
            spots = []
            for i, row in traj.df.iterrows():
                data = {'index': int(i)}
                for k, col in traj.columns.items():
                    data[k] = row[col]
                spots.append({'pos': (row[lon], row[lat]), 'data': data, 'size': 6, 'brush': pg.mkBrush(0, 0, 255, 150)})
            
            self.scatter = pg.ScatterPlotItem()
            self.scatter.addPoints(spots)
            self.scatter.setZValue(2000)  # Ensure points appear above everything
            self.plot.addItem(self.scatter)
            self.selected_point = None
            
            # Set view to trajectory extent for proper initial display
            self.set_view_to_trajectory_extent()
            
        except Exception as e:
            QMessageBox.warning(self, "Render Error", f"Failed to render trajectory:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def clear_form_layout(self, form: QFormLayout):
        while form.rowCount() > 0:
            form.removeRow(0)

    def populate_stats(self, traj: Trajectory):
        self.clear_form_layout(self.stats_form)
        try:
            for k, v in traj.stats().items():
                label_key = QLabel(k+":")
                label_value = QLabel(str(v))
                # Increase font size for stats
                font = label_key.font()
                font.setPointSize(10)  # Increase from default (usually 8-9)
                label_key.setFont(font)
                label_value.setFont(font)
                self.stats_form.addRow(label_key, label_value)
        except Exception:
            pass

    def clear_point_details(self):
        self.clear_form_layout(self.point_form)

    def set_point_details(self, data: Dict):
        self.clear_point_details()
        for k, v in data.items():
            label_key = QLabel(str(k))
            label_value = QLabel(str(v))
            # Increase font size for point details
            font = label_key.font()
            font.setPointSize(10)  # Increase from default (usually 8-9)
            label_key.setFont(font)
            label_value.setFont(font)
            self.point_form.addRow(label_key, label_value)

    def on_mouse_moved(self, pos):
        """Keep the last hovered point active until a new one is chosen."""
        try:
            if self.scatter is None or len(self.scatter.points()) == 0:
                return

            mouse_point = self.plot.vb.mapSceneToView(pos)

            # find the closest point
            closest_dist = float('inf')
            closest_point  = None
            for pt in self.scatter.points():
                dx = pt.pos().x() - mouse_point.x()
                dy = pt.pos().y() - mouse_point.y()
                dist = dx * dx + dy * dy
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = pt

            # same point already selected → nothing to do
            if self.selected_point == closest_point:
                return

            # if we are close enough to a new point, switch to it
            if closest_point is not None and closest_dist < 0.0005:
                if self.selected_point is not None:
                    self.selected_point.setBrush(pg.mkBrush(0, 0, 255, 150))
                closest_point.setBrush(pg.mkBrush(255, 0, 0, 200))
                self.selected_point = closest_point
                self.set_point_details(closest_point.data())

            # otherwise keep the last one unchanged (do nothing)
        except Exception:
            pass

    def on_reset_view(self):
        self.plot.enableAutoRange()

    def closeEvent(self, event):
        """Clean up when closing the application"""
        try:
            if self.simulation_active:
                self.stop_simulation()
            if self.tile_loader:
                self.tile_loader.stop()
                if self.tile_loader.isRunning():
                    self.tile_loader.wait(2000)  # Wait max 2 seconds
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    win = AISViewer()
    win.show()
    sys.exit(app.exec_())