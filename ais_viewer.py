#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized PyQt5 AIS Trajectory Viewer
- Streamlined code structure and reduced redundancy
- Optimized data processing and rendering
- Improved memory efficiency
- Maintained all original functionality
"""

import sys
import math
import os
import requests
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from PIL import Image
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFormLayout, QMessageBox, QSplitter, QGroupBox, QCheckBox,
    QHeaderView
)
from PyQt5.QtGui import QCursor, QFont
import pyqtgraph as pg

os.environ['QT_XCB_GL_INTEGRATION'] = 'none'

# Constants and utilities
COLUMN_ALIASES = {
    'time': {'timestamp', 'time', 'datetime', 'basedatetime', 'datetimestamp'},
    'lat': {'lat', 'latitude', 'y'},
    'lon': {'lon', 'long', 'longitude', 'x'},
    'sog': {'sog', 'speed', 'speedoverground'},
    'cog': {'cog', 'course', 'courseoverground', 'heading'},
    'status': {'status', 'navstatus', 'nav_status'},
    'mmsi': {'mmsi', 'imo', 'vesselid', 'vessel_id', 'shipid'},
}

def _norm_col(s: str) -> str:
    """Normalize column name for comparison"""
    return ''.join(ch for ch in s.lower() if ch.isalnum())

class NumericTableWidgetItem(QTableWidgetItem):
    """Table item with numeric sorting capability"""
    def __init__(self, text: str, numeric_value: Union[int, float]):
        super().__init__(text)
        self.numeric_value = numeric_value
    
    def __lt__(self, other):
        return (self.numeric_value < other.numeric_value 
                if isinstance(other, NumericTableWidgetItem) 
                else super().__lt__(other))

def map_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map DataFrame columns to canonical names"""
    canon = {}
    norm_map = {_norm_col(c): c for c in df.columns}
    
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _norm_col(alias)
            if key in norm_map:
                canon[target] = norm_map[key]
                break
    
    missing = [k for k in ['time', 'lat', 'lon'] if k not in canon]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return canon

def parse_time_series(series: pd.Series) -> pd.Series:
    """Parse time series with multiple formats"""
    try:
        result = pd.to_datetime(series, errors='coerce', utc=False, infer_datetime_format=True)
        if result.isna().all():
            result = pd.to_datetime(series.astype(float), unit='s', errors='coerce', utc=False)
        return result
    except Exception:
        return pd.Series([pd.NaT] * len(series))

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km"""
    try:
        R = 6371.0088
        phi1, phi2 = np.radians([lat1, lat2])
        dphi, dlambda = np.radians([lat2 - lat1, lon2 - lon1])
        a = (np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    except Exception:
        return np.zeros_like(lat1)

def create_transparent_heatmap(data_2d: np.ndarray, colormap: str = 'hot') -> np.ndarray:
    """Create RGBA heatmap with transparency"""
    normalized = (data_2d - data_2d.min()) / (data_2d.max() - data_2d.min())
    rgba = np.zeros((*data_2d.shape, 4), dtype=np.uint8)
    
    if colormap == 'hot':
        rgba[..., 0] = np.clip(normalized * 3 * 255, 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip((normalized * 3 - 1) * 255, 0, 255).astype(np.uint8)
        rgba[..., 2] = np.clip((normalized * 3 - 2) * 255, 0, 255).astype(np.uint8)
    else:  # viridis
        rgba[..., 0] = (normalized * 255).astype(np.uint8)
        rgba[..., 1] = ((normalized ** 0.5) * 255).astype(np.uint8)
        rgba[..., 2] = ((1 - normalized) * 255).astype(np.uint8)
    
    rgba[..., 3] = (normalized * 255).astype(np.uint8)  # Alpha channel
    return rgba

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple:
    """Convert lat/lon to tile numbers"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    return (int((lon_deg + 180.0) / 360.0 * n),
            int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))

def num2deg(xtile: int, ytile: int, zoom: int) -> tuple:
    """Convert tile numbers to lat/lon"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return (math.degrees(lat_rad), lon_deg)

class TileLoader(QThread):
    """Optimized background tile loader"""
    tile_loaded = pyqtSignal(int, int, int, object)
    
    def __init__(self):
        super().__init__()
        self.tiles_to_load = []
        self.running = True
        
    def add_tile(self, x: int, y: int, zoom: int):
        self.tiles_to_load.append((x, y, zoom))
        
    def run(self):
        while self.running and self.tiles_to_load:
            try:
                x, y, zoom = self.tiles_to_load.pop(0)
                if not self.running:
                    break
                    
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                response = requests.get(url, headers={'User-Agent': 'AIS Trajectory Viewer'}, timeout=10)
                
                if response.status_code == 200 and self.running:
                    img = np.array(Image.open(io.BytesIO(response.content)))
                    self.tile_loaded.emit(x, y, zoom, img)
                    
            except Exception as e:
                print(f"Failed to load tile {x},{y},{zoom}: {e}")
                
    def stop(self):
        self.running = False
        self.tiles_to_load.clear()

@dataclass
class Trajectory:
    """Optimized trajectory data container"""
    path: str
    df: pd.DataFrame
    columns: Dict[str, str]
    _stats_cache: Optional[Dict] = None
    
    @property
    def mmsi(self) -> Optional[str]:
        col = self.columns.get('mmsi')
        if not col or col not in self.df.columns:
            return None
        vals = self.df[col].dropna().unique()
        return (str(vals[0]) if len(vals) == 1 
                else f"{str(vals[0])} (+{len(vals)-1} more)" if len(vals) > 1 
                else None)

    def stats(self) -> Dict[str, str]:
        """Cached statistics calculation"""
        if self._stats_cache is not None:
            return self._stats_cache
            
        try:
            tcol, lat, lon = self.columns['time'], self.columns['lat'], self.columns['lon']
            n = len(self.df)
            tmin, tmax = self.df[tcol].min(), self.df[tcol].max()
            dur = (tmax - tmin) if pd.notna(tmax) and pd.notna(tmin) else pd.NaT
            
            # Vectorized distance calculation
            coords = self.df[[lat, lon]].to_numpy()
            if n >= 2:
                dist_km = np.nansum(haversine_km(coords[:-1, 0], coords[:-1, 1], 
                                               coords[1:, 0], coords[1:, 1]))
            else:
                dist_km = 0.0
                
            self._stats_cache = {
                'Vessel ID': self.mmsi or '—',
                'Points': f"{n}",
                'Time start': f"{tmin}",
                'Time end': f"{tmax}",
                'Duration': str(dur) if pd.notna(dur) else '—',
                'Total distance': f"{dist_km:.3f} km",
            }
            return self._stats_cache
        except Exception:
            return {'Vessel ID': '—', 'Points': '0', 'Time start': '—', 
                   'Time end': '—', 'Duration': '—', 'Total distance': '0 km'}

    def table_data(self) -> Dict:
        """Get data for table display"""
        try:
            stats = self.stats()
            n = int(stats['Points'])
            dist_km = float(stats['Total distance'].split()[0])
            
            # Parse duration
            dur_str = stats['Duration']
            if dur_str != '—':
                try:
                    dur_obj = pd.to_timedelta(dur_str)
                    total_seconds = int(dur_obj.total_seconds())
                    hours, minutes = divmod(total_seconds, 3600)
                    minutes //= 60
                    duration_str = f"{hours}h {minutes}m"
                    duration_numeric = total_seconds
                except:
                    duration_str = "—"
                    duration_numeric = 0
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
                'points': 0, 'distance': 0.0, 'duration': "—", 'duration_numeric': 0
            }

class AISViewer(QMainWindow):
    """Optimized AIS Trajectory Viewer"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIS Trajectory Viewer")
        self.resize(1200, 800)
        
        # Initialize state variables
        self.trajectories: List[Trajectory] = []
        self.current_trajectory = None
        self.curve = self.scatter = self.selected_point = None
        self.basemap_items = []
        self.tile_loader = None
        
        # Simulation variables
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.simulation_index = 0
        self.simulation_active = False
        self.vessel_icon = self.dynamic_curve = None
        self.simulation_speed_multiplier = 70
        
        # Measurement variables
        self.measure_start = None
        self.measure_line = self.measure_text = None
        
        # Heatmap variables
        self.heatmap_item = self.heatmap_lut = None
        self.heatmap_processing = False
        
        self._setup_ui()
        self._connect_signals()
        
        pg.setConfigOptions(antialias=True)

    def _setup_ui(self):
        """Setup user interface"""
        splitter = QSplitter()
        self.setCentralWidget(splitter)

        # Left panel
        left = QWidget()
        left.setMaximumWidth(400)
        left_layout = QVBoxLayout(left)

        # Button rows
        btn_layout = self._create_button_layout()
        left_layout.addLayout(btn_layout)
        
        # Checkboxes
        checkbox_layout = self._create_checkbox_layout()
        left_layout.addLayout(checkbox_layout)

        # File table
        self.file_table = self._create_file_table()
        left_layout.addWidget(self.file_table)

        # Info panels
        self.stats_group, self.stats_form = self._create_info_panel("Trajectory Info")
        self.point_group, self.point_form = self._create_info_panel("Point Details")
        left_layout.addWidget(self.stats_group)
        left_layout.addWidget(self.point_group)

        splitter.addWidget(left)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('white')
        self.plot = self.plot_widget.getPlotItem()
        self.plot.getViewBox().setBackgroundColor('white')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        splitter.addWidget(self.plot_widget)

    def _create_button_layout(self) -> QVBoxLayout:
        """Create button layout"""
        layout = QVBoxLayout()
        
        # Row 1
        row1 = QHBoxLayout()
        self.btn_import = QPushButton("Open Files…")
        self.btn_import_folder = QPushButton("Open Folder…")
        self.btn_delete_selected = QPushButton("Delete Selected")
        row1.addWidget(self.btn_import)
        row1.addWidget(self.btn_import_folder)
        row1.addWidget(self.btn_delete_selected)
        
        # Row 2
        row2 = QHBoxLayout()
        self.btn_simulate = QPushButton("▶ Simulate Sailing")
        self.btn_simulate.setCheckable(True)
        self.btn_measure = QPushButton("Measure Distance")
        self.btn_measure.setCheckable(True)
        self.btn_reset = QPushButton("Reset View")
        row2.addWidget(self.btn_simulate)
        row2.addWidget(self.btn_measure)
        row2.addWidget(self.btn_reset)
        
        layout.addLayout(row1)
        layout.addLayout(row2)
        return layout

    def _create_checkbox_layout(self) -> QHBoxLayout:
        """Create checkbox layout"""
        layout = QHBoxLayout()
        self.simulate_speed_checkbox = QCheckBox("Simulate with SOG")
        self.heatmap_checkbox = QCheckBox("Show Heatmap")
        self.basemap_checkbox = QCheckBox("Show Basemap")
        
        layout.addWidget(self.simulate_speed_checkbox)
        layout.addWidget(self.heatmap_checkbox)
        layout.addWidget(self.basemap_checkbox)
        layout.addStretch()
        return layout

    def _create_file_table(self) -> QTableWidget:
        """Create optimized file table"""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['File Name', 'Points', 'Distance (km)', 'Duration'])
        table.setSortingEnabled(True)
        table.setShowGrid(False)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.ExtendedSelection)
        
        # Set column properties
        header = table.horizontalHeader()
        for i in range(4):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
        
        # Set initial widths
        widths = [200, 80, 100, 80]
        for i, width in enumerate(widths):
            table.setColumnWidth(i, width)
            
        return table

    def _create_info_panel(self, title: str) -> tuple:
        """Create info panel with group box"""
        group = QGroupBox(title)
        form = QFormLayout()
        group.setLayout(form)
        return group, form

    def _connect_signals(self):
        """Connect all signals"""
        # Button signals
        self.btn_import.clicked.connect(self.on_import)
        self.btn_import_folder.clicked.connect(self.on_import_folder)
        self.btn_delete_selected.clicked.connect(self.delete_selected_rows)
        self.btn_simulate.clicked.connect(self.on_simulate_toggle)
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        self.btn_reset.clicked.connect(self.on_reset_view)
        
        # Checkbox signals
        self.heatmap_checkbox.toggled.connect(self.on_select_file)
        self.basemap_checkbox.toggled.connect(self.on_basemap_toggled)
        
        # Table and plot signals
        self.file_table.itemSelectionChanged.connect(self.on_select_file)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)

    def _load_single_file(self, path: str):
        """Optimized single file loading"""
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise ValueError("File does not exist or is empty")
            
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV has no data rows")
            
        cols = map_columns(df)
        
        # Efficient column processing
        df[cols['time']] = parse_time_series(df[cols['time']])
        df[cols['lat']] = pd.to_numeric(df[cols['lat']], errors='coerce')
        df[cols['lon']] = pd.to_numeric(df[cols['lon']], errors='coerce')
        
        # Clean and sort in one operation
        df = (df.dropna(subset=[cols['time'], cols['lat'], cols['lon']])
               .sort_values(cols['time'])
               .reset_index(drop=True))
        
        if df.empty:
            raise ValueError("All required columns have NaN values after cleaning")
            
        traj = Trajectory(path=path, df=df, columns=cols)
        self.trajectories.append(traj)
        self.add_trajectory_to_table(traj)

    def add_trajectory_to_table(self, traj: Trajectory):
        """Add trajectory to table efficiently"""
        data = traj.table_data()
        row = self.file_table.rowCount()
        self.file_table.insertRow(row)

        # Set table items
        items = [
            (QTableWidgetItem(data['filename']), traj),
            (NumericTableWidgetItem(str(data['points']), data['points']), None),
            (NumericTableWidgetItem(f"{data['distance']:.3f}", data['distance']), None),
            (NumericTableWidgetItem(data['duration'], data['duration_numeric']), None)
        ]
        
        for col, (item, user_data) in enumerate(items):
            if user_data:
                item.setData(Qt.UserRole, user_data)
            self.file_table.setItem(row, col, item)

    def delete_selected_rows(self):
        """Delete selected rows efficiently"""
        selected_rows = sorted(set(idx.row() for idx in self.file_table.selectedIndexes()), reverse=True)
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select rows to delete.")
            return

        if QMessageBox.question(self, "Confirm Deletion", 
                              f"Delete {len(selected_rows)} file(s)?") != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            for row in selected_rows:
                item = self.file_table.item(row, 0)
                if item:
                    traj = item.data(Qt.UserRole)
                    if traj in self.trajectories:
                        self.trajectories.remove(traj)
                self.file_table.removeRow(row)

            # Clear view if current trajectory was deleted
            if self.current_trajectory not in self.trajectories:
                self._clear_plot_and_forms()
        finally:
            QApplication.restoreOverrideCursor()

    def _clear_plot_and_forms(self):
        """Clear plot and form data"""
        self.current_trajectory = None
        self.plot.clear()
        self._clear_form_layout(self.stats_form)
        self._clear_form_layout(self.point_form)

    def on_import_folder(self):
        """Import folder with error handling"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with CSV files")
        if not folder:
            return

        try:
            csv_files = [os.path.join(folder, f) for f in os.listdir(folder)
                        if f.lower().endswith('.csv') and os.path.isfile(os.path.join(folder, f))]
        except Exception as e:
            QMessageBox.warning(self, "Folder Error", f"Error reading folder: {e}")
            return

        if not csv_files:
            QMessageBox.information(self, "No CSV Files", "No CSV files found in the selected folder.")
            return

        self._batch_load_files(csv_files)

    def on_import(self):
        """Import selected files"""
        paths, _ = QFileDialog.getOpenFileNames(self, "Select AIS CSV files", '', "CSV Files (*.csv)")
        if paths:
            self._batch_load_files(paths)

    def _batch_load_files(self, paths: List[str]):
        """Batch load files with progress feedback"""
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        successful, failed = 0, 0
        
        try:
            for path in paths:
                try:
                    self._load_single_file(path)
                    successful += 1
                except Exception as e:
                    failed += 1
                    print(f"Failed to load {path}: {e}")
                    
        finally:
            QApplication.restoreOverrideCursor()
            
        # Show summary
        msg = f"Successfully loaded {successful} files."
        if failed > 0:
            msg += f" Failed to load {failed} files."
        QMessageBox.information(self, "Import Complete", msg)

    def on_select_file(self):
        """Optimized file selection with heatmap support"""
        if getattr(self, 'heatmap_processing', False):
            return
            
        self.heatmap_processing = True
        selected_rows = sorted(set(idx.row() for idx in self.file_table.selectedIndexes()))
        
        if not selected_rows:
            self.heatmap_processing = False
            return

        try:
            # Always clear everything first - trajectories, heatmaps, legends
            self._clear_all_plot_items()
            
            # Reload basemap if needed
            if self.basemap_checkbox.isChecked():
                self.load_basemap()

            # Get trajectories and coordinates
            trajectories, all_coords = self._get_selected_trajectories_and_coords(selected_rows)
            if not trajectories:
                return

            self.current_trajectory = trajectories[0]

            if self.heatmap_checkbox.isChecked():
                # HEATMAP MODE: Only show heatmap, no trajectories at all
                self._render_heatmap(all_coords)
            else:
                # TRAJECTORY MODE: Show trajectories and points
                self._render_trajectories(trajectories, selected_rows)

            # Fit view
            if len(all_coords) > 0:
                self._fit_view_to_coords(all_coords)
                
        finally:
            self.heatmap_processing = False

    def _clear_all_plot_items(self):
        """Clear all plot items except basemap"""
        # Clear trajectory items
        for item in [self.curve, self.scatter, self.selected_point]:
            if item and item in self.plot.items:
                self.plot.removeItem(item)
        self.curve = self.scatter = self.selected_point = None
        
        # Clear legend if it exists
        if hasattr(self.plot, 'legend') and self.plot.legend is not None:
            self.plot.legend.scene().removeItem(self.plot.legend)
            self.plot.legend = None
            
        # Clear heatmap items
        if self.heatmap_item:
            self.plot.removeItem(self.heatmap_item)
            self.heatmap_item = None
        if self.heatmap_lut:
            try:
                if hasattr(self.heatmap_lut, 'setRect'):
                    self.plot.removeItem(self.heatmap_lut)
                else:
                    self.plot_widget.removeItem(self.heatmap_lut)
            except (ValueError, RuntimeError):
                pass
            self.heatmap_lut = None
            
        # Clear any remaining plot items that aren't basemap (z-value >= 0)
        items_to_remove = []
        for item in self.plot.items:
            if hasattr(item, 'zValue') and item.zValue() >= 0:
                # Don't remove basemap items (they have negative z-values)
                if item not in self.basemap_items:
                    items_to_remove.append(item)
        
        for item in items_to_remove:
            try:
                self.plot.removeItem(item)
            except:
                pass

    def _get_selected_trajectories_and_coords(self, selected_rows: List[int]) -> tuple:
        """Get trajectories and coordinates for selected rows"""
        trajectories = []
        all_lats, all_lons = [], []
        
        for row in selected_rows:
            item = self.file_table.item(row, 0)
            traj = item.data(Qt.UserRole) if item else None
            if traj:
                trajectories.append(traj)
                lat_col, lon_col = traj.columns['lat'], traj.columns['lon']
                all_lats.extend(traj.df[lat_col].to_numpy())
                all_lons.extend(traj.df[lon_col].to_numpy())
                
        return trajectories, list(zip(all_lats, all_lons))

    def _render_heatmap(self, coords: List[tuple]):
        """Render transparent heatmap ONLY - no trajectories"""
        if not coords:
            return
            
        lats, lons = zip(*coords)
        
        # Create 2D histogram
        heatmap, yedges, xedges = np.histogram2d(lats, lons, bins=300)
        heatmap = gaussian_filter(heatmap, sigma=1.0)
        
        # Create transparent RGBA heatmap
        rgba_heatmap = create_transparent_heatmap(heatmap, 'hot')
        rgba_heatmap = np.rot90(np.flipud(rgba_heatmap), k=3)
        
        self.heatmap_item = pg.ImageItem(rgba_heatmap)
        self.heatmap_item.setRect(xedges[0], yedges[0], xedges[-1] - xedges[0], yedges[-1] - yedges[0])
        self.heatmap_item.setZValue(-100)  # Behind everything except basemap
        self.plot.addItem(self.heatmap_item)
        
        # Clear forms since we're only showing heatmap
        self._clear_form_layout(self.stats_form)
        self._clear_form_layout(self.point_form)

    def _render_trajectories(self, trajectories: List[Trajectory], selected_rows: List[int]):
        """Render trajectory lines and points"""
        if len(selected_rows) == 1:
            self.render_trajectory(trajectories[0])
            self.populate_stats(trajectories[0])
        else:
            self._render_multiple_trajectories(trajectories)

    def _render_multiple_trajectories(self, trajectories: List[Trajectory]):
        """Render multiple trajectories with different colors"""
        # Clear any existing legend first
        if hasattr(self.plot, 'legend') and self.plot.legend is not None:
            self.plot.legend.scene().removeItem(self.plot.legend)
            self.plot.legend = None
            
        legend = self.plot.addLegend()
        colors = [(255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 165, 0), 
                 (128, 0, 255), (255, 20, 147), (0, 255, 255), (128, 128, 0)]
        all_spots = []

        for i, traj in enumerate(trajectories):
            color = colors[i % len(colors)]
            lat_col, lon_col = traj.columns['lat'], traj.columns['lon']
            coords = traj.df[[lon_col, lat_col]].to_numpy()
            
            # Add trajectory line
            curve = pg.PlotCurveItem(x=coords[:, 0], y=coords[:, 1], pen=pg.mkPen(width=2, color=color))
            self.plot.addItem(curve)
            legend.addItem(curve, os.path.basename(traj.path))
            
            # Create scatter points
            spots = [{'pos': (row[lon_col], row[lat_col]), 
                     'data': {**{k: row[col] for k, col in traj.columns.items()}, 'index': idx},
                     'size': 5, 'brush': pg.mkBrush(*color, 150)}
                    for idx, (_, row) in enumerate(traj.df.iterrows())]
            all_spots.extend(spots)

        # Add all scatter points at once
        if all_spots:
            self.scatter = pg.ScatterPlotItem()
            self.scatter.addPoints(all_spots)
            self.scatter.setZValue(2000)
            self.plot.addItem(self.scatter)
            
        # Clear forms for multiple trajectories
        self._clear_form_layout(self.stats_form)

    def _fit_view_to_coords(self, coords: List[tuple]):
        """Fit view to coordinate bounds"""
        if not coords:
            return
            
        lats, lons = zip(*coords)
        min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
        min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)
        
        self.plot.vb.setRange(xRange=[min_lon, max_lon], yRange=[min_lat, max_lat], padding=0.1)

    def render_trajectory(self, traj: Trajectory):
        """Render single trajectory efficiently"""
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            lat_col, lon_col = traj.columns['lat'], traj.columns['lon']
            coords = traj.df[[lon_col, lat_col]].to_numpy()
            
            # Create trajectory line
            self.curve = pg.PlotCurveItem(x=coords[:, 0], y=coords[:, 1], pen=pg.mkPen(width=2, color='b'))
            self.curve.setZValue(1000)
            self.plot.addItem(self.curve)
            
            # Create scatter points
            spots = [{'pos': (row[lon_col], row[lat_col]),
                     'data': {**{k: row[col] for k, col in traj.columns.items()}, 'index': idx},
                     'size': 6, 'brush': pg.mkBrush(0, 0, 255, 150)}
                    for idx, (_, row) in enumerate(traj.df.iterrows())]
            
            self.scatter = pg.ScatterPlotItem()
            self.scatter.addPoints(spots)
            self.scatter.setZValue(2000)
            self.plot.addItem(self.scatter)
            
        except Exception as e:
            QMessageBox.warning(self, "Render Error", f"Failed to render trajectory: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def populate_stats(self, traj: Trajectory):
        """Populate statistics form"""
        self._clear_form_layout(self.stats_form)
        try:
            for k, v in traj.stats().items():
                self._add_form_row(self.stats_form, k, str(v))
        except Exception:
            pass

    def _add_form_row(self, form: QFormLayout, key: str, value: str):
        """Add row to form layout with consistent styling"""
        label_key, label_value = QLabel(f"{key}:"), QLabel(value)
        font = label_key.font()
        font.setPointSize(10)
        label_key.setFont(font)
        label_value.setFont(font)
        form.addRow(label_key, label_value)

    def _clear_form_layout(self, form: QFormLayout):
        """Clear form layout"""
        while form.rowCount() > 0:
            form.removeRow(0)

    def set_point_details(self, data: Dict):
        """Set point details in form"""
        self._clear_form_layout(self.point_form)
        for k, v in data.items():
            self._add_form_row(self.point_form, str(k), str(v))

    # Simulation methods
    def on_simulate_toggle(self, checked: bool):
        """Handle simulation toggle"""
        selected_rows = sorted(set(idx.row() for idx in self.file_table.selectedIndexes()))
        if len(selected_rows) != 1:
            QMessageBox.warning(self, "Simulation Disabled", "Please select exactly one trajectory to simulate.")
            self.btn_simulate.setChecked(False)
            return

        if checked:
            self.start_simulation()
        else:
            self.stop_simulation()

    def start_simulation(self):
        """Start vessel sailing simulation"""
        if not self.current_trajectory:
            self.btn_simulate.setChecked(False)
            QMessageBox.warning(self, "No Trajectory", "Please select a trajectory first.")
            return

        self._setup_simulation_view()
        self.simulation_active = True
        self.simulation_index = 0
        self.btn_simulate.setText("⏸ Stop Simulation")
        
        # Create simulation elements
        self.dynamic_curve = pg.PlotCurveItem(pen=pg.mkPen(width=2, color='b'))
        self.dynamic_curve.setZValue(1000)
        self.plot.addItem(self.dynamic_curve)
        
        self.vessel_icon = pg.ScatterPlotItem()
        self.vessel_icon.setZValue(3000)
        self.plot.addItem(self.vessel_icon)
        
        self.simulation_timer.start(50)

    def stop_simulation(self):
        """Stop vessel sailing simulation"""
        self.simulation_active = False
        self.simulation_timer.stop()
        self.btn_simulate.setText("▶ Simulate Sailing")
        self.btn_simulate.setChecked(False)
        
        # Clean up simulation elements
        for item in [self.vessel_icon, self.dynamic_curve]:
            if item:
                self.plot.removeItem(item)
        self.vessel_icon = self.dynamic_curve = None
        
        # Restore trajectory display
        self.plot.vb.enableAutoRange()
        if self.current_trajectory:
            # Clear everything first, then render fresh
            self._clear_all_plot_items()
            if self.basemap_checkbox.isChecked():
                self.load_basemap()
            self.render_trajectory(self.current_trajectory)

    def _setup_simulation_view(self):
        """Setup view for simulation"""
        if not self.current_trajectory:
            return
            
        # Clear existing trajectory display
        self._clear_all_plot_items()
        
        # Set view to trajectory bounds
        lat_col, lon_col = self.current_trajectory.columns['lat'], self.current_trajectory.columns['lon']
        coords = self.current_trajectory.df[[lat_col, lon_col]].to_numpy()
        
        bounds = np.array([np.nanmin(coords, axis=0), np.nanmax(coords, axis=0)])
        ranges = bounds[1] - bounds[0]
        ranges = np.maximum(ranges, 0.001)  # Minimum range
        padding = ranges * 0.1
        
        self.plot.vb.disableAutoRange()
        self.plot.vb.setRange(
            xRange=[bounds[0, 1] - padding[1], bounds[1, 1] + padding[1]],
            yRange=[bounds[0, 0] - padding[0], bounds[1, 0] + padding[0]],
            padding=0
        )

    def update_simulation(self):
        """Update simulation animation"""
        if not self.simulation_active or not self.current_trajectory:
            return
            
        df = self.current_trajectory.df
        if self.simulation_index >= len(df):
            self.stop_simulation()
            return
            
        # Get current position
        current_row = df.iloc[self.simulation_index]
        lat_col, lon_col = self.current_trajectory.columns['lat'], self.current_trajectory.columns['lon']
        current_lat, current_lon = current_row[lat_col], current_row[lon_col]
        
        # Update vessel position
        if self.vessel_icon:
            cog_col = self.current_trajectory.columns.get('cog')
            current_cog = current_row[cog_col] if cog_col else None
            vessel_spot = self._create_vessel_icon(current_lat, current_lon, current_cog)
            self.vessel_icon.clear()
            self.vessel_icon.addPoints([vessel_spot])
        
        # Update trajectory path
        if self.dynamic_curve:
            path_data = df[[lon_col, lat_col]].iloc[:self.simulation_index + 1].to_numpy()
            self.dynamic_curve.setData(path_data[:, 0], path_data[:, 1])
        
        # Calculate next step
        self.simulation_index += self._calculate_simulation_step(current_row, df)

    def _create_vessel_icon(self, lat: float, lon: float, cog: Optional[float] = None) -> Dict:
        """Create vessel icon specification"""
        color = (255, 0, 0, 220) if cog is not None and pd.notna(cog) else (255, 100, 0, 220)
        return {
            'pos': (lon, lat),
            'size': 15,
            'symbol': 'o',
            'brush': pg.mkBrush(*color),
            'pen': pg.mkPen(width=2, color='darkred'),
            'data': {'vessel': True, 'cog': cog}
        }

    def _calculate_simulation_step(self, current_row, df: pd.DataFrame) -> int:
        """Calculate simulation step size"""
        if self.simulation_index >= len(df) - 1:
            return 1
            
        time_col = self.current_trajectory.columns['time']
        current_time = current_row[time_col]
        next_time = df.iloc[self.simulation_index + 1][time_col]
        
        if self.simulate_speed_checkbox.isChecked():
            sog_col = self.current_trajectory.columns.get('sog')
            if sog_col and sog_col in df.columns:
                current_sog = current_row[sog_col]
                if pd.notna(current_sog):
                    speed_mps = current_sog * 0.514444  # knots to m/s
                    time_diff = (next_time - current_time).total_seconds() if pd.notna(current_time) and pd.notna(next_time) else 1
                    return max(1, int(time_diff * (speed_mps / self.simulation_speed_multiplier)))
        
        # Default time-based stepping
        if pd.notna(current_time) and pd.notna(next_time):
            time_diff = (next_time - current_time).total_seconds()
            return max(1, int(time_diff / self.simulation_speed_multiplier))
        return 1

    # Basemap methods
    def on_basemap_toggled(self, checked: bool):
        """Handle basemap toggle"""
        if checked:
            self.load_basemap()
        else:
            self.clear_basemap()

    def clear_basemap(self):
        """Clear basemap tiles"""
        try:
            for item in self.basemap_items:
                if item in self.plot.items:
                    self.plot.removeItem(item)
            self.basemap_items.clear()
            
            if self.tile_loader:
                self.tile_loader.stop()
                if self.tile_loader.isRunning():
                    self.tile_loader.wait(1000)
                self.tile_loader = None
        except Exception as e:
            print(f"Error clearing basemap: {e}")
            self.basemap_items.clear()

    def load_basemap(self):
        """Load basemap tiles for current view"""
        if not self.current_trajectory:
            return
            
        self.clear_basemap()
        
        # Get trajectory bounds with padding
        lat_col, lon_col = self.current_trajectory.columns['lat'], self.current_trajectory.columns['lon']
        coords = self.current_trajectory.df[[lat_col, lon_col]].to_numpy()
        bounds = np.array([np.nanmin(coords, axis=0), np.nanmax(coords, axis=0)])
        
        # Add padding
        ranges = bounds[1] - bounds[0]
        padding = ranges * 0.2
        bounds[0] -= padding
        bounds[1] += padding
        
        # Determine zoom level
        max_range = np.max(ranges)
        zoom_levels = [20, 10, 5, 2, 1, 0.5, 0.25]
        zoom_values = [5, 6, 7, 8, 9, 10, 11, 12]
        zoom = next((z for r, z in zip(zoom_levels, zoom_values) if max_range > r), 12)
        
        # Get tile bounds
        min_x, max_y = deg2num(bounds[0, 0], bounds[0, 1], zoom)
        max_x, min_y = deg2num(bounds[1, 0], bounds[1, 1], zoom)
        
        # Limit tile count
        max_tiles = 6
        for axis in [(min_x, max_x), (min_y, max_y)]:
            if axis[1] - axis[0] > max_tiles:
                center = (axis[0] + axis[1]) // 2
                axis = (max(0, center - max_tiles // 2), center + max_tiles // 2)
        
        # Start tile loader
        self.tile_loader = TileLoader()
        self.tile_loader.tile_loaded.connect(self.on_tile_loaded)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                self.tile_loader.add_tile(x, y, zoom)
                
        self.tile_loader.start()

    def on_tile_loaded(self, x: int, y: int, zoom: int, img_array: np.ndarray):
        """Handle loaded tile"""
        if img_array is None:
            return
            
        # Get tile bounds and create image item
        lat_north, lon_west = num2deg(x, y, zoom)
        lat_south, lon_east = num2deg(x + 1, y + 1, zoom)
        
        img_item = pg.ImageItem(np.flipud(img_array))
        img_item.setRect(lon_west, lat_south, lon_east - lon_west, lat_north - lat_south)
        img_item.setZValue(-1000)
        
        self.plot.addItem(img_item)
        self.basemap_items.append(img_item)

    # Measurement methods
    def _toggle_measure_mode(self, checked: bool):
        """Toggle measuring mode"""
        if not checked:
            self._clear_measure_overlay()
            self.measure_start = None

    def _clear_measure_overlay(self):
        """Clear measurement overlay"""
        for item in [self.measure_line, self.measure_text]:
            if item:
                self.plot.removeItem(item)
        self.measure_line = self.measure_text = None

    def _on_mouse_click(self, ev):
        """Handle mouse clicks for measurement"""
        if not self.btn_measure.isChecked() or ev.button() != Qt.LeftButton:
            return

        pos = self.plot.vb.mapSceneToView(ev.scenePos())

        if self.measure_start is None:
            self.measure_start = pos
            self._clear_measure_overlay()
            
            # Create measurement line
            current_range = self.plot.vb.viewRange()
            eps = 1e-4
            self.measure_line = pg.PlotDataItem(
                [pos.x(), pos.x() + eps], [pos.y(), pos.y() + eps],
                pen=pg.mkPen('m', width=2, style=Qt.DashLine))
            
            self.plot.vb.disableAutoRange()
            self.plot.addItem(self.measure_line)
            self.plot.vb.setRange(xRange=current_range[0], yRange=current_range[1], padding=0)

            self.measure_text = pg.TextItem(anchor=(0, 1), color='m', fill=pg.mkBrush(255, 255, 255, 200))
            self.plot.addItem(self.measure_text)
            self.plot_widget.scene().sigMouseMoved.connect(self._update_measure_drag)
        else:
            self.plot_widget.scene().sigMouseMoved.disconnect(self._update_measure_drag)
            self._update_measure_line(pos)
            self.measure_start = None

    def _update_measure_drag(self, scene_pos):
        """Update measurement during drag"""
        if self.measure_start:
            cur = self.plot.vb.mapSceneToView(scene_pos)
            self._update_measure_line(cur)

    def _update_measure_line(self, end_pt):
        """Update measurement line and distance"""
        if not self.measure_start:
            return

        x0, y0 = self.measure_start.x(), self.measure_start.y()
        x1, y1 = end_pt.x(), end_pt.y()

        self.measure_line.setData([x0, x1], [y0, y1])
        
        d_km = haversine_km(y0, x0, y1, x1)
        self.measure_text.setText(f"{d_km:.3f} km")
        self.measure_text.setPos(x1, y1)

    # Mouse and interaction methods
    def on_mouse_moved(self, pos):
        """Handle mouse movement for point highlighting"""
        try:
            if not self.scatter or len(self.scatter.points()) == 0:
                return

            mouse_point = self.plot.vb.mapSceneToView(pos)
            closest_dist = float('inf')
            closest_point = None

            # Find closest point efficiently
            for pt in self.scatter.points():
                dx, dy = pt.pos().x() - mouse_point.x(), pt.pos().y() - mouse_point.y()
                dist = dx * dx + dy * dy
                if dist < closest_dist:
                    closest_dist, closest_point = dist, pt

            # Update selection if close enough and different
            if closest_point and closest_dist < 0.0005 and self.selected_point != closest_point:
                if self.selected_point:
                    self.selected_point.setBrush(pg.mkBrush(0, 0, 255, 150))
                closest_point.setBrush(pg.mkBrush(255, 0, 0, 200))
                self.selected_point = closest_point
                self.set_point_details(closest_point.data())
        except Exception:
            pass

    def on_reset_view(self):
        """Reset plot view"""
        self.plot.enableAutoRange()

    def closeEvent(self, event):
        """Clean up on close"""
        try:
            if self.simulation_active:
                self.stop_simulation()
            if self.tile_loader:
                self.tile_loader.stop()
                if self.tile_loader.isRunning():
                    self.tile_loader.wait(2000)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    win = AISViewer()
    win.show()
    sys.exit(app.exec())