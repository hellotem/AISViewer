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
"""

import sys
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFormLayout, QMessageBox, QSplitter, QGroupBox, QCheckBox,
    QHeaderView
)
from PyQt5.QtGui import QCursor

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
                'Vessel ID': '—',
                'Points': '0',
                'Time start': '—',
                'Time end': '—',
                'Duration': '—',
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

        # except Exception:
        #     return {
        #         'filename': os.path.basename(self.path),
        #         'points': 0,
        #         'distance': 0.0,
        #         'duration': "—"
        #     }

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
        self.btn_measure = QPushButton("Measure Distance")
        btn_row.addWidget(self.btn_measure)

        btn_row2 = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom +")
        btn_row2.addWidget(self.btn_zoom_in)
        self.btn_zoom_out = QPushButton("Zoom −")
        btn_row2.addWidget(self.btn_zoom_out)
        self.btn_reset = QPushButton("Reset View")
        btn_row2.addWidget(self.btn_reset)
                
        self.btn_import_folder.clicked.connect(self.on_import_folder)

        self.btn_measure.setCheckable(True)
        self.measure_start = None          # QPointF
        self.measure_line   = None         # pg.PlotDataItem
        self.measure_text   = None         # pg.TextItem

        left_layout.addLayout(btn_row)
        left_layout.addLayout(btn_row2)

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
        self.file_table.setSelectionMode(QTableWidget.SingleSelection)
        
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
        splitter.addWidget(self.plot_widget)

        self.trajectories: List[Trajectory] = []

        self.btn_import.clicked.connect(self.on_import)
        self.file_table.itemSelectionChanged.connect(self.on_select_file)
        self.btn_reset.clicked.connect(self.on_reset_view)
        self.btn_zoom_in.clicked.connect(lambda: self.plot.getViewBox().scaleBy((0.8, 0.8)))
        self.btn_zoom_out.clicked.connect(lambda: self.plot.getViewBox().scaleBy((1.25, 1.25)))

        pg.setConfigOptions(antialias=True)

        # Enable hover events
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)

    def add_trajectory_to_table(self, traj: Trajectory):
        data = traj.table_data()
        row = self.file_table.rowCount()
        self.file_table.insertRow(row)

        # Column 0: filename (store the trajectory for easy retrieval)
        name_item = QTableWidgetItem(data['filename'])
        name_item.setData(Qt.UserRole, traj)
        self.file_table.setItem(row, 0, name_item)

        # # Col 0: filename (store the trajectory handle here)
        # name_item = QTableWidgetItem(data['filename'])
        # name_item.setData(Qt.UserRole, traj)
        # self.file_table.setItem(row, 0, name_item)

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

    def on_import_folder(self):
        """Import every CSV file found in the selected folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with CSV files")
        if not folder:
            return

        csv_files = [os.path.join(folder, f) for f in os.listdir(folder)
                     if f.lower().endswith('.csv')]
        if not csv_files:
            QMessageBox.information(self, "No CSV", "No CSV files found in the selected folder.")
            return

        # Set waiting cursor for long operation
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            for path in csv_files:
                try:
                    if os.path.getsize(path) == 0:
                        raise ValueError("File is empty")
                    df = pd.read_csv(path)
                    if df.empty:
                        raise ValueError("CSV has no rows")
                    cols = map_columns(df)
                    df[cols['time']] = parse_time_series(df[cols['time']])
                    df[cols['lat']] = pd.to_numeric(df[cols['lat']], errors='coerce')
                    df[cols['lon']] = pd.to_numeric(df[cols['lon']], errors='coerce')
                    df = df.dropna(subset=[cols['time'], cols['lat'], cols['lon']]).reset_index(drop=True)
                    if df.empty:
                        raise ValueError("All required columns have NaN values")
                    df = df.sort_values(cols['time']).reset_index(drop=True)
                    traj = Trajectory(path=path, df=df, columns=cols)
                    self.trajectories.append(traj)
                    self.add_trajectory_to_table(traj)
                except Exception as e:
                    QMessageBox.warning(self, "Load Error", f"Failed to load {path}:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

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
        paths, _ = QFileDialog.getOpenFileNames(self, "Select AIS CSV files", '', "CSV Files (*.csv);;All Files (*)")
        if not paths:
            return
        
        # Set waiting cursor for long operation
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            for p in paths:
                try:
                    if os.path.getsize(p) == 0:
                        raise ValueError("File is empty")
                    df = pd.read_csv(p)
                    if df.empty:
                        raise ValueError("CSV has no rows")
                    cols = map_columns(df)
                    df[cols['time']] = parse_time_series(df[cols['time']])
                    df[cols['lat']] = pd.to_numeric(df[cols['lat']], errors='coerce')
                    df[cols['lon']] = pd.to_numeric(df[cols['lon']], errors='coerce')
                    df = df.dropna(subset=[cols['time'], cols['lat'], cols['lon']]).reset_index(drop=True)
                    if df.empty:
                        raise ValueError("All required columns have NaN values")
                    df = df.sort_values(cols['time']).reset_index(drop=True)
                    traj = Trajectory(path=p, df=df, columns=cols)
                    self.trajectories.append(traj)
                    self.add_trajectory_to_table(traj)
                except Exception as e:
                    QMessageBox.warning(self, "Load Error", f"Failed to load {p}:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def on_select_file(self):
        """Handle file selection from table"""
        current_row = self.file_table.currentRow()
        if current_row < 0:
            return

        # First try getting the trajectory directly from the UserRole
        item0 = self.file_table.item(current_row, 0)
        traj = item0.data(Qt.UserRole) if item0 is not None else None

        # Fallback for older rows (no UserRole) — find by file name
        if traj is None:
            filename = item0.text() if item0 else None
            for t in self.trajectories:
                if os.path.basename(t.path) == filename:
                    traj = t
                    break

        # Still nothing? Abort.
        if traj is None:
            return

        # Plot and update info
        self.render_trajectory(traj)
        self.populate_stats(traj)

    def render_trajectory(self, traj: Trajectory):
        # Set waiting cursor for potentially long rendering operation
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            self.plot.clear()
            lat = traj.columns['lat']
            lon = traj.columns['lon']
            lats = traj.df[lat].to_numpy()
            lons = traj.df[lon].to_numpy()
            self.curve = pg.PlotCurveItem(x=lons, y=lats, pen=pg.mkPen(width=2, color='b'))
            self.plot.addItem(self.curve)
            spots = []
            for i, row in traj.df.iterrows():
                data = {'index': int(i)}
                for k, col in traj.columns.items():
                    data[k] = row[col]
                spots.append({'pos': (row[lon], row[lat]), 'data': data, 'size': 6, 'brush': pg.mkBrush(0, 0, 255, 150)})
            self.scatter = pg.ScatterPlotItem()
            self.scatter.addPoints(spots)
            self.plot.addItem(self.scatter)
            self.selected_point = None
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

    def toggle_basemap(self, state):
        if state == Qt.Checked:
            QMessageBox.information(self, "Basemap", "Basemap display not implemented yet.")

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