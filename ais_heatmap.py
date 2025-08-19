import sys
import os
import tempfile
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QFileDialog, QLabel, QSlider, 
                            QSpinBox, QComboBox, QProgressBar, QStatusBar, QCheckBox,
                            QGroupBox, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
                            QHeaderView, QTabWidget, QAbstractItemView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QFont, QIcon
import folium
from folium.plugins import HeatMap, MarkerCluster, MeasureControl
import json
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import warnings
from geopy.distance import geodesic
from datetime import datetime
import dateutil.parser
warnings.filterwarnings('ignore')
from typing import Dict, List, Optional, Union

class NumericTableWidgetItem(QTableWidgetItem):
    """Table item with numeric sorting capability"""
    def __init__(self, text: str, numeric_value: Union[int, float]):
        super().__init__(text)
        self.numeric_value = numeric_value
    
    def __lt__(self, other):
        return (self.numeric_value < other.numeric_value 
                if isinstance(other, NumericTableWidgetItem) 
                else super().__lt__(other))

class DataProcessor(QThread):
    """Background thread for processing CSV data"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, object)  # data, stats, trajectory_stats
    error = pyqtSignal(str)
    
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        
    def run(self):
        try:
            all_data = []
            file_stats = []
            total_files = len(self.file_paths)
            
            for i, file_path in enumerate(self.file_paths):
                df = pd.read_csv(file_path)
                
                # Common AIS column name variations
                lat_cols = ['lat', 'latitude', 'LAT', 'LATITUDE', 'Latitude', 'y', 'Y']
                lon_cols = ['lon', 'longitude', 'lng', 'LON', 'LONGITUDE', 'Longitude', 'x', 'X']
                time_cols = ['timestamp', 'time', 'datetime', 'BaseDateTime','TIME', 'TIMESTAMP', 'DATETIME']
                mmsi_cols = ['mmsi', 'MMSI', 'vessel_id', 'ship_id', 'id', 'ID']
                
                lat_col = lon_col = time_col = mmsi_col = None
                
                # Find the correct column names
                for col in df.columns:
                    if col in lat_cols and lat_col is None:
                        lat_col = col
                    elif col in lon_cols and lon_col is None:
                        lon_col = col
                    elif col in time_cols and time_col is None:
                        time_col = col
                    elif col in mmsi_cols and mmsi_col is None:
                        mmsi_col = col
                
                if lat_col is None or lon_col is None:
                    # Fallback: use first two numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        lat_col, lon_col = numeric_cols[:2]
                    else:
                        raise ValueError(f"Could not find latitude/longitude columns in {file_path}")
                
                # Prepare data columns
                data_cols = [lat_col, lon_col]
                if time_col is not None:
                    data_cols.append(time_col)
                if mmsi_col is not None:
                    data_cols.append(mmsi_col)
                
                # Clean and filter data
                df_clean = df[data_cols].dropna()
                
                # Filter reasonable lat/lon values
                df_clean = df_clean[
                    (df_clean[lat_col] >= -90) & (df_clean[lat_col] <= 90) &
                    (df_clean[lon_col] >= -180) & (df_clean[lon_col] <= 180)
                ]
                
                # Store file statistics
                file_stats.append({
                    'filename': os.path.basename(file_path),
                    'total_records': len(df),
                    'valid_records': len(df_clean),
                    'unique_vessels': df_clean[mmsi_col].nunique() if mmsi_col else 'N/A',
                    'lat_range': (df_clean[lat_col].min(), df_clean[lat_col].max()),
                    'lon_range': (df_clean[lon_col].min(), df_clean[lon_col].max())
                })
                
                # Rename columns for consistency
                df_clean = df_clean.rename(columns={
                    lat_col: 'latitude',
                    lon_col: 'longitude'
                })
                if time_col:
                    df_clean = df_clean.rename(columns={time_col: 'timestamp'})
                if mmsi_col:
                    df_clean = df_clean.rename(columns={mmsi_col: 'mmsi'})
                
                all_data.append(df_clean)
                self.progress.emit(int((i + 1) / total_files * 100))
            
            # Combine all data
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Calculate trajectory statistics
                trajectory_stats = self.calculate_trajectory_stats(combined_df)
                
                self.finished.emit(combined_df, file_stats, trajectory_stats)
            else:
                self.error.emit("No valid data found in the selected files")
                
        except Exception as e:
            self.error.emit(str(e))
    
    def calculate_trajectory_stats(self, df):
        """Calculate statistics for each trajectory/vessel"""
        trajectory_stats = []
        
        if 'mmsi' not in df.columns:
            return trajectory_stats
        
        for mmsi, group in df.groupby('mmsi'):
            group = group.sort_values('timestamp') if 'timestamp' in group.columns else group
            
            # Calculate distance
            total_distance = 0
            if len(group) > 1:
                for i in range(1, len(group)):
                    try:
                        dist = geodesic(
                            (group.iloc[i-1]['latitude'], group.iloc[i-1]['longitude']),
                            (group.iloc[i]['latitude'], group.iloc[i]['longitude'])
                        ).kilometers
                        total_distance += dist
                    except:
                        pass
            
            # Calculate duration
            total_duration = 'N/A'
            if 'timestamp' in group.columns and len(group) > 1:
                try:
                    # Try multiple timestamp parsing methods
                    timestamps = []
                    for ts in [group.iloc[0]['timestamp'], group.iloc[-1]['timestamp']]:
                        if pd.isna(ts):
                            continue
                        # Convert to string first if it's not already
                        ts_str = str(ts)
                        try:
                            # Try pandas parsing first
                            parsed_ts = pd.to_datetime(ts_str)
                            timestamps.append(parsed_ts)
                        except:
                            try:
                                # Try dateutil parsing for more flexible formats
                                parsed_ts = dateutil.parser.parse(ts_str)
                                timestamps.append(pd.to_datetime(parsed_ts))
                            except:
                                continue
                    
                    if len(timestamps) >= 2:
                        duration = timestamps[-1] - timestamps[0]
                        # Format duration nicely
                        total_seconds = int(duration.total_seconds())
                        if total_seconds > 0:
                            days = total_seconds // 86400
                            hours = (total_seconds % 86400) // 3600
                            minutes = (total_seconds % 3600) // 60
                            
                            if days > 0:
                                total_duration = f"{days}d {hours}h {minutes}m"
                            elif hours > 0:
                                total_duration = f"{hours}h {minutes}m"
                            else:
                                total_duration = f"{minutes}m"
                        else:
                            total_duration = "0m"
                except Exception as e:
                    total_duration = 'N/A'
            
            # Calculate bounds
            lat_min, lat_max = group['latitude'].min(), group['latitude'].max()
            lon_min, lon_max = group['longitude'].min(), group['longitude'].max()
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            
            trajectory_stats.append({
                'MMSI': str(mmsi),
                'Data Points': len(group),
                'Distance (km)': round(total_distance, 2),
                'Duration': total_duration,
                'Center Lat': round(center_lat, 6),
                'Center Lon': round(center_lon, 6),
                'Lat Range': f"{lat_min:.4f} - {lat_max:.4f}",
                'Lon Range': f"{lon_min:.4f} - {lon_max:.4f}"
            })
        
        return trajectory_stats

class TrajectoryTableWidget(QTableWidget):
    """Interactive table for trajectory information"""
    trajectory_selected = pyqtSignal(str, float, float)  # mmsi, center_lat, center_lon
    
    def __init__(self):
        super().__init__()
        self.setup_table()
        self.trajectory_data = []
        
    def setup_table(self):
        """Setup the table appearance and behavior"""
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        
        # Connect selection signal
        self.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Set column headers
        headers = ['MMSI', 'Data Points', 'Distance (km)', 'Duration', 
                  'Center Lat', 'Center Lon', 'Lat Range', 'Lon Range']
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        
        # Set header resize mode
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # MMSI
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Data Points
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Distance
        header.setSectionResizeMode(3, QHeaderView.Stretch)  # Duration
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Center Lat
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Center Lon
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # Lat Range
        header.setSectionResizeMode(7, QHeaderView.Stretch)  # Lon Range
    
    def load_trajectory_data(self, trajectory_stats):
        """Load trajectory data into the table"""
        self.trajectory_data = trajectory_stats
        self.setRowCount(len(trajectory_stats))
        
        for row, data in enumerate(trajectory_stats):
            self.setItem(row, 0, NumericTableWidgetItem(str(data['MMSI']), data['MMSI']))
            self.setItem(row, 1, NumericTableWidgetItem(str(data['Data Points']), data['Data Points']))
            self.setItem(row, 2, NumericTableWidgetItem(str(data['Distance (km)']), data['Distance (km)']))
            self.setItem(row, 3, NumericTableWidgetItem(str(data['Duration']), data['Duration']))
            self.setItem(row, 4, NumericTableWidgetItem(str(data['Center Lat']), data['Center Lat']))
            self.setItem(row, 5, NumericTableWidgetItem(str(data['Center Lon']), data['Center Lon']))
            self.setItem(row, 6, NumericTableWidgetItem(str(data['Lat Range']), data['Lat Range']))
            self.setItem(row, 7, NumericTableWidgetItem(str(data['Lon Range']),data['Lon Range']))
            
            # Make numeric columns sortable as numbers
            for col in [1, 2, 4, 5]:  # Data Points, Distance, Center Lat, Center Lon
                item = self.item(row, col)
                if col in [1]:  # Integer columns
                    item.setData(Qt.UserRole, int(float(data[self.horizontalHeaderItem(col).text()])))
                else:  # Float columns
                    item.setData(Qt.UserRole, float(data[self.horizontalHeaderItem(col).text()]))
    
    def on_selection_changed(self):
        """Handle row selection"""
        current_row = self.currentRow()
        if current_row >= 0 and current_row < len(self.trajectory_data):
            data = self.trajectory_data[current_row]
            self.trajectory_selected.emit(
                data['MMSI'], 
                data['Center Lat'], 
                data['Center Lon']
            )

class FoliumMapWidget(QWidget):
    """Widget containing the Folium map with web engine"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.current_map = None
        self.temp_file = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create web engine view
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        
        # Initialize with empty map
        self.create_empty_map()
        
    def create_empty_map(self):
        """Create an empty map centered on world view"""
        m = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        # Add some basic controls
        folium.plugins.Fullscreen().add_to(m)
        MeasureControl().add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        self.current_map = m
        self.display_map()
        
    def load_data(self, data):
        """Load AIS data into the widget"""
        self.data = data
        
    def create_heatmap(self, intensity=1.0, radius=15, blur=15, max_zoom=18, 
                      gradient=None, min_opacity=0.4, sample_size=None):
        """Create heatmap visualization"""
        if self.data is None or len(self.data) == 0:
            return
            
        # Sample data if requested for better performance
        df = self.data.copy()
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            
        # Calculate map center
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create base map with OpenStreetMap as default (and keep it as default)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.calculate_zoom_level(df),
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        # Add different tile layers with proper attributions but don't make any default
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
            name='Stamen Terrain',
            max_zoom=18,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
            name='Stamen Toner',
            max_zoom=18,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='CartoDB Positron',
            max_zoom=19,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='CartoDB Dark Matter',
            max_zoom=19,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            name='Satellite',
            max_zoom=19,
            show=False
        ).add_to(m)
        
        # Prepare heatmap data
        heat_data = df[['latitude', 'longitude']].values.tolist()
        
        # Create gradient dictionary
        if gradient is None:
            gradient = {
                0.2: 'blue',
                0.4: 'lime', 
                0.6: 'orange',
                0.8: 'red',
                1.0: 'darkred'
            }
        
        # Add heatmap layer
        HeatMap(
            heat_data,
            name='AIS Density Heatmap',
            min_opacity=min_opacity,
            radius=radius,
            blur=blur,
            max_zoom=max_zoom,
            gradient=gradient
        ).add_to(m)
        
        # Add individual points layer (optional, controlled by layer control)
        self.add_points_layer(m, df)
        
        # Add trajectory layer (optional, controlled by layer control)
        self.add_trajectory_layer(m, df)
        
        # Add statistics popup
        self.add_statistics_popup(m, df)
        
        # Add controls
        folium.plugins.Fullscreen().add_to(m)
        MeasureControl().add_to(m)
        folium.LayerControl().add_to(m)
        
        # Add mini map
        minimap = folium.plugins.MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        self.current_map = m
        self.display_map()
    
    def add_points_layer(self, map_obj, df):
        """Add individual points as a separate layer"""
        points_layer = folium.FeatureGroup(name='Individual Points', show=False)
        
        # Sample points for performance
        point_sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        if 'mmsi' in point_sample.columns:
            # Color code by vessel
            unique_vessels = point_sample['mmsi'].unique()
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                     'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                     'gray', 'black', 'lightgray']
            
            vessel_colors = {vessel: colors[i % len(colors)] 
                           for i, vessel in enumerate(unique_vessels[:100])}
            
            for _, row in point_sample.iterrows():
                color = vessel_colors.get(row['mmsi'], 'gray')
                popup_text = f"MMSI: {row['mmsi']}<br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}"
                if 'timestamp' in row:
                    popup_text += f"<br>Time: {row['timestamp']}"
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    popup=popup_text,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(points_layer)
        else:
            for _, row in point_sample.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=2,
                    popup=f"Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}",
                    color='red',
                    fillColor='red',
                    fillOpacity=0.6
                ).add_to(points_layer)
        
        points_layer.add_to(map_obj)
    
    def add_trajectory_layer(self, map_obj, df):
        """Add vessel trajectories as a separate layer"""
        if 'mmsi' not in df.columns:
            return
            
        trajectory_layer = folium.FeatureGroup(name='Vessel Trajectories', show=False)
        
        vessel_groups = df.groupby('mmsi')
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']
        
        for i, (vessel_id, vessel_data) in enumerate(vessel_groups):
            if len(vessel_data) < 2:
                continue
                
            if i >= 20:  # Limit to first 20 vessels for performance
                break
                
            # Sort by timestamp if available
            if 'timestamp' in vessel_data.columns:
                vessel_data = vessel_data.sort_values('timestamp')
            
            # Create trajectory line
            trajectory_coords = vessel_data[['latitude', 'longitude']].values.tolist()
            color = colors[i % len(colors)]
            
            folium.PolyLine(
                locations=trajectory_coords,
                color=color,
                weight=2,
                opacity=0.7,
                popup=f'Vessel {vessel_id} - {len(vessel_data)} points'
            ).add_to(trajectory_layer)
            
            # Add start and end markers
            if len(trajectory_coords) > 1:
                folium.Marker(
                    location=trajectory_coords[0],
                    popup=f'Start - Vessel {vessel_id}',
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(trajectory_layer)
                
                folium.Marker(
                    location=trajectory_coords[-1],
                    popup=f'End - Vessel {vessel_id}',
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(trajectory_layer)
        
        trajectory_layer.add_to(map_obj)
    
    def add_statistics_popup(self, map_obj, df):
        """Add statistics information to the map"""
        stats = {
            'Total Points': f"{len(df):,}",
            'Unique Vessels': df['mmsi'].nunique() if 'mmsi' in df.columns else 'N/A',
            'Lat Range': f"{df['latitude'].min():.4f} to {df['latitude'].max():.4f}",
            'Lon Range': f"{df['longitude'].min():.4f} to {df['longitude'].max():.4f}",
        }
        
        if 'timestamp' in df.columns:
            stats['Time Range'] = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        
        stats_html = """
        <div style="background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <h4 style="margin-top: 0;">AIS Data Statistics</h4>
        """
        
        for key, value in stats.items():
            stats_html += f"<b>{key}:</b> {value}<br>"
        
        stats_html += "</div>"
        
        # Add to top-right corner
        folium.plugins.FloatImage(
            stats_html,
            bottom=85,
            left=85
        ).add_to(map_obj)
    
    def focus_on_trajectory(self, mmsi, center_lat, center_lon):
        """Focus the map on a specific trajectory without changing the base layer"""
        if self.current_map is None:
            return
        
        # Create a new map centered on the trajectory but keep OpenStreetMap
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        # Add all the same layers as the main map (but keep them hidden)
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
            name='Stamen Terrain',
            max_zoom=18,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
            name='Stamen Toner',
            max_zoom=18,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='CartoDB Positron',
            max_zoom=19,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='CartoDB Dark Matter',
            max_zoom=19,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            name='Satellite',
            max_zoom=19,
            show=False
        ).add_to(m)
        
        # Filter data for the specific vessel
        if self.data is not None and 'mmsi' in self.data.columns:
            vessel_data = self.data[self.data['mmsi'] == mmsi]
            
            # Add heatmap for this vessel
            if len(vessel_data) > 0:
                heat_data = vessel_data[['latitude', 'longitude']].values.tolist()
                HeatMap(
                    heat_data,
                    name=f'Vessel {mmsi} Heatmap',
                    min_opacity=0.4,
                    radius=15,
                    blur=15,
                    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}
                ).add_to(m)
                
                # Add trajectory line for this vessel
                if len(vessel_data) > 1:
                    trajectory_coords = vessel_data[['latitude', 'longitude']].values.tolist()
                    folium.PolyLine(
                        locations=trajectory_coords,
                        color='red',
                        weight=3,
                        opacity=0.8,
                        popup=f'Vessel {mmsi} Trajectory'
                    ).add_to(m)
                    
                    # Add start and end markers
                    folium.Marker(
                        location=trajectory_coords[0],
                        popup=f'Start - Vessel {mmsi}',
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(m)
                    
                    folium.Marker(
                        location=trajectory_coords[-1],
                        popup=f'End - Vessel {mmsi}',
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(m)
        
        # Add controls
        folium.plugins.Fullscreen().add_to(m)
        MeasureControl().add_to(m)
        folium.LayerControl().add_to(m)
        
        # Update the current map and display
        self.current_map = m
        self.display_map()
    
    def calculate_zoom_level(self, df):
        """Calculate appropriate zoom level based on data extent"""
        lat_range = df['latitude'].max() - df['latitude'].min()
        lon_range = df['longitude'].max() - df['longitude'].min()
        
        max_range = max(lat_range, lon_range)
        
        if max_range > 50:
            return 3
        elif max_range > 20:
            return 4
        elif max_range > 10:
            return 5
        elif max_range > 5:
            return 6
        elif max_range > 2:
            return 7
        elif max_range > 1:
            return 8
        elif max_range > 0.5:
            return 9
        else:
            return 10
    
    def display_map(self):
        """Display the current map in the web view"""
        if self.current_map is None:
            return
            
        # Clean up previous temp file
        if hasattr(self, 'temp_file') and self.temp_file and os.path.exists(self.temp_file):
            try:
                os.unlink(self.temp_file)
            except:
                pass
        
        # Save map to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            self.current_map.save(f.name)
            self.temp_file = f.name
        
        # Load in web view
        self.web_view.load(QUrl.fromLocalFile(os.path.abspath(self.temp_file)))
    
    def closeEvent(self, event):
        """Clean up temp file on close"""
        if hasattr(self, 'temp_file') and self.temp_file and os.path.exists(self.temp_file):
            try:
                os.unlink(self.temp_file)
            except:
                pass
        super().closeEvent(event)

class AISHeatmapViewer(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.processor = None
        self.file_stats = []
        self.trajectory_stats = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Advanced AIS Trajectory Heatmap Viewer')
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Create left panel with controls and trajectory table
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Create map widget
        self.map_widget = FoliumMapWidget()
        main_splitter.addWidget(self.map_widget)
        
        # Set splitter sizes (left panel smaller)
        main_splitter.setSizes([400, 1200])
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Ready - Load CSV files to begin')
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.statusBar.addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self):
        """Create the left panel with controls and trajectory table"""
        panel = QWidget()
        panel.setFixedWidth(380)
        layout = QVBoxLayout(panel)
        
        # Create tab widget for controls and trajectory table
        tab_widget = QTabWidget()
        
        # Controls tab
        controls_tab = self.create_controls_tab()
        tab_widget.addTab(controls_tab, "🎛️ Controls")
        
        # Trajectory table tab
        trajectory_tab = self.create_trajectory_tab()
        tab_widget.addTab(trajectory_tab, "📊 Trajectories")
        
        layout.addWidget(tab_widget)
        
        return panel
    
    def create_controls_tab(self):
        """Create the controls tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel('Heatmap Controls')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # File Controls Group
        file_group = QGroupBox('Data Loading')
        file_layout = QVBoxLayout(file_group)
        
        self.load_button = QPushButton('📁 Load CSV Files')
        self.load_button.clicked.connect(self.load_files)
        file_layout.addWidget(self.load_button)
        
        self.file_count_label = QLabel('No files loaded')
        self.file_count_label.setWordWrap(True)
        file_layout.addWidget(self.file_count_label)
        
        layout.addWidget(file_group)
        
        # Heatmap Controls Group
        heatmap_group = QGroupBox('Heatmap Settings')
        heatmap_layout = QGridLayout(heatmap_group)
        
        # Intensity control
        heatmap_layout.addWidget(QLabel('Intensity:'), 0, 0)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 50)
        self.intensity_slider.setValue(10)
        heatmap_layout.addWidget(self.intensity_slider, 0, 1)
        self.intensity_label = QLabel('1.0')
        heatmap_layout.addWidget(self.intensity_label, 0, 2)
        
        # Radius control
        heatmap_layout.addWidget(QLabel('Radius:'), 1, 0)
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(5, 50)
        self.radius_slider.setValue(15)
        heatmap_layout.addWidget(self.radius_slider, 1, 1)
        self.radius_label = QLabel('15')
        heatmap_layout.addWidget(self.radius_label, 1, 2)
        
        # Blur control
        heatmap_layout.addWidget(QLabel('Blur:'), 2, 0)
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(5, 50)
        self.blur_slider.setValue(15)
        heatmap_layout.addWidget(self.blur_slider, 2, 1)
        self.blur_label = QLabel('15')
        heatmap_layout.addWidget(self.blur_label, 2, 2)
        
        # Min opacity control
        heatmap_layout.addWidget(QLabel('Min Opacity:'), 3, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(1, 10)
        self.opacity_slider.setValue(4)
        heatmap_layout.addWidget(self.opacity_slider, 3, 1)
        self.opacity_label = QLabel('0.4')
        heatmap_layout.addWidget(self.opacity_label, 3, 2)
        
        # Gradient selection
        heatmap_layout.addWidget(QLabel('Color Scheme:'), 4, 0)
        self.gradient_combo = QComboBox()
        self.gradient_combo.addItems([
            'Heat (Default)', 'Ocean', 'Plasma', 'Viridis', 'Cool', 'Rainbow'
        ])
        heatmap_layout.addWidget(self.gradient_combo, 4, 1, 1, 2)
        
        layout.addWidget(heatmap_group)
        
        # Display Options Group
        display_group = QGroupBox('Display Options')
        display_layout = QVBoxLayout(display_group)
        
        # Sample size control
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel('Sample Size:'))
        self.sample_spinbox = QSpinBox()
        self.sample_spinbox.setRange(1000, 1000000)
        self.sample_spinbox.setValue(50000)
        self.sample_spinbox.setSuffix(' points')
        sample_layout.addWidget(self.sample_spinbox)
        display_layout.addLayout(sample_layout)
        
        layout.addWidget(display_group)
        
        # Action Buttons
        button_layout = QVBoxLayout()
        
        self.update_button = QPushButton('🔄 Update Heatmap')
        self.update_button.clicked.connect(self.update_heatmap)
        self.update_button.setEnabled(False)
        button_layout.addWidget(self.update_button)
        
        self.reset_button = QPushButton('🏠 Reset View')
        self.reset_button.clicked.connect(self.reset_view)
        self.reset_button.setEnabled(False)
        button_layout.addWidget(self.reset_button)
        
        self.export_button = QPushButton('💾 Export Map')
        self.export_button.clicked.connect(self.export_map)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        
        # Connect sliders to update labels
        self.intensity_slider.valueChanged.connect(
            lambda v: self.intensity_label.setText(f'{v/10:.1f}'))
        self.radius_slider.valueChanged.connect(
            lambda v: self.radius_label.setText(str(v)))
        self.blur_slider.valueChanged.connect(
            lambda v: self.blur_label.setText(str(v)))
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f'{v/10:.1f}'))
        
        layout.addStretch()
        
        # Instructions
        instructions = QLabel(
            '<b>🗺️ Map Controls:</b><br>'
            '• Mouse wheel: Zoom in/out<br>'
            '• Left drag: Pan around<br>'
            '• Layer control: Toggle layers<br>'
            '• Measure tool: Distance/area<br>'
            '• Fullscreen: F11 or button<br><br>'
            '<b>💡 Tips:</b><br>'
            '• Use sampling for large datasets<br>'
            '• Try different color schemes<br>'
            '• Check Trajectories tab for details'
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet('''
            QLabel { 
                background-color: #f8f9fa; 
                padding: 12px; 
                border-radius: 8px; 
                border: 1px solid #dee2e6;
                font-size: 10px;
            }
        ''')
        layout.addWidget(instructions)
        
        return widget
    
    def create_trajectory_tab(self):
        """Create the trajectory analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel('Trajectory Analysis')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Info label
        info_label = QLabel('Select a trajectory to focus on map:')
        info_label.setFont(QFont('Arial', 10))
        layout.addWidget(info_label)
        
        # Create trajectory table
        self.trajectory_table = TrajectoryTableWidget()
        self.trajectory_table.trajectory_selected.connect(self.on_trajectory_selected)
        layout.addWidget(self.trajectory_table)
        
        # Summary stats
        self.trajectory_summary = QLabel('No trajectory data loaded')
        self.trajectory_summary.setWordWrap(True)
        self.trajectory_summary.setStyleSheet('''
            QLabel { 
                background-color: #e9ecef; 
                padding: 8px; 
                border-radius: 5px; 
                font-size: 10px;
                margin-top: 5px;
            }
        ''')
        layout.addWidget(self.trajectory_summary)
        
        return widget
        
    def load_files(self):
        """Load CSV files with enhanced error handling"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select AIS CSV Files', '', 
            'CSV files (*.csv);;All files (*.*)')
        
        if not file_paths:
            return
            
        self.file_count_label.setText(f'Loading {len(file_paths)} files...')
        self.load_button.setEnabled(False)
        self.progress_bar.show()
        
        # Start background processing
        self.processor = DataProcessor(file_paths)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.finished.connect(self.on_data_loaded)
        self.processor.error.connect(self.on_error)
        self.processor.start()
    
    def on_data_loaded(self, data, file_stats, trajectory_stats):
        """Handle loaded data with enhanced statistics"""
        self.data = data
        self.file_stats = file_stats
        self.trajectory_stats = trajectory_stats
        self.map_widget.load_data(data)
        
        # Update UI
        stats_text = f'{len(data):,} AIS positions loaded\n'
        if 'mmsi' in data.columns:
            stats_text += f'{data["mmsi"].nunique()} unique vessels\n'
        stats_text += f'From {len(file_stats)} files'
        
        self.file_count_label.setText(stats_text)
        self.load_button.setEnabled(True)
        self.update_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.progress_bar.hide()
        
        # Load trajectory table
        if trajectory_stats:
            self.trajectory_table.load_trajectory_data(trajectory_stats)
            self.update_trajectory_summary()
        
        # Auto-generate initial heatmap
        self.update_heatmap()
        
        self.statusBar.showMessage(f'Successfully loaded {len(data):,} AIS positions')
    
    def on_error(self, error_msg):
        """Handle loading errors with user-friendly messages"""
        self.file_count_label.setText(f'❌ Error loading data')
        self.load_button.setEnabled(True)
        self.progress_bar.hide()
        self.statusBar.showMessage(f'Error: {error_msg}')
    
    def update_trajectory_summary(self):
        """Update trajectory summary statistics"""
        if not self.trajectory_stats:
            return
        
        total_trajectories = len(self.trajectory_stats)
        total_points = sum(t['Data Points'] for t in self.trajectory_stats)
        total_distance = sum(t['Distance (km)'] for t in self.trajectory_stats if isinstance(t['Distance (km)'], (int, float)))
        
        avg_points = total_points / total_trajectories if total_trajectories > 0 else 0
        avg_distance = total_distance / total_trajectories if total_trajectories > 0 else 0
        
        summary_text = f"""<b>Trajectory Summary:</b><br>
        • Total Trajectories: {total_trajectories}<br>
        • Total Data Points: {total_points:,}<br>
        • Total Distance: {total_distance:,.1f} km<br>
        • Avg Points per Trajectory: {avg_points:.1f}<br>
        • Avg Distance per Trajectory: {avg_distance:.1f} km"""
        
        self.trajectory_summary.setText(summary_text)
    
    def on_trajectory_selected(self, mmsi, center_lat, center_lon):
        """Handle trajectory selection from table"""
        self.map_widget.focus_on_trajectory(mmsi, center_lat, center_lon)
        self.statusBar.showMessage(f'Focused on trajectory for vessel {mmsi}')
    
    def get_gradient_dict(self, gradient_name):
        """Get gradient dictionary for different color schemes"""
        gradients = {
            'Heat (Default)': {0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'},
            'Ocean': {0.2: 'navy', 0.4: 'blue', 0.6: 'aqua', 0.8: 'lime', 1.0: 'yellow'},
            'Plasma': {0.2: 'purple', 0.4: 'darkred', 0.6: 'red', 0.8: 'orange', 1.0: 'yellow'},
            'Viridis': {0.2: 'purple', 0.4: 'darkblue', 0.6: 'green', 0.8: 'lime', 1.0: 'yellow'},
            'Cool': {0.2: 'darkblue', 0.4: 'blue', 0.6: 'aqua', 0.8: 'lime', 1.0: 'white'},
            'Rainbow': {0.1: 'purple', 0.3: 'blue', 0.5: 'green', 0.7: 'yellow', 0.9: 'red'}
        }
        return gradients.get(gradient_name, gradients['Heat (Default)'])
    
    def update_heatmap(self):
        """Update heatmap with current settings"""
        if self.data is None:
            return
            
        # Get current settings
        intensity = self.intensity_slider.value() / 10.0
        radius = self.radius_slider.value()
        blur = self.blur_slider.value()
        min_opacity = self.opacity_slider.value() / 10.0
        gradient = self.get_gradient_dict(self.gradient_combo.currentText())
        sample_size = self.sample_spinbox.value() if len(self.data) > self.sample_spinbox.value() else None
        
        # Update map
        self.map_widget.create_heatmap(
            intensity=intensity,
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            gradient=gradient,
            sample_size=sample_size
        )
        
        self.statusBar.showMessage('Heatmap updated successfully')
    
    def reset_view(self):
        """Reset map view to show all data"""
        if self.data is None:
            return
        
        # Recreate map with default zoom
        self.update_heatmap()
        self.statusBar.showMessage('View reset to show all data')
    
    def export_map(self):
        """Export current map to HTML file"""
        if self.map_widget.current_map is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Map', 'ais_heatmap.html', 
            'HTML files (*.html);;All files (*.*)')
        
        if file_path:
            try:
                self.map_widget.current_map.save(file_path)
                self.statusBar.showMessage(f'Map exported to {file_path}')
            except Exception as e:
                self.statusBar.showMessage(f'Export failed: {str(e)}')
    
    def closeEvent(self, event):
        """Handle application close"""
        if hasattr(self.map_widget, 'temp_file') and self.map_widget.temp_file:
            if os.path.exists(self.map_widget.temp_file):
                try:
                    os.unlink(self.map_widget.temp_file)
                except:
                    pass
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName('AIS Heatmap Viewer')
    app.setOrganizationName('AIS Analytics')
    
    # Set application style
    app.setStyle('Fusion')
    
    # Apply modern theme
    palette = app.palette()
    palette.setColor(palette.Window, palette.color(palette.Base))
    app.setPalette(palette)
    
    # Create and show main window
    viewer = AISHeatmapViewer()
    viewer.show()
    
    # Handle cleanup on exit
    def cleanup():
        if hasattr(viewer, 'map_widget') and hasattr(viewer.map_widget, 'temp_file'):
            if viewer.map_widget.temp_file and os.path.exists(viewer.map_widget.temp_file):
                try:
                    os.unlink(viewer.map_widget.temp_file)
                except:
                    pass
    
    app.aboutToQuit.connect(cleanup)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()