import sys
import json
import csv
import requests
import websocket
import threading
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QTextEdit, 
                             QLabel, QLineEdit, QComboBox, QDateTimeEdit, 
                             QTableWidget, QTableWidgetItem, QFileDialog,
                             QMessageBox, QProgressBar, QCheckBox, QSpinBox,
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, QDateTime, QTimer
from PyQt5.QtGui import QFont, QTextCursor
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import pandas as pd

class AISStreamThread(QThread):
    """Thread for handling real-time AIS data from AISStream.io"""
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        
    def run(self):
        self.running = True
        self.reconnect_attempts = 0
        
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.connection_status.emit(f"Connecting to AISStream.io... (attempt {self.reconnect_attempts + 1})")
                self.ws = websocket.WebSocketApp("wss://stream.aisstream.io/v0/stream",
                                                on_open=self.on_open,
                                                on_message=self.on_message,
                                                on_error=self.on_error,
                                                on_close=self.on_close)
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
                
                if not self.running:
                    break
                    
                self.reconnect_attempts += 1
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.connection_status.emit(f"Reconnecting in 5 seconds...")
                    time.sleep(5)
                    
            except Exception as e:
                self.error_occurred.emit(f"Connection error: {str(e)}")
                self.reconnect_attempts += 1
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    time.sleep(5)
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.error_occurred.emit("Maximum reconnection attempts reached")
    
    def on_open(self, ws):
        self.connection_status.emit("Connected to AISStream.io")
        # Subscribe to global AIS data - Correct format for AISStream.io API
        api_key = getattr(self, 'api_key', 'guest')
        subscribe_message = {
            "APIKey": api_key,
            "BoundingBoxes": [[[-90, -180], [90, 180]]]  # Global coverage: [[lat, lon], [lat, lon]]
        }
        self.connection_status.emit(f"Subscribing with API key: {api_key}")
        ws.send(json.dumps(subscribe_message))
    
    def on_message(self, ws, message):
        if self.running:
            try:
                data = json.loads(message)
                # Check for error messages from the server
                if "error" in data:
                    self.error_occurred.emit(f"Server error: {data['error']}")
                elif "Message" in data:  # AISStream.io message format
                    self.data_received.emit(data)
                else:
                    # Handle other message types
                    self.data_received.emit(data)
            except json.JSONDecodeError as e:
                self.error_occurred.emit(f"JSON decode error: {str(e)} - Raw message: {message[:200]}")
    
    def on_error(self, ws, error):
        if self.running:
            self.error_occurred.emit(f"WebSocket error: {str(error)}")
    
    def on_close(self, ws, close_status_code, close_msg):
        if self.running:
            close_reason = f"Code: {close_status_code}, Message: {close_msg}" if close_status_code else "Unknown reason"
            self.connection_status.emit(f"Connection closed - {close_reason}")
        else:
            self.connection_status.emit("Disconnected from AISStream.io")
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

class HistoricalDataThread(QThread):
    """Thread for fetching historical AIS data"""
    data_received = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, source, params):
        super().__init__()
        self.source = source
        self.params = params
        
    def run(self):
        try:
            if self.source == "AISHub":
                self.fetch_aishub_data()
            elif self.source == "MarineTraffic":
                self.fetch_marinetraffic_data()
            elif self.source == "Global Fishing Watch":
                self.fetch_gfw_data()
        except Exception as e:
            self.error_occurred.emit(f"Historical data error: {str(e)}")
    
    def fetch_aishub_data(self):
        """Fetch data from AISHub"""
        base_url = "http://data.aishub.net/ws.php"
        params = {
            'username': self.params.get('username', 'guest'),
            'format': '1',  # JSON format
            'output': 'json',
            'compress': '0'
        }
        
        # Add geographic bounds if provided
        if 'lat_min' in self.params:
            params.update({
                'latmin': self.params['lat_min'],
                'latmax': self.params['lat_max'],
                'lonmin': self.params['lon_min'],
                'lonmax': self.params['lon_max']
            })
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                self.data_received.emit(data)
            else:
                self.data_received.emit([data])
        except requests.RequestException as e:
            self.error_occurred.emit(f"AISHub request failed: {str(e)}")
    
    def fetch_marinetraffic_data(self):
        """Fetch data from MarineTraffic (requires API key)"""
        # This is a placeholder for MarineTraffic API
        # You would need a valid API key and endpoint
        self.error_occurred.emit("MarineTraffic requires API key - not implemented in demo")
    
    def fetch_gfw_data(self):
        """Fetch data from Global Fishing Watch"""
        # This is a placeholder for GFW API
        # You would need to implement the specific GFW API calls
        self.error_occurred.emit("Global Fishing Watch API - not implemented in demo")

class AISDataRetriever(QMainWindow):
    def __init__(self):
        super().__init__()
        self.realtime_thread = None
        self.historical_thread = None
        self.received_messages = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("AIS Data Retriever - Real-time & Historical")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Real-time data tab
        self.create_realtime_tab(tab_widget)
        
        # Historical data tab
        self.create_historical_tab(tab_widget)
        
        # Data viewer tab
        self.create_data_viewer_tab(tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_realtime_tab(self, parent):
        """Create real-time data retrieval tab"""
        realtime_widget = QWidget()
        layout = QVBoxLayout(realtime_widget)
        
        # Connection controls
        conn_group = QGroupBox("Real-time Connection")
        conn_layout = QGridLayout(conn_group)
        
        self.status_label = QLabel("Status: Disconnected")
        conn_layout.addWidget(self.status_label, 0, 0, 1, 2)
        
        # API Key input
        conn_layout.addWidget(QLabel("API Key:"), 1, 0)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Get your free API key from aisstream.io/authenticate")
        self.api_key_edit.setToolTip("Register at https://aisstream.io/authenticate to get your free API key")
        conn_layout.addWidget(self.api_key_edit, 1, 1)
        
        # Info label
        info_label = QLabel("Note: Register at aisstream.io/authenticate for free API access")
        info_label.setStyleSheet("color: blue; font-size: 10px;")
        conn_layout.addWidget(info_label, 2, 0, 1, 2)
        
        self.connect_btn = QPushButton("Connect to AISStream.io")
        self.connect_btn.clicked.connect(self.toggle_realtime_connection)
        conn_layout.addWidget(self.connect_btn, 3, 0)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_realtime)
        self.disconnect_btn.setEnabled(False)
        conn_layout.addWidget(self.disconnect_btn, 3, 1)
        
        layout.addWidget(conn_group)
        
        # Data display controls
        display_group = QGroupBox("Data Display")
        display_layout = QVBoxLayout(display_group)
        
        controls_layout = QHBoxLayout()
        
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        controls_layout.addWidget(self.auto_scroll_cb)
        
        self.max_messages_spin = QSpinBox()
        self.max_messages_spin.setRange(10, 10000)
        self.max_messages_spin.setValue(1000)
        controls_layout.addWidget(QLabel("Max messages:"))
        controls_layout.addWidget(self.max_messages_spin)
        
        self.clear_btn = QPushButton("Clear Display")
        self.clear_btn.clicked.connect(self.clear_realtime_display)
        controls_layout.addWidget(self.clear_btn)
        
        controls_layout.addStretch()
        display_layout.addLayout(controls_layout)
        
        # Real-time data display
        self.realtime_display = QTextEdit()
        self.realtime_display.setFont(QFont("Courier", 9))
        self.realtime_display.setReadOnly(True)
        display_layout.addWidget(self.realtime_display)
        
        layout.addWidget(display_group)
        
        # Export controls
        export_layout = QHBoxLayout()
        self.export_realtime_btn = QPushButton("Export Real-time Data")
        self.export_realtime_btn.clicked.connect(self.export_realtime_data)
        export_layout.addWidget(self.export_realtime_btn)
        export_layout.addStretch()
        
        layout.addLayout(export_layout)
        
        parent.addTab(realtime_widget, "Real-time Data")
        
    def create_historical_tab(self, parent):
        """Create historical data retrieval tab"""
        historical_widget = QWidget()
        layout = QVBoxLayout(historical_widget)
        
        # Source selection
        source_group = QGroupBox("Data Source")
        source_layout = QGridLayout(source_group)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["AISHub", "MarineTraffic", "Global Fishing Watch"])
        source_layout.addWidget(QLabel("Source:"), 0, 0)
        source_layout.addWidget(self.source_combo, 0, 1)
        
        layout.addWidget(source_group)
        
        # Parameters
        params_group = QGroupBox("Query Parameters")
        params_layout = QGridLayout(params_group)
        
        # Geographic bounds
        params_layout.addWidget(QLabel("Latitude Min:"), 0, 0)
        self.lat_min_edit = QLineEdit("-90")
        params_layout.addWidget(self.lat_min_edit, 0, 1)
        
        params_layout.addWidget(QLabel("Latitude Max:"), 0, 2)
        self.lat_max_edit = QLineEdit("90")
        params_layout.addWidget(self.lat_max_edit, 0, 3)
        
        params_layout.addWidget(QLabel("Longitude Min:"), 1, 0)
        self.lon_min_edit = QLineEdit("-180")
        params_layout.addWidget(self.lon_min_edit, 1, 1)
        
        params_layout.addWidget(QLabel("Longitude Max:"), 1, 2)
        self.lon_max_edit = QLineEdit("180")
        params_layout.addWidget(self.lon_max_edit, 1, 3)
        
        # Time range
        params_layout.addWidget(QLabel("Start Time:"), 2, 0)
        self.start_time_edit = QDateTimeEdit(QDateTime.currentDateTime().addDays(-1))
        params_layout.addWidget(self.start_time_edit, 2, 1)
        
        params_layout.addWidget(QLabel("End Time:"), 2, 2)
        self.end_time_edit = QDateTimeEdit(QDateTime.currentDateTime())
        params_layout.addWidget(self.end_time_edit, 2, 3)
        
        # Additional parameters
        params_layout.addWidget(QLabel("Username (AISHub):"), 3, 0)
        self.username_edit = QLineEdit("guest")
        params_layout.addWidget(self.username_edit, 3, 1)
        
        layout.addWidget(params_group)
        
        # Fetch controls
        fetch_layout = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch Historical Data")
        self.fetch_btn.clicked.connect(self.fetch_historical_data)
        fetch_layout.addWidget(self.fetch_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        fetch_layout.addWidget(self.progress_bar)
        
        layout.addLayout(fetch_layout)
        
        # Historical data display
        self.historical_display = QTextEdit()
        self.historical_display.setFont(QFont("Courier", 9))
        self.historical_display.setReadOnly(True)
        layout.addWidget(self.historical_display)
        
        parent.addTab(historical_widget, "Historical Data")
        
    def create_data_viewer_tab(self, parent):
        """Create data viewer and export tab"""
        viewer_widget = QWidget()
        layout = QVBoxLayout(viewer_widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.refresh_table_btn = QPushButton("Refresh Table")
        self.refresh_table_btn.clicked.connect(self.refresh_data_table)
        controls_layout.addWidget(self.refresh_table_btn)

        self.decode_btn = QPushButton("Decode & Show DataFrame")
        self.decode_btn.clicked.connect(self.show_decoded_dataframe)
        controls_layout.addWidget(self.decode_btn)
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        controls_layout.addWidget(self.export_csv_btn)
        
        self.export_json_btn = QPushButton("Export to JSON")
        self.export_json_btn.clicked.connect(self.export_to_json)
        controls_layout.addWidget(self.export_json_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Data table
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        
        parent.addTab(viewer_widget, "Data Viewer")


    def decode_realtime_messages(self):
        """
        Decode AIS real-time messages into a pandas DataFrame
        with columns: mmsi, shipname, lat, lon, sog, cog, heading, status, timestamp.
        """
        decoded_rows = []
        for msg in self.received_messages:
            try:
                message_type = msg.get("MessageType", "")
                ais_msg = msg.get("Message", {})
                meta = msg.get("MetaData", {})

                # Default row from MetaData
                row = {
                    "mmsi": meta.get("MMSI"),
                    "shipname": meta.get("ShipName"),
                    "lat": meta.get("latitude"),
                    "lon": meta.get("longitude"),
                    "sog": None,
                    "cog": None,
                    "heading": None,
                    "status": None,
                    "timestamp": meta.get("time_utc")
                }

                # Handle message types
                if message_type == "StandardClassBPositionReport":
                    rep = ais_msg.get("StandardClassBPositionReport", {})
                    row.update({
                        "mmsi": rep.get("UserID", row["mmsi"]),
                        "lat": rep.get("Latitude", row["lat"]),
                        "lon": rep.get("Longitude", row["lon"]),
                        "sog": rep.get("Sog", row["sog"]),
                        "cog": rep.get("Cog", row["cog"]),
                        "heading": rep.get("TrueHeading", row["heading"]),
                        "status": rep.get("NavigationalStatus", row["status"])
                    })
                elif message_type == "PositionReportClassA":
                    rep = ais_msg.get("PositionReportClassA", {})
                    row.update({
                        "mmsi": rep.get("UserID", row["mmsi"]),
                        "lat": rep.get("Latitude", row["lat"]),
                        "lon": rep.get("Longitude", row["lon"]),
                        "sog": rep.get("Sog", row["sog"]),
                        "cog": rep.get("Cog", row["cog"]),
                        "heading": rep.get("TrueHeading", row["heading"]),
                        "status": rep.get("NavigationalStatus", row["status"])
                    })
                elif message_type == "PositionReport":
                    rep = ais_msg.get("PositionReport", {})
                    row.update({
                        "mmsi": rep.get("UserID", row["mmsi"]),
                        "lat": rep.get("Latitude", row["lat"]),
                        "lon": rep.get("Longitude", row["lon"]),
                        "sog": rep.get("Sog", row["sog"]),
                        "cog": rep.get("Cog", row["cog"]),
                        "heading": rep.get("TrueHeading", row["heading"]),
                        "status": rep.get("NavigationalStatus", row["status"])
                    })

                # Important: keep 0 values (don’t drop them)
                for k in ["sog", "cog", "heading", "status"]:
                    if row[k] is None and meta.get(k) is not None:
                        row[k] = meta.get(k)

                decoded_rows.append(row)

            except Exception as e:
                print(f"Decode error: {e}, raw={msg}")

        if decoded_rows:
            return pd.DataFrame(decoded_rows)
        else:
            return pd.DataFrame(
                columns=["mmsi", "shipname", "lat", "lon", "sog", "cog", "heading", "status", "timestamp"]
            )

    def show_decoded_dataframe(self):
        """Display decoded messages in the QTableWidget."""
        df = self.decode_realtime_messages()
        if df.empty:
            QMessageBox.information(self, "Info", "No decoded AIS messages yet")
            return

        # Update QTableWidget
        self.data_table.setColumnCount(len(df.columns))
        self.data_table.setRowCount(len(df))
        self.data_table.setHorizontalHeaderLabels(df.columns.tolist())

        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                self.data_table.setItem(i, j, QTableWidgetItem(str(row[col])))

        self.data_table.resizeColumnsToContents()

    def toggle_realtime_connection(self):
        """Toggle real-time connection"""
        if self.realtime_thread is None or not self.realtime_thread.isRunning():
            self.connect_realtime()
        else:
            self.disconnect_realtime()
            
    def connect_realtime(self):
        """Connect to real-time AIS stream"""
        api_key = self.api_key_edit.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "Warning", 
                              "Please enter your AISStream.io API key.\n\n" +
                              "Register for free at: https://aisstream.io/authenticate")
            return
        
        self.realtime_thread = AISStreamThread()
        
        # Set the API key from the input field
        self.realtime_thread.api_key = api_key
        
        self.realtime_thread.data_received.connect(self.handle_realtime_data)
        self.realtime_thread.error_occurred.connect(self.handle_realtime_error)
        self.realtime_thread.connection_status.connect(self.update_connection_status)
        
        self.realtime_thread.start()
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.api_key_edit.setEnabled(False)
        
    def disconnect_realtime(self):
        """Disconnect from real-time stream"""
        if self.realtime_thread and self.realtime_thread.isRunning():
            self.realtime_thread.stop()
            self.realtime_thread.wait(5000)  # Wait up to 5 seconds
            
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.api_key_edit.setEnabled(True)
        self.status_label.setText("Status: Disconnected")
        
    def handle_realtime_data(self, data):
        """Handle incoming real-time data"""
        self.received_messages.append(data)
        
        # Limit stored messages
        max_messages = self.max_messages_spin.value()
        if len(self.received_messages) > max_messages:
            self.received_messages = self.received_messages[-max_messages:]
            
        # Display in text area
        timestamp = datetime.now().strftime("%H:%M:%S")
        display_text = f"[{timestamp}] {json.dumps(data, indent=2)}\n"
        
        self.realtime_display.append(display_text)
        
        if self.auto_scroll_cb.isChecked():
            cursor = self.realtime_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.realtime_display.setTextCursor(cursor)
            
    def handle_realtime_error(self, error_msg):
        """Handle real-time connection errors"""
        self.realtime_display.append(f"ERROR: {error_msg}\n")
        self.statusBar().showMessage(f"Real-time error: {error_msg}")
        
    def update_connection_status(self, status):
        """Update connection status"""
        self.status_label.setText(f"Status: {status}")
        self.statusBar().showMessage(status)
        
    def clear_realtime_display(self):
        """Clear real-time display"""
        self.realtime_display.clear()
        self.received_messages.clear()
        
    def fetch_historical_data(self):
        """Fetch historical AIS data"""
        source = self.source_combo.currentText()
        
        params = {
            'lat_min': float(self.lat_min_edit.text()),
            'lat_max': float(self.lat_max_edit.text()),
            'lon_min': float(self.lon_min_edit.text()),
            'lon_max': float(self.lon_max_edit.text()),
            'start_time': self.start_time_edit.dateTime().toPyDateTime(),
            'end_time': self.end_time_edit.dateTime().toPyDateTime(),
            'username': self.username_edit.text()
        }
        
        self.historical_thread = HistoricalDataThread(source, params)
        self.historical_thread.data_received.connect(self.handle_historical_data)
        self.historical_thread.error_occurred.connect(self.handle_historical_error)
        self.historical_thread.progress_update.connect(self.update_progress)
        
        self.progress_bar.setVisible(True)
        self.fetch_btn.setEnabled(False)
        self.historical_thread.start()
        
    def handle_historical_data(self, data):
        """Handle historical data response"""
        self.historical_display.clear()
        self.historical_display.append(json.dumps(data, indent=2))
        self.progress_bar.setVisible(False)
        self.fetch_btn.setEnabled(True)
        self.statusBar().showMessage(f"Received {len(data)} historical records")
        
    def handle_historical_error(self, error_msg):
        """Handle historical data errors"""
        self.historical_display.append(f"ERROR: {error_msg}")
        self.progress_bar.setVisible(False)
        self.fetch_btn.setEnabled(True)
        self.statusBar().showMessage(f"Historical data error: {error_msg}")
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def refresh_data_table(self):
        """Refresh the data table with current messages"""
        if not self.received_messages:
            return
            
        # Extract unique columns from all messages
        columns = set()
        for msg in self.received_messages:
            if isinstance(msg, dict):
                columns.update(msg.keys())
        
        columns = sorted(list(columns))
        
        # Set up table
        self.data_table.setColumnCount(len(columns))
        self.data_table.setRowCount(len(self.received_messages))
        self.data_table.setHorizontalHeaderLabels(columns)
        
        # Populate table
        for row, msg in enumerate(self.received_messages):
            for col, column_name in enumerate(columns):
                value = msg.get(column_name, '') if isinstance(msg, dict) else str(msg)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                self.data_table.setItem(row, col, QTableWidgetItem(str(value)))
                
        self.data_table.resizeColumnsToContents()
        
    def export_realtime_data(self):
        """Export real-time data"""
        if not self.received_messages:
            QMessageBox.warning(self, "Warning", "No data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export Real-time Data", 
                                                "ais_realtime.json", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.received_messages, f, indent=2, default=str)
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
                
    def export_to_csv(self):
        """Export data to CSV"""
        if not self.received_messages:
            QMessageBox.warning(self, "Warning", "No data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export to CSV", 
                                                "ais_data.csv", "CSV Files (*.csv)")
        if filename:
            try:
                # Flatten data for CSV
                flattened_data = []
                for msg in self.received_messages:
                    if isinstance(msg, dict):
                        flattened_data.append(msg)
                    else:
                        flattened_data.append({'data': str(msg)})
                
                if flattened_data:
                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                        writer.writeheader()
                        writer.writerows(flattened_data)
                    
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"CSV export failed: {str(e)}")
                
    def export_to_json(self):
        """Export data to JSON"""
        self.export_realtime_data()  # Same functionality
        
    def closeEvent(self, event):
        """Handle application close event"""
        if self.realtime_thread and self.realtime_thread.isRunning():
            self.disconnect_realtime()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = AISDataRetriever()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()