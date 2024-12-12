import gc

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

from scipy import ndimage as ndi
from shutil import copyfile
from skimage import exposure
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QGridLayout, QHBoxLayout, QSpinBox
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QSpinBox, QHBoxLayout, QSlider
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF
from obspy.io.segy.segy import SEGYFile
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import butter, filtfilt
import sys
import struct
from matplotlib.lines import Line2D
import matplotlib.patches as patches


from scipy.signal import butter, lfilter
from PyQt5.QtWidgets import QLineEdit
from scipy.ndimage import gaussian_filter1d
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QScrollBar
from PyQt5.QtWidgets import QMessageBox
from scipy.signal import iirnotch, medfilt

import psutil

class GprParser():
    '''
    class to store GPR data read from file
    '''
    def __init__(self, file_path : str):
        self.segy_stream = None
        try:
            self.segy_stream = read(file_path, format='SEGY')
        except Exception as e:
            print(f"Error opening SEGY file with ObsPy: {e}")

        self.cutoff_freqs = [50]  # Default cutoff frequency
        self.order = 5  # Default filter order
        
        self.traces_num = len(self.segy_stream)
        self.samples_num = len(self.segy_stream[0].data)
        #   2D numpy array to hold all trace data
        self.seismic_data = np.zeros((self.samples_num, self.traces_num))
        #   2D numpy array to hold X - Y coordinates per trace. indices correspond to traces in self.seismic_data
        self.trace_coordinates = np.zeros((2, self.traces_num), dtype=float)
        # Fill the array with trace data and coordinates
        for i, trace in enumerate(self.segy_stream):
            self.seismic_data[:, i] = trace.data  # Assign each trace's data to a column in the 2D array
            #   coordinates are extracted wrongly by library, so have to do it manually
            bin_header = trace.stats.segy['trace_header']['unpacked_header']
            lonX = struct.unpack('<f', bin_header[72:76])[0]
            latY = struct.unpack('<f', bin_header[76:80])[0]
            self.trace_coordinates[:, i] = (lonX, latY)
            #   DEBUG
            #print(f'trace #{i}:\nsample interval: {trace.stats.delta}')
        
        # Create the time axis (assuming uniform sample interval)
        self.sample_interval = self.segy_stream[0].stats.delta  # sample interval in seconds
        self.fs = self.segy_stream[0].stats.sampling_rate       # samoke freq in Hz

        self.time_axis = np.arange(0, self.samples_num * self.sample_interval, self.sample_interval)

        # init filter instance
        self.filter = self.GprFilter(self)

    #   TODO
    # TODO: be able to apply filters on top of each other + reset to original data or go back button
    # TODO: configuration file for default filter settings
    class GprFilter:
        '''
        inner class for filtering data
        '''
        def __init__(self, parent_parser):
            self.parent = parent_parser
            self.data = None

        # Filtering functions
        def butter_lowpass(self, cutoff, fs, order=5):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
            return b, a
        
        def butter_highpass(self, cutoff, fs, order=5):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
            return b, a

#TODO fix bandpass
        def butter_bandpass(self, lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist

            # Ensure low < high
            if low >= high:
                raise ValueError(f"Low cutoff ({lowcut} Hz) must be less than high cutoff ({highcut} Hz)")

            b, a = butter(order, [low, high], btype='bandpass')
            return b, a
        
        def moving_average(self, data, window_size):
            if len(data) < window_size:
                raise ValueError("Window size must be less than or equal to the length of the data.")
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        def notch_filter(self, freq, fs):
            nyquist = 0.5 * fs
            notch_freq = freq / nyquist
            if not (0 < notch_freq < 1):
                raise ValueError(f"Invalid notch frequency: {notch_freq}. Must be between 0 and 1.")
            b, a = iirnotch(notch_freq, Q=30)
            return b, a

        def median_filter(self, data, kernel_size):
            return medfilt(data, kernel_size)

        def apply_notch_filter(self, data, freq, fs):
            # Ensure data is 2D
            if data.ndim != 2:
                raise ValueError("Data must be a 2D array.")

            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                data = np.nan_to_num(data)

            gc.collect()
            print(f"Memory usage before filter: {psutil.Process().memory_info().rss / (1024 * 1024)} MB")

            b, a = self.notch_filter(freq, fs)

            filtered_data = np.zeros_like(data)

            # Apply filter to each row
            for i in range(data.shape[0]):
                try:
                    filtered_data[i, :] = lfilter(b, a, data[i, :])
                except Exception as e:
                    print(f"Error applying lfilter to row {i}: {e}")
                    raise

            # Apply filtfilt for each chunk if necessary
            chunk_size = 100  # Adjust as needed
            for start in range(0, data.shape[1], chunk_size):
                end = min(start + chunk_size, data.shape[1])
                try:
                    for i in range(data.shape[0]):  # Apply filtfilt row-by-row
                        filtered_data[i, start:end] = filtfilt(b, a, data[i, start:end], axis=0)
                except Exception as e:
                    print(f"Error processing chunk {start}-{end}: {e}")
                    raise

            return filtered_data

        def apply_filter(self, data, filter_type, cutoff_freqs, fs, order=5):
            # Add error handling and input validation
            if not isinstance(data, np.ndarray):
                raise ValueError("Input data must be a NumPy array")

            if data.size == 0:
                raise ValueError("Input data array is empty")

            try:
                if filter_type == 'lowpass':
                    b, a = self.butter_lowpass(cutoff_freqs[0], fs, order=order)
                elif filter_type == 'highpass':
                    b, a = self.butter_highpass(cutoff_freqs[0], fs, order=order)
                elif filter_type == 'bandpass':
                    # Ensure two cutoff frequencies are provided
                    if len(cutoff_freqs) != 2:
                        raise ValueError("Bandpass filter requires two cutoff frequencies: [lowcut, highcut]")
                    lowcut, highcut = cutoff_freqs
                    b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
                elif filter_type == 'moving_average':
                    return self.moving_average(data, order)  # window size is `order`
                elif filter_type == 'gaussian':
                    sigma = order  # Use order as sigma for Gaussian
                    return gaussian_filter1d(data, sigma=sigma, mode='nearest')
                elif filter_type == 'notch':
                    return self.apply_notch_filter(data, cutoff_freqs[0], fs)
                elif filter_type == 'median':
                    kernel_size = order  # Use `order` as kernel size
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # Ensure odd kernel size
                    return self.median_filter(data, kernel_size)
                else:
                    raise ValueError("Unknown filter type")

                # Handle potential NaN or Inf values
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    data = np.nan_to_num(data)

                # Apply the filter
                filtered_data = filtfilt(b, a, data)

                # Ensure output matches input length
                if len(filtered_data) > len(data):
                    filtered_data = filtered_data[:len(data)]
                elif len(filtered_data) < len(data):
                    filtered_data = np.pad(filtered_data, (0, len(data) - len(filtered_data)), mode='constant')

                return filtered_data

            except Exception as e:
                print(f"Error applying {filter_type} filter: {e}")
                return data  # Return original data if filter fails


class MplCanvas(FigureCanvasQTAgg):
    '''
    separate obj for plot widget
    '''
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        #   vertical line to show up when clicked on trace
        self.vertical_line = None
        #   connect event handlers
        fig.canvas.mpl_connect('button_press_event', self._on_left_click)
        super(MplCanvas, self).__init__(fig)

    #   click event handler (show vertical line)
    def _on_left_click(self, event):
        #   ignore if clicked outside graph
        if event.inaxes != self.axes:
            return
        #   ignore if no file parsed at moment
        if self.parent.parser == None:
            return
        #   show only integer traces num
        trace_number = int(round(event.xdata))
        #   ignore if outside of range
        if trace_number < 0 or trace_number >= self.parent.parser.traces_num:
            return
        #   remove prev line
        if self.vertical_line:
            self.vertical_line.remove()
        #   show location on GPS widget
        lonX, latY = self.parent.parser.trace_coordinates[:, trace_number]
        self.parent.mpl_gps_canvas.show_current_location(lonX, latY)

        
        self.vertical_line = self.axes.axvline(trace_number, color='r', linestyle='-')
        self.draw()
        self.parent.mpl_toolbar.set_message(f'Trace#{trace_number} Time[ns]: {trace_number}')

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Zoom in or out based on the wheel event
        scale_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        x_range = (xlim[1] - xlim[0]) * scale_factor
        y_range = (ylim[1] - ylim[0]) * scale_factor

        self.axes.set_xlim([xlim[0], xlim[0] + x_range])
        self.axes.set_ylim([ylim[0], ylim[0] + y_range])
        self.draw()
    
    def parent(self):
        return self.parent

class MplGpsCanvas(FigureCanvasQTAgg):
    '''
    coordinates widget
    '''
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplGpsCanvas, self).__init__(fig)

    def plot_coordinates(self, lonX, latY):
        self.axes.clear()
        self.axes.scatter(lonX, latY, c='red', marker='+')
        self.axes.set_xlabel('Longtitude X')
        self.axes.set_ylabel('Latitude Y')
        self.axes.set_title('GPS coordinates')
        self.axes.set_xlim([lonX.min(), lonX.max()])
        self.axes.set_ylim([latY.min(), latY.max()])
        self.axes.set_aspect('equal')

        #   force full numbers on axes
        self.axes.yaxis.get_major_formatter().set_scientific(False)
        self.axes.yaxis.get_major_formatter().set_useOffset(False)
        self.axes.xaxis.get_major_formatter().set_scientific(False)
        self.axes.xaxis.get_major_formatter().set_useOffset(False)

        self.draw()

    def show_current_location(self, lonX, latY):
        '''
        should show location of current trace (when clicked) by two lines
        '''
        for line in self.axes.lines:
            line.remove()
        self.axes.axvline(x=lonX, color='blue', linestyle='--', linewidth=1)
        self.axes.axhline(y=latY, color='blue', linestyle='--', linewidth=1)
        self.draw()

class GainWidget(FigureCanvasQTAgg):
    '''
    separate obj for gain widget
    '''
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.parent = parent
        self.dragging_point = None

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(GainWidget, self).__init__(fig)
        self.axes.set_xlabel('Gain [dB]')
        self.axes.set_ylabel('Time [ns]')
        self.axes.yaxis.label.set_rotation(90)
        self.axes.set_title('Gain Function')
        secax = self.axes.secondary_xaxis('top')
        secax.set_xlabel('Attenuation[dB]')
        #   change axis direction
        self.axes.invert_yaxis()
        #self.axes.set_ylim(0)
        
        #   initial positions
        self.point_positions = [[0, 0], [0, 0]]

        # Add points to the plot
        self.points = [
            self.axes.plot(x, y, 'ro', picker=5)[0]  # Red points with pick radius of 5
            for x, y in self.point_positions
        ]
    
    # TODO
    def plot_data(self):
        '''
        refreshes the widget, should be called when new .sgy file is opened
        '''
        self.axes.clear()

class MainWindow(QMainWindow):
    '''
    Main window
    '''

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()
        self.color_scheme = 'seismic'
        self.zoom_factor = 1.1  # Factor for zooming in and out
        # self.cutoff_freqs = [50]  # Default cutoff frequency
        # self.order = 5  # Default filter order
        # self.cutoff_input_high = 50  # Default high cutoff frequency
        # self.cutoff_input_low = 30  # Default low cutoff frequency

    def init_ui(self):

        self.setWindowTitle('GPR plotter')
        self.setGeometry(0, 0, 1920, 1080)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        #   one main layout
        layout = QGridLayout(self)

        #   button to open file
        self.file_btn = QPushButton("Open .sgy file")
        self.file_btn.clicked.connect(self.file_btn_clicked)

        #   button to apply filter
        self.filter_btn = QPushButton("Apply filter")
        self.filter_btn.clicked.connect(self.filter_btn_clicked)

        #   dropdown menu to select filter
        self.filter_box = QComboBox()
        self.filter_box.addItems(['None', 'lowpass', 'highpass', 'bandpass', 'moving_average', 'gaussian', 'notch', 'median'])
        self.filter_box.currentTextChanged.connect(self.on_filter_change)

        # Color scheme selection
        self.color_scheme_box = QComboBox()
        self.color_scheme_box.addItems(['seismic', 'viridis', 'plasma', 'inferno', 'gray', 'magma', 'cividis'])
        self.color_scheme_box.currentTextChanged.connect(self.update_color_scheme)

        # Zoom in/out buttons
        self.zoom_controls_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_controls_layout.addWidget(self.zoom_in_btn)
        self.zoom_controls_layout.addWidget(self.zoom_out_btn)

        # Matplotlib FigureCanvas and toolbar for zoom/pan
        self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar2QT(self.mpl_canvas, self)


        # Input fields for cutoff frequencies
        self.cutoff_input_low = QLineEdit()
        self.cutoff_input_low.setPlaceholderText("Cutoff Frequency (Low)")
        self.cutoff_input_low.setText("50")  # Default value

        self.cutoff_input_high = QLineEdit()
        self.cutoff_input_high.setPlaceholderText("Cutoff Frequency (High)")
        self.cutoff_input_high.setVisible(False)  # Hide initially

        # Order spinbox
        self.order_spinbox = QSpinBox()
        self.order_spinbox.setRange(1, 10)  # Adjust as needed
        self.order_spinbox.setValue(5)
        self.order_spinbox.valueChanged.connect(self.update_order)

        # Filter control layout
        cutoff_label_low = QLabel("Cutoff Frequency (Low):")
        cutoff_label_high = QLabel("Cutoff Frequency (High):")
        order_label = QLabel("Filter Order:")

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Low Cutoff:"))
        controls_layout.addWidget(QLabel("High Cutoff:"))
        controls_layout.addWidget(self.cutoff_input_high)
        controls_layout.addWidget(self.cutoff_input_low)
        controls_layout.addWidget(QLabel("Order:"))
        controls_layout.addWidget(self.order_spinbox)

        # Add Reset Zoom Button
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)

        # Wrap the canvas in a scroll area
        # self.scroll_area = QScrollArea()
        # self.scroll_area.setWidget(self.mpl_canvas)
        # self.scroll_area.setWidgetResizable(True)

        #   matplotlib FigureCanvas obj for GPS coordinates plot
        self.mpl_gps_canvas = MplGpsCanvas(self, width=5, height=4)

        #   gain function
        self.gain_widget = GainWidget(self, width=4)

        # Add all widgets to the layout
        # layout.addWidget(self.scroll_area, 0, 0)  # Use scroll area for the canvas
        layout.addWidget(self.mpl_canvas, 0, 0)  # Directly add the canvas
        layout.addLayout(controls_layout, 1, 0)
        layout.addWidget(self.filter_box, 2, 0)
        layout.addWidget(self.filter_btn, 3, 0)
        layout.addWidget(self.toolbar, 4, 0)
        layout.addWidget(self.color_scheme_box, 5, 0)
        layout.addLayout(self.zoom_controls_layout, 6, 0)
        layout.addWidget(self.reset_zoom_btn, 7, 0)
        layout.addWidget(self.file_btn, 8, 0)
        layout.addWidget(self.mpl_gps_canvas, 0, 1)
        layout.addWidget(self.gain_widget, 1, 1, 6, 1, Qt.AlignRight)

        self.central_widget.setLayout(layout)

    def reset_zoom(self):
        """Reset the zoom level to the original limits."""
        if hasattr(self.mpl_canvas, 'original_xlim') and hasattr(self.mpl_canvas, 'original_ylim'):
            if self.mpl_canvas.original_xlim and self.mpl_canvas.original_ylim:
                self.mpl_canvas.axes.set_xlim(self.mpl_canvas.original_xlim)
                self.mpl_canvas.axes.set_ylim(self.mpl_canvas.original_ylim)
                self.mpl_canvas.draw()
            else:
                QMessageBox.warning(self, "Reset Zoom", "No zoom level to reset.")
        else:
            QMessageBox.warning(self, "Reset Zoom", "Original limits not set.")

    def zoom_in(self):
        # Increase the size of the canvas when zooming in
        self.mpl_canvas.setFixedSize(self.mpl_canvas.width() * 1.1, self.mpl_canvas.height() * 1.1)
        self.mpl_canvas.axes.set_xlim(self.mpl_canvas.axes.get_xlim()[0] * 0.9,
                                      self.mpl_canvas.axes.get_xlim()[1] * 1.1)
        self.mpl_canvas.axes.set_ylim(self.mpl_canvas.axes.get_ylim()[0] * 0.9,
                                      self.mpl_canvas.axes.get_ylim()[1] * 1.1)
        self.mpl_canvas.draw()

    def zoom_out(self):
        # Decrease the size of the canvas when zooming out
        self.mpl_canvas.setFixedSize(self.mpl_canvas.width() / 1.1, self.mpl_canvas.height() / 1.1)
        self.mpl_canvas.axes.set_xlim(self.mpl_canvas.axes.get_xlim()[0] / 1.1,
                                      self.mpl_canvas.axes.get_xlim()[1] / 0.9)
        self.mpl_canvas.axes.set_ylim(self.mpl_canvas.axes.get_ylim()[0] / 1.1,
                                      self.mpl_canvas.axes.get_ylim()[1] / 0.9)
        self.mpl_canvas.draw()

    # def update_view(self):
    #     # Update plot limits based on current x and y limits
    #     self.mpl_canvas.axes.set_xlim(self.current_xlim)
    #     self.mpl_canvas.axes.set_ylim(self.current_ylim)
    #     self.mpl_canvas.draw()
    #
    #     # Adjust scroll bar ranges and positions based on current view
    #     self.h_scroll.setRange(int(self.initial_xlim[0]),
    #                            int(self.initial_xlim[1] - (self.current_xlim[1] - self.current_xlim[0])))
    #     self.v_scroll.setRange(int(self.initial_ylim[0]),
    #                            int(self.initial_ylim[1] - (self.current_ylim[1] - self.current_ylim[0])))
    #     self.h_scroll.setPageStep(int(self.current_xlim[1] - self.current_xlim[0]))
    #     self.v_scroll.setPageStep(int(self.current_ylim[1] - self.current_ylim[0]))
    #
    # def update_xlim(self, value):
    #     # Update the x-axis limits based on the scroll bar position
    #     new_xlim = (value, value + (self.current_xlim[1] - self.current_xlim[0]))
    #     self.current_xlim = new_xlim
    #     self.update_view()
    #
    # def update_ylim(self, value):
    #     # Update the y-axis limits based on the scroll bar position
    #     new_ylim = (value, value + (self.current_ylim[1] - self.current_ylim[0]))
    #     self.current_ylim = new_ylim
    #     self.update_view()
    def update_color_scheme(self, scheme):
        self.color_scheme = scheme
        if self.parser.seismic_data is not None:
            self.plot_radargram(self.parser.seismic_data, "Radargram with Color Scheme")

    def on_filter_change(self, text):
        # Show/hide the high cutoff input based on the selected filter
        if text == 'bandpass':
            self.cutoff_input_high.setVisible(True)
        else:
            self.cutoff_input_high.setVisible(False)  # button handlers

    def file_btn_clicked(self):
        '''
        Open .sgy file and plot
        '''
        print('file_btn_clicked event triggered!')
        self.file_path = QFileDialog.getOpenFileName(self, 'Open file', '', '*.sgy')[0]
        if (len(self.file_path) == 0):
            return
        print(f'opened file {self.file_path}')
        
        # init Gpr parser
        self.parser = GprParser(self.file_path)
        self.plot_radargram(self.parser.seismic_data, "Radargram (Seismic Section)")
        # plot the radargram
        # self.mpl_canvas.axes.imshow(self.parser.seismic_data, aspect='auto', cmap='seismic', extent=[0, self.parser.traces_num, self.parser.time_axis[-1], self.parser.time_axis[0]])
        # #self.mpl_canvas.axes.colorbar(label="Amplitude")
        # self.mpl_canvas.axes.set_title("Radargram (Seismic Section)")
        # self.mpl_canvas.axes.set_xlabel("Trace number")
        # self.mpl_canvas.axes.set_ylabel("Time (ns)")
        # #   refresh canvas
        # self.mpl_canvas.draw()

        # plot GPS coordinates
        lonX, latY = self.parser.trace_coordinates
        self.mpl_gps_canvas.plot_coordinates(lonX, latY)

    def plot_radargram(self, data, title):
        '''
        Helper function to plot radargram on the canvas
        '''
        self.mpl_canvas.axes.clear()
        self.mpl_canvas.axes.imshow(data, aspect='auto', cmap=self.color_scheme)
        self.mpl_canvas.axes.set_title(title)
        self.mpl_canvas.axes.set_xlabel("Trace number")
        self.mpl_canvas.axes.set_ylabel("Time (s)")

        # Store the original limits for the reset function
        self.mpl_canvas.original_xlim = self.mpl_canvas.axes.get_xlim()
        self.mpl_canvas.original_ylim = self.mpl_canvas.axes.get_ylim()

        print(f"Original limits set: xlim={self.mpl_canvas.original_xlim}, ylim={self.mpl_canvas.original_ylim}")

        self.mpl_canvas.draw()

    # def update_cutoff(self):
    #     self.cutoff_freqs = [self.cutoff_slider.value()]

    def update_order(self):
        if self.parser != None:     #   ignore if parser not initialized
            self.parser.order = self.order_spinbox.value()

    def filter_btn_clicked(self):
        '''
        Apply the selected filter to the data and re-plot
        '''

        try:
            if self.parser is None:
                msg1 = QMessageBox()
                msg1.setWindowTitle("Error")
                msg1.setText("No data loaded to apply filter!")
                msg1.setIcon(QMessageBox.Critical)
                msg1.setStandardButtons(QMessageBox.Ok)
                msg1.exec_()
                return
            else:
                print(f"Data ready for filtering. Shape: {self.parser.seismic_data.shape}, fs: {self.parser.fs}")

            filter_type = self.filter_box.currentText().lower()

            # Get cutoff frequencies from input boxes

            low = self.cutoff_input_low.text()
            high = self.cutoff_input_high.text()

            print(f"Filter type: {filter_type}, low: {low}, high: {high}")

            if not low:
                raise ValueError("Please set lowcut frequency")

            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please set both lowcut and highcut\nfrequencies for the bandpass filter.")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)

            if low == '':
                msg.exec_()
                return
            if high == '' and filter_type == 'bandpass':
                msg.exec_()
                return

            low_cutoff = float(self.cutoff_input_low.text())
            high_cutoff = float(self.cutoff_input_high.text()) if filter_type == 'bandpass' else None

            if filter_type == 'none':
                print("No filter selected")
                self.plot_radargram(self.parser.seismic_data, "Radargram (Original Data)")
                return

                # Validate cutoff frequencies
            if low_cutoff <= 0 or (high_cutoff is not None and high_cutoff <= 0):
                raise ValueError("Cutoff frequencies must be positive")

                # Validate against Nyquist frequency
            nyquist = 0.5 * self.parser.fs
            if low_cutoff > nyquist or (high_cutoff and high_cutoff > nyquist):
                raise ValueError(f"Cutoff frequency must be less than Nyquist frequency ({nyquist} Hz)")

            if filter_type == 'bandpass':
                # Validate both low and high cutoff frequencies
                if not low or not high:
                    raise ValueError("Both low and high cutoff frequencies are required for bandpass filter")

                low_cutoff = float(low)
                high_cutoff = float(high)

                # Ensure low cutoff is less than high cutoff
                if low_cutoff >= high_cutoff:
                    raise ValueError("Low cutoff frequency must be less than high cutoff frequency")

                cutoff_freqs = [low_cutoff, high_cutoff]
            else:
                cutoff_freqs = [low_cutoff]

            filtered_seismic_data = np.zeros_like(self.parser.seismic_data)
            for i in range(self.parser.seismic_data.shape[1]):
                try:
                    filtered_seismic_data[:, i] = self.parser.filter.apply_filter(
                        self.parser.seismic_data[:, i],
                        filter_type,
                        cutoff_freqs,
                        self.parser.fs,
                        self.parser.order
                    )
                except Exception as trace_filter_error:
                    print(f"Error filtering trace {i}: {trace_filter_error}")
                    # Use original trace if filtering fails
                    filtered_seismic_data[:, i] = self.parser.seismic_data[:, i]

            # # Apply the filter
            # if filter_type == 'moving_average':
            #     window_size = self.parser.order  # Use order as the window size for moving average
            #     filtered_seismic_data = np.apply_along_axis(
            #         lambda trace: self.parser.filter.moving_average(trace, window_size),
            #         axis=0, arr=self.parser.seismic_data
            #     )
            # else:
            #     filtered_seismic_data = np.apply_along_axis(
            #         lambda trace: self.parser.filter.apply_filter(trace, filter_type, cutoff_freqs, self.parser.fs, self.parser.order),
            #         axis=0, arr=self.parser.seismic_data
            #     )
            #
            # if filter_type == 'notch':
            #     print(f"Applying Notch Filter: freq={low_cutoff}, Q=30")
            #     filtered_seismic_data = np.apply_along_axis(
            #         lambda trace: self.parser.filter.apply_filter(trace, 'notch', cutoff_freqs, self.parser.fs),
            #         axis=0, arr=self.parser.seismic_data
            #     )
            # if filter_type == 'median':
            #     print(f"Applying Median Filter: kernel_size={self.parser.order}")
            #     filtered_seismic_data = np.apply_along_axis(
            #         lambda trace: self.parser.filter.apply_filter(trace, 'median', cutoff_freqs, self.parser.fs,
            #                                                       self.parser.order),
            #         axis=0, arr=self.parser.seismic_data
            #     )

            self.plot_radargram(filtered_seismic_data, f"Filtered Radargram ({filter_type.capitalize()} Filter)")

        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Filtering Error", f"An unexpected error occurred: {e}")

    def update_filtered_plot(self, filtered_seismic_data):
        # Re-plot the filtered radargram
        self.plot_radargram(filtered_seismic_data, f"Filtered Radargram ({self.filter_box.currentText()} Filter)")
    

def main():
    app = QApplication(sys.argv)
    plotter = MainWindow()
    plotter.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()