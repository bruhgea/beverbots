import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT, NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

from scipy import ndimage as ndi
from shutil import copyfile
from skimage import exposure
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QGridLayout, QHBoxLayout, QSpinBox
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QSpinBox, QHBoxLayout, QSlider
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from obspy.io.segy.segy import SEGYFile
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import butter, filtfilt
import sys
import struct
from matplotlib.lines import Line2D
import matplotlib.patches as patches


from scipy.signal import butter, lfilter  # Use lfilter instead of filtfilt
from PyQt5.QtWidgets import QLineEdit
from scipy.ndimage import gaussian_filter1d
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QScrollBar
from PyQt5.QtWidgets import QMessageBox

file_path = '20240625132506554_TestGraaf22.25.sgy'

# Filtering functions
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def apply_filter(data, filter_type, cutoff_freqs, fs, order=5):
    if filter_type == 'lowpass':
        b, a = butter_lowpass(cutoff_freqs[0], fs, order=order)
    elif filter_type == 'highpass':
        b, a = butter_highpass(cutoff_freqs[0], fs, order=order)
    elif filter_type == 'bandpass':
        if len(cutoff_freqs) != 2 or cutoff_freqs[0] >= cutoff_freqs[1]:
            raise ValueError("For bandpass filter, cutoff_freqs should contain two values: [lowcut, highcut]")
        b, a = butter_bandpass(cutoff_freqs[0], cutoff_freqs[1], fs, order=order)
    elif filter_type == 'moving_average':
        return moving_average(data, order)  # window size is `order`
    elif filter_type == 'gaussian':
        sigma = order  # Use order as sigma for Gaussian
        return gaussian_filter1d(data, sigma=sigma, mode='nearest')
    else:
        raise ValueError("Unknown filter type. Choose from 'lowpass', 'highpass', or 'bandpass'.")

    # Apply the filter
    filtered_data = filtfilt(b, a, data)

    # Adjust length to match original data
    if len(filtered_data) > len(data):
        filtered_data = filtered_data[:len(data)]
    elif len(filtered_data) < len(data):
        filtered_data = np.pad(filtered_data, (0, len(data) - len(filtered_data)), mode='constant')

    return filtered_data


def moving_average(data, window_size):
    if len(data) < window_size:
        raise ValueError("Window size must be less than or equal to the length of the data.")
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def filter_in_chunks(seismic_data, filter_type, cutoff_freqs, fs, order=5, chunk_size=100):
    num_traces = seismic_data.shape[1]
    filtered_seismic_data = np.zeros_like(seismic_data)

    for i in range(0, num_traces, chunk_size):
        end = min(i + chunk_size, num_traces)
        for j in range(i, end):
            try:
                filtered_trace = apply_filter(seismic_data[:, j], filter_type, cutoff_freqs, fs, order)
                if filtered_trace.shape[0] != seismic_data.shape[0]:
                    print(f"Warning: Trace {j} shape mismatch after filtering. Adjusting shape.")
                filtered_seismic_data[:, j] = filtered_trace[:seismic_data.shape[0]]  # Match length
            except Exception as e:
                print(f"Error filtering trace {j}: {e}")

    return filtered_seismic_data

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
        sample_interval = self.segy_stream[0].stats.delta * 1000  # Sample interval to nanoseconds (is encoded in microseconds in file)
        self.time_axis = np.arange(0, self.samples_num * sample_interval, sample_interval)


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
        self.file_path = str()
        self.parser = None
        self.seismic_data = None  # Store seismic data here for filtering
        self.fs = None  # Store sampling rate
        self.cutoff_freqs = [50]  # Default cutoff frequency
        self.order = 5  # Default filter order
        # self.color_scheme_box = 'seismic'
        self.color_scheme = 'seismic'
        self.zoom_factor = 1.1  # Factor for zooming in and out

    def init_ui(self):

        self.setWindowTitle('GPR plotter')
        self.setGeometry(0, 0, 1920, 1080)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self)

        #   button to open file
        self.file_btn = QPushButton("Open .sgy file")
        self.file_btn.clicked.connect(self.file_btn_clicked)

        #   button to apply filter
        self.filter_btn = QPushButton("Apply filter")
        self.filter_btn.clicked.connect(self.filter_btn_clicked)

        #   dropdown menu to select filter
        self.filter_box = QComboBox()
        self.filter_box.addItems(['None', 'lowpass', 'highpass', 'bandpass', 'moving_average', 'gaussian'])
        self.filter_box.currentTextChanged.connect(self.on_filter_change)

        # Color scheme selection
        self.color_scheme_box = QComboBox()
        self.color_scheme_box.addItems(['seismic', 'viridis', 'plasma', 'inferno', 'gray', 'magma', 'cividis'])
        self.color_scheme_box.currentTextChanged.connect(self.update_color_scheme)

        # Zoom in/out buttons
        zoom_controls_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_controls_layout.addWidget(self.zoom_in_btn)
        zoom_controls_layout.addWidget(self.zoom_out_btn)

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
        controls_layout.addWidget(self.cutoff_input_low)
        controls_layout.addWidget(QLabel("High Cutoff:"))
        controls_layout.addWidget(self.cutoff_input_high)
        controls_layout.addWidget(QLabel("Order:"))
        controls_layout.addWidget(self.order_spinbox)

        # Matplotlib FigureCanvas object
        self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)

        # Add Reset Zoom Button
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)

        # Wrap the canvas in a scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.mpl_canvas)
        self.scroll_area.setWidgetResizable(True)

        #   matplotlib FigureCanvas obj for GPS coordinates plot
        self.mpl_gps_canvas = MplGpsCanvas(self, width=5, height=4)

         #   gain function
        self.gain_widget = GainWidget(self, width=4)

        # Add all widgets to the layout
        #layout.addWidget(self.mpl_canvas, 0, 0)
        layout.addWidget(self.filter_box, 0, 0)
        layout.addWidget(self.filter_btn, 1, 0)
        layout.addLayout(controls_layout, 2, 0)
        layout.addWidget(self.scroll_area, 3, 0)  # Use scroll area for the canvas
        layout.addWidget(self.toolbar, 4, 0)
        layout.addWidget(self.color_scheme_box, 5, 0)
        layout.addLayout(zoom_controls_layout, 6, 0)
        layout.addWidget(self.reset_zoom_btn, 7, 0)
        layout.addWidget(self.file_btn, 8, 0)
        layout.addWidget(self.mpl_gps_canvas, 0, 1)
        layout.addWidget(self.gain_widget, 1, 1, 6, 1, Qt.AlignRight)

        self.central_widget.setLayout(layout)

    def reset_zoom(self):
        """Reset the zoom level to the original limits."""
        if self.mpl_canvas.original_xlim is not None and self.mpl_canvas.original_ylim is not None:
            self.mpl_canvas.axes.set_xlim(self.mpl_canvas.original_xlim)
            self.mpl_canvas.axes.set_ylim(self.mpl_canvas.original_ylim)
            self.mpl_canvas.draw()

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

    def update_view(self):
        # Update plot limits based on current x and y limits
        self.mpl_canvas.axes.set_xlim(self.current_xlim)
        self.mpl_canvas.axes.set_ylim(self.current_ylim)
        self.mpl_canvas.draw()

        # Adjust scroll bar ranges and positions based on current view
        self.h_scroll.setRange(int(self.initial_xlim[0]),
                               int(self.initial_xlim[1] - (self.current_xlim[1] - self.current_xlim[0])))
        self.v_scroll.setRange(int(self.initial_ylim[0]),
                               int(self.initial_ylim[1] - (self.current_ylim[1] - self.current_ylim[0])))
        self.h_scroll.setPageStep(int(self.current_xlim[1] - self.current_xlim[0]))
        self.v_scroll.setPageStep(int(self.current_ylim[1] - self.current_ylim[0]))

    def update_xlim(self, value):
        # Update the x-axis limits based on the scroll bar position
        new_xlim = (value, value + (self.current_xlim[1] - self.current_xlim[0]))
        self.current_xlim = new_xlim
        self.update_view()

    def update_ylim(self, value):
        # Update the y-axis limits based on the scroll bar position
        new_ylim = (value, value + (self.current_ylim[1] - self.current_ylim[0]))
        self.current_ylim = new_ylim
        self.update_view()
    def update_color_scheme(self, scheme):
        self.color_scheme = scheme
        if self.seismic_data is not None:
            self.plot_radargram(self.seismic_data, "Radargram with Color Scheme")

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
        try:
            # Read the SEGY file using ObsPy
            segy_stream = read(self.file_path, format="SEGY")

            # Get the number of traces and the number of samples per trace
            num_traces = len(segy_stream)
            num_samples = len(segy_stream[0].data)

            # Print some basic info about the traces
            print(f"Number of traces: {num_traces}")
            print(f"Sample points per trace: {num_samples}")

            # Create a 2D numpy array to hold all trace data
            self.seismic_data = np.zeros((num_samples, num_traces))

            # Fill the array with trace data
            for i, trace in enumerate(segy_stream):
                self.seismic_data[:, i] = trace.data  # Assign each trace's data to a column in the 2D array

            # Create the time axis (assuming uniform sample interval)
            sample_interval = segy_stream[0].stats.delta  # Sample interval in seconds
            self.fs = 1 / sample_interval  # Sampling rate (Hz)

            # Debug: Confirm seismic_data is correctly loaded
            print(f"Seismic data loaded: {self.seismic_data.shape}")
            print(f"Sampling rate (fs): {self.fs}")

            # Plot the original radargram
            self.plot_radargram(self.seismic_data, "Radargram (Seismic Section)")

        except Exception as e:
            print(f"Error opening SEGY file with ObsPy: {e}")
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText(f"Error opening SEGY file: {e}")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def plot_radargram(self, data, title):
        '''
        Helper function to plot radargram on the canvas
        '''
        self.mpl_canvas.axes.clear()
        self.mpl_canvas.axes.imshow(data, aspect='auto', cmap=self.color_scheme)
        self.mpl_canvas.axes.set_title(title)
        self.mpl_canvas.axes.set_xlabel("Trace number")
        self.mpl_canvas.axes.set_ylabel("Time (s)")
        self.mpl_canvas.draw()

        # Store the original limits for the reset function
        self.mpl_canvas.original_xlim = self.mpl_canvas.axes.get_xlim()
        self.mpl_canvas.original_ylim = self.mpl_canvas.axes.get_ylim()

        self.mpl_canvas.draw()

    # def update_cutoff(self):
    #     self.cutoff_freqs = [self.cutoff_slider.value()]

    def update_order(self):
        self.order = self.order_spinbox.value()

    def moving_average(self, data, window_size):
        """
        Apply a moving average filter to the input data.
        """
        if len(data) < window_size:
            raise ValueError("Window size must be less than or equal to the length of the data.")
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    def filter_btn_clicked(self):
        '''
        Apply the selected filter to the data and re-plot
        '''
        if self.seismic_data is None:
            msg1 = QMessageBox()
            msg1.setWindowTitle("Error")
            msg1.setText("No data loaded to apply filter!")
            msg1.setIcon(QMessageBox.Critical)
            msg1.setStandardButtons(QMessageBox.Ok)
            msg1.exec_()
            return
        else:
            print(f"Data ready for filtering. Shape: {self.seismic_data.shape}, fs: {self.fs}")

        filter_type = self.filter_box.currentText().lower()

        # Get cutoff frequencies from input boxes

        low = self.cutoff_input_low.text()
        high = self.cutoff_input_high.text()

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
            self.plot_radargram(self.seismic_data, "Radargram (Original Data)")
            return

        if filter_type == 'bandpass':
            cutoff_freqs = [low_cutoff, high_cutoff]
        else:
            cutoff_freqs = [low_cutoff]

        # Apply the filter
        if filter_type == 'moving_average':
            window_size = self.order  # Use order as the window size for moving average
            filtered_seismic_data = np.apply_along_axis(
                lambda trace: self.moving_average(trace, window_size),
                axis=0, arr=self.seismic_data
            )
        else:
            filtered_seismic_data = np.apply_along_axis(
                lambda trace: apply_filter(trace, filter_type, cutoff_freqs, self.fs, self.order),
                axis=0, arr=self.seismic_data
            )

        self.plot_radargram(filtered_seismic_data, f"Filtered Radargram ({filter_type.capitalize()} Filter)")

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