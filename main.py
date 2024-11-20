import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
from scipy import ndimage as ndi
from shutil import copyfile
from skimage import exposure
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, \
    QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QSpinBox, QHBoxLayout, QSlider
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF
from obspy.io.segy.segy import SEGYFile
import matplotlib.pyplot as plt
from obspy.io.segy.segy import SEGYFile
from obspy import read
from scipy.signal import butter, filtfilt
import sys
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


class MplCanvas(FigureCanvasQTAgg):
    '''
    separate obj for matplotlib.figure
    '''

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

        # Store the original limits for resetting later
        self.original_xlim = None
        self.original_ylim = None

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


class MainWindow(QMainWindow):
    '''
    Main window
    '''

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()
        self.file_path = str()
        self.seismic_data = None  # Store seismic data here for filtering
        self.fs = None  # Store sampling rate
        self.cutoff_freqs = [50]  # Default cutoff frequency
        self.order = 5  # Default filter order
        # self.color_scheme_box = 'seismic'
        self.color_scheme = 'seismic'
        self.zoom_factor = 1.1  # Factor for zooming in and out

    def init_ui(self):

        self.setWindowTitle('GPR plotter')
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()

        #   button to open file
        self.file_btn = QPushButton("Open .sgy file")
        self.file_btn.clicked.connect(self.file_btn_clicked)

        #   button to apply filter
        self.filter_btn = QPushButton("Apply filter")
        self.filter_btn.clicked.connect(self.filter_btn_clicked)

        # Dropdown menu to select filter
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
        self.toolbar = NavigationToolbar2QT(self.mpl_canvas, self)

        # Add Reset Zoom Button
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)

        # Wrap the canvas in a scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.mpl_canvas)
        self.scroll_area.setWidgetResizable(True)

        # Add all widgets to the layout
        layout.addWidget(self.file_btn)
        layout.addWidget(self.filter_box)
        layout.addWidget(self.filter_btn)
        layout.addLayout(controls_layout)
        layout.addWidget(self.scroll_area)  # Use scroll area for the canvas
        layout.addWidget(self.toolbar)
        layout.addWidget(self.color_scheme_box)
        layout.addLayout(zoom_controls_layout)
        layout.addWidget(self.reset_zoom_btn)


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
        self.file_path = QFileDialog.getOpenFileName(self, 'Open file', '', '*.sgy')[0]
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

""" Initial code for filtering and plotting the data """
# # Open the SEGY file using ObsPy
# try:
#     # Read the SEGY file using ObsPy
#     segy_stream = read(file_path, format="SEGY")
#
#     # Get the number of traces and the number of samples per trace
#     num_traces = len(segy_stream)
#     num_samples = len(segy_stream[0].data)
#
#     # Print some basic info about the traces
#     print(f"Number of traces: {num_traces}")
#     print(f"Sample points per trace: {num_samples}")
#
#     # Create a 2D numpy array to hold all trace data
#     seismic_data = np.zeros((num_samples, num_traces))
#
#     # Fill the array with trace data
#     for i, trace in enumerate(segy_stream):
#         seismic_data[:, i] = trace.data  # Assign each trace's data to a column in the 2D array
#
#     # Create the time axis (assuming uniform sample interval)
#     sample_interval = segy_stream[0].stats.delta  # Sample interval in seconds
#     time_axis = np.arange(0, num_samples * sample_interval, sample_interval)
#
#     # Plot the radargram
#     plt.figure(figsize=(10, 6))
#     plt.imshow(seismic_data, aspect='auto', cmap='seismic', extent=[0, num_traces, time_axis[-1], time_axis[0]])
#     plt.colorbar(label="Amplitude")
#     plt.title("Radargram (Seismic Section)")
#     plt.xlabel("Trace number")
#     plt.ylabel("Time (s)")
#     plt.show()
#
#     # Graphic
#
#     # Create a single array to hold the continuous time series data (all traces concatenated)
#     continuous_data = np.concatenate([trace.data for trace in segy_stream])
#
#     # Create a time axis for the entire continuous plot
#     sample_interval = segy_stream[0].stats.delta  # Sample interval in secondsfrom PyQt5.QtWidgets import QScrollArea
#     total_time = num_samples * num_traces * sample_interval
#     time_axis = np.linspace(0, total_time, num_samples * num_traces)
#
#     fs = 1 / sample_interval
#     filter_type = 'lowpass' # Choose from 'lowpass', 'highpass', or 'bandpass'
#     cutoff_freqs = [50]  # Cutoff frequency (Hz) for the filter
#     fs = 1 / segy_stream[0].stats.delta  # Sampling rate (Hz)
#
#     filtered_data = apply_filter(continuous_data, filter_type, cutoff_freqs, fs, order=5)
#     # Apply filter on each trace individually
#     filtered_seismic_data = np.zeros_like(seismic_data)
#     for i in range(num_traces):
#         filtered_seismic_data[:, i] = apply_filter(seismic_data[:, i], filter_type, cutoff_freqs, fs)
#
#         # Plot the radargram after filtering
#     plt.figure(figsize=(10, 6))
#     plt.imshow(filtered_seismic_data, aspect='auto', cmap='seismic', extent=[0, num_traces, num_samples, 0])
#     plt.colorbar(label="Amplitude")
#     plt.title(f"Filtered Radargram ({filter_type.capitalize()} Filter)")
#     plt.xlabel("Trace number")
#     plt.ylabel("Sample")
#     plt.show()
#
#     # Apply moving average filter (optional)
#     # Uncomment to apply moving average
#     window_size = 100  # Window size for the moving average
#     filtered_data = moving_average(filtered_data, window_size)
#
#     # Plot the continuous line for all traces
#     plt.figure(figsize=(12, 6))
#     # Plot original data
#     plt.plot(time_axis, continuous_data, color='gray', alpha=0.5, label="Original Data")
#
#     # Plot filtered data
#     plt.plot(time_axis[:len(filtered_data)], filtered_data, color='k', label=f"Filtered Data ({filter_type})")
#
#     plt.title("All Seismic Traces with Filtering")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.legend(loc='upper right')
#     plt.grid(True)
#     plt.show()
#
# except Exception as e:
#     print(f"Error opening SEGY file with ObsPy: {e}")