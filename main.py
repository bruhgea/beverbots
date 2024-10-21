import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from scipy import ndimage as ndi
from shutil import copyfile
from skimage import exposure
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF
from obspy.io.segy.segy import SEGYFile
import matplotlib.pyplot as plt
from obspy.io.segy.segy import SEGYFile
from obspy import read
from scipy.signal import butter, filtfilt
import sys

file_path = '20240625132506554_TestGraaf22.25.sgy'

# Filtering functions
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filter(data, filter_type, cutoff_freqs, fs, order=5):
    if filter_type == 'lowpass':
        b, a = butter_lowpass(cutoff_freqs[0], fs, order=order)
    elif filter_type == 'highpass':
        b, a = butter_highpass(cutoff_freqs[0], fs, order=order)
    elif filter_type == 'bandpass':
        b, a = butter_bandpass(cutoff_freqs[0], cutoff_freqs[1], fs, order=order)
    else:
        raise ValueError("Unknown filter type. Choose from 'lowpass', 'highpass', or 'bandpass'.")

    filtered_data = filtfilt(b, a, data)
    return filtered_data


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

#
# class MplCanvas(FigureCanvasQTAgg):
#     '''
#     separate obj for matplotlib.figure
#     '''
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)
#         super(MplCanvas, self).__init__(fig)
#
#
#
# class MainWindow(QMainWindow):
#     '''
#     Main window
#     '''
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)
#         self.init_ui()
#         self.file_path = str()
#
#     def init_ui(self):
#         self.setWindowTitle('GPR plotter')
#         self.setGeometry(100, 100, 800, 600)
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         layout = QVBoxLayout()
#
#         #   button to open file
#         self.file_btn = QPushButton("Open .sgy file")
#         self.file_btn.clicked.connect(self.file_btn_clicked)
#
#         #   button to apply filter
#         self.filter_btn = QPushButton("Apply filter")
#         self.filter_btn.clicked.connect(self.filter_btn_clicked)
#
#         #   dropdown menu to select filter
#         self.filter_box = QComboBox()
#         self.filter_box.addItems(['None', 'Low-pass', 'High-pass', 'Band-pass'])
#
#         #   matplotlib FigureCanvas obj
#         self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)
#
#         #   add all widgets
#         layout.addWidget(self.file_btn)
#         layout.addWidget(self.filter_box)
#         layout.addWidget(self.filter_btn)
#         layout.addWidget(self.mpl_canvas)
#         self.central_widget.setLayout(layout)
#
#     #   button handlers
#     def file_btn_clicked(self):
#         '''
#         Open .sgy file and plot
#         '''
#         self.file_path = QFileDialog.getOpenFileName(self, 'Open file', '', '*.sgy')[0]
#         print(f'opened file {self.file_path}')
#         try:
#             # Read the SEGY file using ObsPy
#             segy_stream = read(self.file_path, format="SEGY")
#
#             # Get the number of traces and the number of samples per trace
#             num_traces = len(segy_stream)
#             num_samples = len(segy_stream[0].data)
#
#             # Print some basic info about the traces
#             print(f"Number of traces: {num_traces}")
#             print(f"Sample points per trace: {num_samples}")
#
#             # Create a 2D numpy array to hold all trace data
#             seismic_data = np.zeros((num_samples, num_traces))
#
#             # Fill the array with trace data
#             for i, trace in enumerate(segy_stream):
#                 seismic_data[:, i] = trace.data  # Assign each trace's data to a column in the 2D array
#
#             # Create the time axis (assuming uniform sample interval)
#             sample_interval = segy_stream[0].stats.delta  # Sample interval in seconds
#             time_axis = np.arange(0, num_samples * sample_interval, sample_interval)
#
#             # Plot the radargram
#             self.mpl_canvas.axes.imshow(seismic_data, aspect='auto', cmap='seismic', extent=[0, num_traces, time_axis[-1], time_axis[0]])
#             #self.mpl_canvas.axes.colorbar(label="Amplitude")
#             self.mpl_canvas.axes.set_title("Radargram (Seismic Section)")
#             self.mpl_canvas.axes.set_xlabel("Trace number")
#             self.mpl_canvas.axes.set_ylabel("Time (s)")
#
#             #   refresh canvas
#             self.mpl_canvas.draw()
#
#         except Exception as e:
#             print(f"Error opening SEGY file with ObsPy: {e}")
#
#     def filter_btn_clicked(self):
#         pass    # TODO
#
#
# def main():
#     app = QApplication(sys.argv)
#     plotter = MainWindow()
#     plotter.show()
#     sys.exit(app.exec_())
#
# if __name__ == '__main__':
#     main()


# Open the SEGY file using ObsPy
try:
    # Read the SEGY file using ObsPy
    segy_stream = read(file_path, format="SEGY")

    # Get the number of traces and the number of samples per trace
    num_traces = len(segy_stream)
    num_samples = len(segy_stream[0].data)

    # Print some basic info about the traces
    print(f"Number of traces: {num_traces}")
    print(f"Sample points per trace: {num_samples}")

    # Create a 2D numpy array to hold all trace data
    seismic_data = np.zeros((num_samples, num_traces))

    # Fill the array with trace data
    for i, trace in enumerate(segy_stream):
        seismic_data[:, i] = trace.data  # Assign each trace's data to a column in the 2D array

    # Create the time axis (assuming uniform sample interval)
    sample_interval = segy_stream[0].stats.delta  # Sample interval in seconds
    time_axis = np.arange(0, num_samples * sample_interval, sample_interval)

    # Plot the radargram
    plt.figure(figsize=(10, 6))
    plt.imshow(seismic_data, aspect='auto', cmap='seismic', extent=[0, num_traces, time_axis[-1], time_axis[0]])
    plt.colorbar(label="Amplitude")
    plt.title("Radargram (Seismic Section)")
    plt.xlabel("Trace number")
    plt.ylabel("Time (s)")
    plt.show()

    # Graphic

    # Create a single array to hold the continuous time series data (all traces concatenated)
    continuous_data = np.concatenate([trace.data for trace in segy_stream])

    # Create a time axis for the entire continuous plot
    sample_interval = segy_stream[0].stats.delta  # Sample interval in seconds
    total_time = num_samples * num_traces * sample_interval
    time_axis = np.linspace(0, total_time, num_samples * num_traces)

    fs = 1 / sample_interval
    filter_type = 'lowpass' # Choose from 'lowpass', 'highpass', or 'bandpass'
    cutoff_freqs = [0.1]  # Cutoff frequency (Hz) for the filter

    filtered_data = apply_filter(continuous_data, filter_type, cutoff_freqs, fs, order=5)

    # Apply moving average filter (optional)
    # Uncomment to apply moving average
    window_size = 100  # Window size for the moving average
    filtered_data = moving_average(filtered_data, window_size)

    # Plot the continuous line for all traces
    plt.figure(figsize=(12, 6))
    # Plot original data
    plt.plot(time_axis, continuous_data, color='gray', alpha=0.5, label="Original Data")

    # Plot filtered data
    plt.plot(time_axis[:len(filtered_data)], filtered_data, color='k', label=f"Filtered Data ({filter_type})")

    plt.title("All Seismic Traces with Filtering")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Error opening SEGY file with ObsPy: {e}")
