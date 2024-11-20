import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from scipy import ndimage as ndi
from shutil import copyfile
from skimage import exposure
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QSizePolicy, QGridLayout
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



#file_path = '20240625132506554_TestGraaf22.25.sgy'

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
        self.axes.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.axes.yaxis.get_major_formatter().set_scientific(False)
        self.axes.yaxis.get_major_formatter().set_useOffset(False)
        self.axes.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
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
        self.filter_box.addItems(['None', 'Low-pass', 'High-pass', 'Band-pass'])

        #   matplotlib FigureCanvas obj for seismic plot
        self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)

        #   matplotlib FigureCanvas obj for GPS coordinates plot
        self.mpl_gps_canvas = MplGpsCanvas(self, width=5, height=4)

        #   graph toolbar
        self.mpl_toolbar = NavigationToolbar2QT(self.mpl_canvas, self)

        #   gain function
        self.gain_widget = GainWidget(self, width=4)

        #   add all widgets
        layout.addWidget(self.mpl_canvas, 0, 0)
        layout.addWidget(self.mpl_gps_canvas, 1, 0)
        layout.addWidget(self.file_btn, 2, 0)
        layout.addWidget(self.filter_box, 3, 0)
        layout.addWidget(self.filter_btn, 4, 0)
        layout.addWidget(self.mpl_toolbar, 5, 0)
        layout.addWidget(self.gain_widget, 0, 1, 6, 1, Qt.AlignRight)
        
        self.central_widget.setLayout(layout)

    def __str__(self):
        print('MainWindow() called!')
        return self
    
    #   button handlers
    #   TODO: clear data when open new seismogram instead of existing one
    def file_btn_clicked(self):
        '''
        Open .sgy file and plot
        '''
        print('file_btn_clicked event triggered!')
        self.file_path = QFileDialog.getOpenFileName(self, 'Open file', '', '*.sgy')[0]
        if (len(self.file_path) == 0):
            return
        print(f'opened file {self.file_path}')
        #   init Gpr parser
        self.parser = GprParser(self.file_path)
        # Plot the radargram
        self.mpl_canvas.axes.imshow(self.parser.seismic_data, aspect='auto', cmap='seismic', extent=[0, self.parser.traces_num, self.parser.time_axis[-1], self.parser.time_axis[0]])
        #self.mpl_canvas.axes.colorbar(label="Amplitude")
        self.mpl_canvas.axes.set_title("Radargram (Seismic Section)")
        self.mpl_canvas.axes.set_xlabel("Trace number")
        self.mpl_canvas.axes.set_ylabel("Time (ns)")            
        #   refresh canvas
        self.mpl_canvas.draw()

        # Plot GPS coordinates
        lonX, latY = self.parser.trace_coordinates
        self.mpl_gps_canvas.plot_coordinates(lonX, latY)
       
    def filter_btn_clicked(self):
        pass    # TODO
    

def main():
    app = QApplication(sys.argv)
    plotter = MainWindow()
    plotter.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()