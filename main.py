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
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, \
    QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QGridLayout, QHBoxLayout, QSpinBox
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QLabel, \
    QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox, QSpinBox, QHBoxLayout, QSlider
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

from gpr_parser import GprParser


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


class GainWidget(QWidget):
    '''
    separate obj for gain widget
    '''

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout()

        self.gain_input = QLineEdit(self)
        self.gain_input.setPlaceholderText('Gain [dB]')

        self.gain_button = QPushButton('Apply gain', self)
        self.gain_button.clicked.connect(self.gain_button_clicked)

        self.layout.addWidget(self.gain_input)
        self.layout.addWidget(self.gain_button)

        self.setLayout(self.layout)

    def gain_button_clicked(self):
        gain_db = float(self.gain_input.text())
        self.parent.parser.apply_gain(gain_db)
        self.parent.plot_radargram(self.parent.parser.seismic_data, 'gain applied!')


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
        self.filter_box.addItems(
            ['None', 'lowpass', 'highpass', 'bandpass', 'moving_average', 'gaussian', 'notch', 'median'])
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

        # Add Reset Data Button
        self.reset_data_btn = QPushButton("Reset Data")
        self.reset_data_btn.clicked.connect(self.reset_data)

        # Wrap the canvas in a scroll area
        # self.scroll_area = QScrollArea()
        # self.scroll_area.setWidget(self.mpl_canvas)
        # self.scroll_area.setWidgetResizable(True)

        #   matplotlib FigureCanvas obj for GPS coordinates plot
        self.mpl_gps_canvas = MplGpsCanvas(self, width=5, height=4)

        #   gain function
        self.gain_widget = GainWidget(self)

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
        layout.addWidget(self.reset_data_btn, 9, 0)
        layout.addWidget(self.mpl_gps_canvas, 0, 1)
        layout.addWidget(self.gain_widget, 1, 1, 7, 1, Qt.AlignRight)

        self.central_widget.setLayout(layout)

    def reset_data(self):
        """Reset the data and plot to original state"""
        if self.parser is not None:
            # Reset the data
            self.parser.reset_data()

            # Re-plot the original radargram
            self.plot_radargram(self.parser.seismic_data, "Original Radargram (Reset)")

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
        print('plot_radargram called!')
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
        if self.parser != None:  # ignore if parser not initialized
            self.parser.order = self.order_spinbox.value()

    def filter_btn_clicked(self):
        '''
        Apply the selected filter to the data and re-plot
        '''

        try:
            if self.parser is None:
                QMessageBox.critical(self, "Error", "No data loaded to apply filter!")

                return
            else:
                print(f"Data ready for filtering. Shape: {self.parser.seismic_data.shape}, fs: {self.parser.fs}")

            filter_type = self.filter_box.currentText().lower()

            # Get cutoff frequencies from input boxes

            low = self.cutoff_input_low.text()
            high = self.cutoff_input_high.text()

            print(f"Filter type: {filter_type}, low: {low}, high: {high}")

            if not low:
                QMessageBox.critical(self, "Error", "Please set lowcut frequency")
                return

            if high == '' and filter_type == 'bandpass':
                QMessageBox.critical(self, "Error",
                                     "Please set both lowcut and highcut frequencies for bandpass filter")
                return

            low_cutoff = float(self.cutoff_input_low.text())
            high_cutoff = float(self.cutoff_input_high.text()) if filter_type == 'bandpass' else None

            if filter_type == 'none':
                print("No filter selected")
                self.plot_radargram(self.parser.seismic_data, "Radargram (Original Data)")
                return

                # Validate cutoff frequencies
            if low_cutoff <= 0 or (high_cutoff is not None and high_cutoff <= 0):
                QMessageBox.critical(self, "Error", "Cutoff frequencies must be positive")
                return

                # Validate against Nyquist frequency
            nyquist = 0.5 * self.parser.fs
            if low_cutoff > nyquist or (high_cutoff and high_cutoff > nyquist):
                QMessageBox.critical(self, "Error",
                                     f"Cutoff frequency must be less than Nyquist frequency ({nyquist} Hz)")
                return

            if filter_type == 'bandpass':
                # Validate both low and high cutoff frequencies
                if not low or not high:
                    QMessageBox.critical(self, "Error",
                                         "Both low and high cutoff frequencies are required for bandpass filter")
                    return

                low_cutoff = float(low)
                high_cutoff = float(high)

                # Ensure low cutoff is less than high cutoff
                if low_cutoff >= high_cutoff:
                    QMessageBox.critical(self, "Error", "Low cutoff frequency must be less than high cutoff frequency")
                    return

                cutoff_freqs = [low_cutoff, high_cutoff]
            else:
                cutoff_freqs = [low_cutoff]

            filtered_seismic_data = self.parser.seismic_data.copy()

            for i in range(self.parser.seismic_data.shape[1]):
                try:
                    filtered_seismic_data[:, i] = self.parser.filter.apply_filter_stack(
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

            self.parser.seismic_data = filtered_seismic_data

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
