# parsing & filtering logic

import numpy as np
from obspy import read
from scipy.signal import butter, lfilter, filtfilt, medfilt, iirnotch
from scipy.ndimage import gaussian_filter1d
import struct
import gc
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

        print(f'sample_interval = {self.sample_interval}')
        self.fs = self.segy_stream[0].stats.sampling_rate       # sample freq in Hz

        self.time_axis = np.linspace(0, (self.samples_num - 1) * self.sample_interval, self.samples_num)

        # init filter instance
        self.filter = self.GprFilter(self)

        # Store the original, unmodified seismic data
        self.original_seismic_data = self.seismic_data.copy()

        # List to track applied filters and gains
        self.applied_modifications = []

    def reset_data(self):
        """Reset seismic data to original state"""
        self.seismic_data = self.original_seismic_data.copy()
        self.applied_modifications = []

    #   gain
    def apply_gain(self, gain_db):
        exp_factor = 10**(gain_db/20)
        #   DEBUG
        print(f'apply_gain() called, gain_db = {gain_db}, exp_factor = {exp_factor}')
        print(f'original first trace\n{self.seismic_data[0]}')

        self.seismic_data = np.sign(self.seismic_data) * (np.abs(self.seismic_data) ** exp_factor)
        print(f'new first trace\n{self.seismic_data[0]}')
        
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

        def apply_filter_stack(self, data, filter_type, cutoff_freqs, fs, order=5):
            """Apply a filter to the current data and log the modification"""
            try:
                filtered_data = self.apply_filter(data, filter_type, cutoff_freqs, fs, order)

                # Store the modification for tracking
                self.parent.applied_modifications.append({
                    'type': 'filter',
                    'filter_type': filter_type,
                    'cutoff_freqs': cutoff_freqs,
                    'order': order
                })

                return filtered_data
            except Exception as e:
                print(f"Error in filter stack: {e}")
                return data
