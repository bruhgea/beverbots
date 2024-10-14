import numpy as np
import matplotlib.pyplot as plt
import segyio
from scipy import ndimage as ndi
from shutil import copyfile
from skimage import exposure

# # Get access to data
# from google.colab import drive
# drive.mount('/content/drive')

# seismic_data = segyio.tools.cube('20191207173136114_Bentelo14April2022/20191207173136114_Bentelo14April2022.sgy')
#
# print('Survey Inline/Xline shape:' +str(np.shape(seismic_data)[0])+' / ' +str(np.shape(seismic_data)[1]))
#
# fig = plt.figure(figsize=(18,9))
# ax = fig.add_subplot(121)
# sim =ax.imshow(seismic_data[:,120,:],cmap='gray');
# fig.colorbar(sim,ax=ax)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.invert_xaxis()
#
# fig = plt.figure(figsize=(18,9))
# ax = fig.add_subplot(121)
# sim =ax.imshow(seismic_data[:,120,:].T,cmap='gray');
# fig.colorbar(sim,ax=ax)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.invert_xaxis()

from obspy.io.segy.segy import SEGYFile
import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import SEGYFile
from obspy import read
from scipy.signal import butter, filtfilt

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


    # # Plot the first trace
    # #segy_stream[0].plot()
    #
    # segy_stream.filter("lowpass", freq=0.1, corners=2)
    # segy_stream.plot(type='dayplot', interval=60, right_vertical_labels=False,
    #                  vertical_scaling_range=1, one_tick_per_line=True, color=['k', 'r', 'b', 'g'],
    #                  show_y_UTC_label=False, events={'text': 'text', 'color': 'r', 'linestyle': '-.'})
    # # print array of data
    # print(segy_stream[0].data)
    #
    # # Read metadata
    # for trace in segy_stream:
    #     print(trace.stats)
    #
    # # Get the number of traces and the number of samples per trace
    # num_traces = len(segy_stream)
    # num_samples = len(segy_stream[0].data)
    #
    # # Create a time axis (assuming uniform sample interval)
    # sample_interval = segy_stream[0].stats.delta  # Sample interval (in seconds)
    # time_axis = np.arange(0, num_samples * sample_interval, sample_interval)
    #
    # # Plot all traces
    # plt.figure(figsize=(12, 6))
    #
    # for i, trace in enumerate(segy_stream):
    #     # Offset each trace for clarity in the plot (e.g., offset by i * constant)
    #     plt.plot(time_axis, trace.data + i * 1000, label=f'Trace {i + 1}')
    #
    # plt.title('All Traces from SEGY File')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude (offset for clarity)')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    # plt.show()

    # # Read the SEGY file using ObsPy
    # segy_stream = read(file_path, format="SEGY")
    #
    # # Get the number of traces and the number of samples per trace
    # num_traces = len(segy_stream)
    # num_samples = len(segy_stream[0].data)
    #
    # # Create a time axis (assuming uniform sample interval)
    # sample_interval = segy_stream[0].stats.delta  # Sample interval (in seconds)
    # time_axis = [sample_interval * i for i in range(num_samples)]
    #
    # # Plot all traces
    # plt.figure(figsize=(12, 6))
    #
    # for i, trace in enumerate(segy_stream):
    #     # Offset each trace for clarity in the plot (e.g., offset by i * constant)
    #     plt.plot(time_axis, trace.data + i * 1000, label=f'Trace {i + 1}')
    #
    # plt.title('All Traces from SEGY File')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude (offset for clarity)')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    # plt.show()


except Exception as e:
    print(f"Error opening SEGY file with ObsPy: {e}")

# import matplotlib.pyplot as plt
# import pathlib
#
# V3D_path = pathlib.Path("data/volve10r12-full-twt-sub3d.sgy")
# print("3D", V3D_path, V3D_path.exists())

# x <- readGPR(dsn = "20191207173136114_Bentelo14April2022/20191207173136114_Bentelo14April2022.sgy")