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
from obspy import read

file_path = '20240625132506554_TestGraaf22.25.sgy'

# Open the SEGY file using ObsPy
try:
    # Read the SEGY file using ObsPy
    segy_stream = read(file_path, format="SEGY")

    # Print some basic info about the traces
    print(f"Number of traces: {len(segy_stream)}")
    print(f"Sample points per trace: {len(segy_stream[0].data)}")

    # Plot the first trace
    segy_stream[0].plot()

    # print array of data
    print(segy_stream[0].data)
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