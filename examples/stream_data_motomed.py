"""
This example show how to use the StreamData class to stream data from an interface, process it using several process to
finally send the data through a server to a client. Each task in a separate process to allow the streaming and the
 processing to be done in real-time. Please note that for now only a number equal to the number of cores of the computer
  is supported.
First an interface is created and device and marker_set are added to it (please refer to EMG_streming.py and
marker_streaming.py for more details).
Then a StreamData object is created. The StreamData object takes as argument the targeted frequency at which the data will
be streamed. Then the interface is added to the StreamData object. If the user want to start a server to
disseminate the data a server can be added to the StreamData object specifying the ip address and the port and the
data buffer for the device and the marker set. The data buffer is the number of frame that will be stored in the server,
it will be use if the client need a specific amount of data.
Then the streaming will be started with all the data streaming, processing and the server in seperate process. If no
 processing method is specified the data will be streamed as it is and no additional process will be started. A file can
  be specified to save the data. The data will be saved in a *.bio file at each loop of the data streaming by default or
  at the save frequency specified in the start method.
Please note that it is not yet possible to plot the data in real-time.
"""
from motomed_interface import MotomedInterface
from biosiglive import (
    ViconClient,
    StreamData,
    DeviceType,
)

try:
    import biorbd
except ModuleNotFoundError:
    biorbd_package = False

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    vicon_package = False


if __name__ == "__main__":
    server_ip = "127.0.0.1"
    server_port = 50000
    interface = ViconClient(system_rate=100)
    interface_motomed = MotomedInterface(port="/dev/ttyUSB0", system_rate=100, motomed_instance=None)
    interface_motomed.init_motomed(speed=30, gear=5, active=False, go_forward=True, spasm_detection=False)
    nb_channels_stim = 8
    interface.add_device(
        name="Stim",
        device_type=DeviceType.Emg,
        rate=10000,
        nb_channels=nb_channels_stim,
        data_buffer_size=1000,
    )
    data_streaming = StreamData(stream_rate=100)
    data_streaming.add_interface(interface)
    data_streaming.add_interface(interface_motomed)
    data_streaming.start(save_streamed_data=True, save_path="data_streamed_motomed")
