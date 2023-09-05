"""
This file contains a class that allows to stream data from a source and start some multiprocess
 to process and disseminate. It is a work in progress so basic functions are available but need to be fine-tuned.
"""

from typing import Union
from time import time, sleep, strftime
import datetime
import numpy as np
import multiprocessing as mp
from biosiglive.streaming.server import Server
from ..file_io.save_and_load import save
from ..interfaces.generic_interface import GenericInterface
from ..interfaces.param import Device, MarkerSet
from ..gui.plot import LivePlot
from .utils import dic_merger


# TODO add enum for command type
class StreamData:
    def __init__(self, stream_rate: int = 100):
        """
        Initialize the StreamData class.
        Careful this class do not return anything, you will have to turn the save option to True to save the data.

        Parameters
        ----------
        stream_rate: int
            The stream rate of the data.
        """
        self.process = mp.Process
        self.devices = []
        self.marker_sets = []
        self.plots = []
        self.stream_rate = stream_rate
        self.interfaces_type = []
        self.processes = []
        self.interfaces = []
        self.multiprocess_started = False
        self.main_interface_idx = 0

        # Multiprocessing stuff
        manager = mp.Manager()
        self.device_queue_in = []
        self.device_queue_out = []
        self.kin_queue_in = []
        self.kin_queue_out = []
        self.plots_queue = []
        self.device_event = []
        self.is_device_data = []
        self.is_kin_data = []
        self.is_interface_data = []
        self.interfaces_data_queue = []
        self.interface_event = []
        self.kin_event = []
        self.custom_processes = []
        self.custom_processes_kwargs = []
        self.custom_processes_names = []
        self.custom_queue_in = []
        self.custom_queue_out = []
        self.custom_event = []
        self.save_data = None
        self.save_path = None
        self.save_frequency = None
        self.plots_multiprocess = False
        self.device_buffer_size = []
        self.marker_set_buffer_size = []
        self.raw_plot = None
        self.data_to_plot = None

        # Server stuff
        self.start_server = None
        self.server_ip = None
        self.ports = []
        self.client_type = None
        self.count_server = 0
        self.server_queue = []
        self.device_decimals = 8
        self.kin_decimals = 6

    def _add_device(self, device: Device, interface_idx: int):
        """
        Add a device to the stream.

        Parameters
        ----------
        device: Device
            Device to add.
        interface_idx: int
            The index of the interface to which the device is added.
        """
        self.devices[interface_idx].append(device)
        self.device_queue_in[interface_idx].append(mp.Manager().Queue())
        self.device_queue_out[interface_idx].append(mp.Manager().Queue())
        self.device_event[interface_idx].append(mp.Manager().Event())

    def add_interface(self, interface: GenericInterface()):
        """
        Add an interface to the stream.

        Parameters
        ----------
        interface: GenericInterface
            Interface to add. Interface should inherit from the generic interface.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        if len(self.interfaces) != 0:
            if interface.system_rate > self.interfaces[0].system_rate :
                self.main_interface_idx += 1
        self.interfaces.append(interface)
        interface_idx = len(self.interfaces) - 1
        self.interfaces_type.append(interface.interface_type)
        self.interface_event.append(mp.Manager().Event())
        self.devices.append([])
        self.device_queue_in.append([])
        self.device_queue_out.append([])
        self.device_event.append([])
        self.marker_sets.append([])
        self.kin_queue_in.append([])
        self.kin_queue_out.append([])
        self.kin_event.append([])
        self.interfaces_data_queue.append(mp.Manager().Queue())
        self.is_kin_data.append(mp.Manager().Event())
        self.is_device_data.append(mp.Manager().Event())
        for device in interface.devices:
            self._add_device(device, interface_idx)
        for marker in interface.marker_sets:
            self._add_marker_set(marker, interface_idx)
        if len(self.interfaces) > 2:
            raise ValueError("Only two interfaces can be added for now.")

    def add_server(
        self,
        server_ip: str = "127.0.0.1",
        ports: Union[int, list] = 50000,
        client_type: str = "TCP",
        device_buffer_size: Union[int, list] = None,
        marker_set_buffer_size: [int, list] = None,
        save_data: bool = False,
        save_path: str = None,
    ):
        """
        Add a server to the stream.

        Parameters
        ----------
        server_ip: str
            The ip address of the server.
        ports: int or list
            The port(s) of the server.
        client_type: str
            The type of client to use. Can be TCP.
        device_buffer_size: int or list
            The size of the buffer for the devices.
        marker_set_buffer_size: int or list
            The size of the buffer for the marker sets.
        save_data: bool
            If True, the data will be saved. Here it will save the data sent to the server.
        save_path: str
            The path to save the data.
        """
        if self.multiprocess_started:
            raise Exception("Cannot add interface after the stream has started.")
        self.server_ip = server_ip
        self.ports = ports
        if not isinstance(self.ports, list):
            self.ports = [self.ports]

        for p in range(len(self.ports)):
            self.server_queue.append(mp.Manager().Queue())
        self.client_type = client_type

        if not device_buffer_size:
            device_buffer_size = [None] * len(self.devices)
        if isinstance(device_buffer_size, list):
            if len(device_buffer_size) != len(self.devices):
                raise ValueError("The device buffer size list should have the same length as the number of devices.")
            self.device_buffer_size = device_buffer_size
        elif isinstance(device_buffer_size, int):
            self.device_buffer_size = [device_buffer_size] * len(self.devices)

        if not marker_set_buffer_size:
            marker_set_buffer_size = [None] * len(self.marker_sets)
        if isinstance(marker_set_buffer_size, list):
            if len(marker_set_buffer_size) != len(self.marker_sets):
                raise ValueError(
                    "The marker set buffer size list should have the same length as the number of marker sets."
                )
            self.marker_set_buffer_size = marker_set_buffer_size
        elif isinstance(marker_set_buffer_size, int):
            self.marker_set_buffer_size = [marker_set_buffer_size] * len(self.marker_sets)
        if len(self.ports) > 1:
            raise ValueError("Only one server can be added for now.")

    def start(self, save_streamed_data: bool = False, save_path: str = None, save_frequency: int = None):
        """
        Start the stream.

        Parameters
        ----------
        save_streamed_data: bool
            If True, the streamed data will be saved.
        save_path: str
            The path to save the streamed data.
        save_frequency:
            The frequency at which the data will be saved.
        """
        self.save_data = save_streamed_data
        self.save_path = save_path if save_path else f"streamed_data_{strftime('%Y%m%d_%H%M%S')}.bio"
        self.save_frequency = save_frequency if save_frequency else self.stream_rate
        self._init_multiprocessing()

    def _add_marker_set(self, marker: MarkerSet, interface_idx: int):
        """
        Add a marker set to the stream.

        Parameters
        ----------
        marker: MarkerSet
            Marker set to add from given interface.
        interface_idx: int
            The index of the interface to which the marker set is added.
        """
        self.marker_sets[interface_idx].append(marker)
        self.kin_queue_in[interface_idx].append(mp.Manager().Queue())
        self.kin_queue_out[interface_idx].append(mp.Manager().Queue())
        self.kin_event[interface_idx].append(mp.Manager().Event())

    # TODO : add buffer directly in the server
    def device_processing(self, device_idx: int, interface_idx: int):
        """
        Process the data from the device

        Parameters
        ----------
        device_idx: int
            The index of the device in the list of devices.
        interface_idx: int
            The index of the interface in the list of interfaces.
        """
        if self.device_buffer_size:
            if not self.device_buffer_size[device_idx]:
                self.device_buffer_size[device_idx] = self.devices[interface_idx][device_idx].rate
            buffer_size = self.device_buffer_size[device_idx]
        else:
            buffer_size = self.devices[interface_idx][device_idx].rate
        while True:
            self.is_device_data[interface_idx].wait()
            device_data = self._get_from_queue(self.device_queue_in[interface_idx][device_idx])
            if device_data is not None:
                self.devices[interface_idx][device_idx].new_data = device_data
                self.devices[interface_idx][device_idx].append_data(device_data)
                processed_data = self.devices[interface_idx][device_idx].process(**self.devices[interface_idx][device_idx].processing_method_kwargs)
                self.device_queue_out[interface_idx][device_idx].put_nowait({"processed_data": processed_data[:, -buffer_size:]})
                self.device_event[interface_idx][device_idx].set()
            sleep(0.001)

    def recons_kin(self, marker_set_idx: int, interface_idx: int):
        """
        Compute inverse kinematics from markers.

        Parameters
        ----------
        marker_set_idx: int
            Index of the marker set in the list of markers.
        interface_idx: int
            Index of the interface in the list of interfaces.
        """
        if self.marker_set_buffer_size:
            if not self.marker_set_buffer_size[marker_set_idx]:
                self.marker_set_buffer_size[marker_set_idx] = self.marker_sets[interface_idx][marker_set_idx].rate
            buffer_size = self.marker_set_buffer_size[marker_set_idx]
        else:
            buffer_size = self.marker_sets[interface_idx][marker_set_idx].rate
        if "model_path" not in self.marker_sets[interface_idx][marker_set_idx].kin_method_kwargs.keys():
            raise ValueError("No model to compute the kinematics.")
        while True:
            self.is_kin_data[interface_idx].wait()
            markers = self._get_from_queue(self.kin_queue_in[interface_idx][marker_set_idx])
            if markers is not None:
                self.marker_sets[interface_idx][marker_set_idx].new_data = markers
                self.marker_sets[interface_idx][marker_set_idx].append_data(markers)
                states, _ = self.marker_sets[interface_idx][marker_set_idx].get_kinematics(
                    **self.marker_sets[interface_idx][marker_set_idx].kin_method_kwargs
                )
                self.kin_queue_out[interface_idx][marker_set_idx].put_nowait({"kinematics_data": states[:, -buffer_size:]})
                self.kin_event[interface_idx][marker_set_idx].set()
            sleep(0.001)

    def open_server(self, server_idx: int):
        """
        Open the server to send data from the devices.

        Parameters
        ----------
        server_idx: int
            The index of the server in the list of servers.
        """
        server = Server(self.server_ip, self.ports[server_idx], server_type=self.client_type)
        server.start()
        while True:
            connection, message = server.client_listening()
            # use Try statement as the queue can be empty and is_empty function is not reliable.
            data_queue = self._get_from_queue(self.server_queue[server_idx])
            if data_queue:  # use this method to avoid blocking the server with Windows os.
                data_queue.append(data_queue)
                server.send_data(data_queue, connection, message)
            
    def _init_multiprocessing(self):
        """
        Initialize the multiprocessing.
        """
        processes = []
        for i in range(len(self.interfaces)):
            processes.append(
                self.process(name="reader", target=StreamData.save_streamed_data, args=(self, i), daemon=True)
            )

            for d, device in enumerate(self.devices[i]):
                if device.processing_method is not None:
                    processes.append(
                        self.process(
                            name=f"process_{device.name}",
                            target=StreamData.device_processing,
                            args=(
                                self,
                                d,
                                i,
                            ),
                            daemon=True,
                        )
                    )
            for m, marker in enumerate(self.marker_sets[i]):
                if marker.kin_method:
                    processes.append(
                        self.process(
                            name=f"process_{marker.name}",
                            target=StreamData.recons_kin,
                            args=(
                                self,
                                m,
                                i,
                            ),
                            daemon=True,
                        )
                    )
        for j in range(len(self.ports)):
            processes.append(
                self.process(name="listen" + f"_{j}", target=StreamData.open_server, args=(self, j), daemon=True)
            )

        for p, plot in enumerate(self.plots):
            for device in self.devices:
                for marker_set in self.marker_sets:
                    if self.data_to_plot[p] not in device.name and self.data_to_plot[p] not in marker_set.name:
                        raise ValueError(f"The name of the data to plot ({self.data_to_plot[p]}) is not correct.")
            if self.plots_multiprocess:
                processes.append(self.process(name="plot", target=StreamData.plot_update, args=(self, p), daemon=True))
            else:
                processes.append(self.process(name="plot", target=StreamData.plot_update, args=(self, -1), daemon=True))
                break

        for i, funct in enumerate(self.custom_processes):
            processes.append(
                self.process(
                    name=self.custom_processes_names[i],
                    target=funct,
                    args=(self,),
                    kwargs=self.custom_processes_kwargs[i],
                    daemon=True,
                )
            )
        for p in processes:
            p.start()
        self.multiprocess_started = True
        for p in processes:
            p.join()

    def _check_nb_processes(self):
        """
        compute the number of process.
        """
        nb_processes = 0
        for device in self.devices:
            if device.process_method is not None:
                nb_processes += 1
        if self.start_server:
            nb_processes += len(self.ports)
        nb_processes += len(self.plots)
        nb_processes += len(self.interfaces)
        for marker in self.marker_sets:
            if marker.kin_method:
                nb_processes += 1
        nb_processes += len(self.custom_processes)
        return nb_processes

    def add_plot(
        self,
        plot: Union[LivePlot, list],
        data_to_plot: Union[str, list],
        raw: Union[bool, list] = None,
        multiprocess=False,
    ):
        """
        Add a plot to the live data. Still Not working for now.

        Parameters
        ----------
        plot: Union[LivePlot, list]
            Plot to add.
        data_to_plot: Union[str, list]
            Name of the data to plot.
        raw: Union[bool, list]
            If True, the raw data will be plotted.
        multiprocess: bool
            If True, if several plot each plot will be on a separate process. If False, each plot will be on the same one.
        """
        raise NotImplementedError("Plot are not implemented yet with StreamData class.")
        # if isinstance(data_to_plot, str):
        #     data_to_plot = [data_to_plot]
        # if isinstance(raw, bool):
        #     raw = [raw]
        # if len(data_to_plot) != len(raw):
        #     raise ValueError("The length of the data to plot and the raw list must be the same.")
        # if not raw:
        #     raw = [True] * len(data_to_plot)
        # self.plots_queue.append(mp.Manager().Queue())
        # self.raw_plot = raw
        # self.data_to_plot = data_to_plot
        # if self.multiprocess_started:
        #     raise Exception("Cannot add plot after the stream has started.")
        # self.plots_multiprocess = multiprocess
        # if not isinstance(plot, list):
        #     plot = [plot]
        # for plt in plot:
        #     if plt.rate:
        #         if plt.rate > self.stream_rate:
        #             raise ValueError("Plot rate cannot be higher than stream rate.")
        #     self.plots.append(plt)

    def plot_update(self, plot_idx: int = -1):
        """
        Update the plots.

        Parameters
        ----------
        plot_idx: int
            index of the plot to update. If -1, all plots will be updated.
        """
        if plot_idx == -1:
            plots = self.plots
            queue = self.plots_queue[0]
        else:
            plots = self.plots[plot_idx]
            queue = self.plots_queue[plot_idx]
        data_to_plot = []
        data = None
        device_names = []
        marker_set_names = []
        for device in self.devices:
            device_names.append(device.name)
        for marker in self.marker_sets:
            marker_set_names.append(marker.name)
        while True:
            try:
                data = queue.get_nowait()
                is_working = True
            except Exception:
                is_working = False
            if is_working:
                for p, plot in enumerate(plots):
                    if self.data_to_plot[p] in device_names:
                        if not self.raw_plot[p]:
                            data_to_plot = data["proc_device_data"][device_names.index(self.data_to_plot[p])]
                        else:
                            data_to_plot = data["raw_device_data"][device_names.index(self.data_to_plot[p])]
                    if self.data_to_plot[p] in marker_set_names:
                        if not self.raw_plot[p]:
                            data_to_plot = data["kinematics_data"][marker_set_names.index(self.data_to_plot[p])]
                        else:
                            data_to_plot = data["marker_set_data"][marker_set_names.index(self.data_to_plot[p])][
                                :, :, -1
                            ].T
                    plot.update(data_to_plot)

    def save_streamed_data(self, interface_idx: int):
        """
        Stream, process and save the data.

        Parameters
        ----------
        interface_idx: idx
            Interface index to use from the interface list. for now only one interface is supported.

        """
        initial_time = 0
        iteration = 0
        save_count = [0] * len(self.interfaces)
        self.save_frequency = self.save_frequency if self.save_frequency else self.stream_rate
        interface = self.interfaces[interface_idx]
        saving_time = 0
        frame_number = None
        last_frame_number = None
        lost_frames = []
        last_frame_number_others = [0] * len(self.interfaces)
        while True:
            data_dic = {}
            dic_to_save = {}
            data_dic_all = [{}] * len(self.interfaces)
            proc_device_data = []
            raw_device_data = []
            raw_markers_data = []
            all_device_data = []
            all_markers_tmp = []
            kin_data = []
            tic = time()
            data_other_int = {}
            if iteration == 0:
                initial_time = time() - tic
            interface_latency = interface.get_latency()
            is_frame = interface.get_frame()
            if iteration == 0:
                frame_number = interface.get_frame_number()
                frame_number = 0 if not frame_number else frame_number
            else:
                last_frame_number = interface.get_frame_number()
            if not last_frame_number:
                last_frame_number = frame_number + 1
            if last_frame_number == frame_number:
                pass
            elif last_frame_number > frame_number + 1:
                lost_frames.append(last_frame_number)
            else:
                frame_number = last_frame_number
                absolute_time_frame = datetime.datetime.now()
                absolute_time_frame_dic = {
                    "day": absolute_time_frame.day,
                    "hour": absolute_time_frame.hour,
                    "hour_s": absolute_time_frame.hour * 3600,
                    "minute": absolute_time_frame.minute,
                    "minute_s": absolute_time_frame.minute * 60,
                    "second": absolute_time_frame.second,
                    "millisecond": int(absolute_time_frame.microsecond / 1000),
                    "millisecond_s": int(absolute_time_frame.microsecond / 1000) * 0.001,
                }
                self.is_kin_data[interface_idx].clear()
                self.is_device_data[interface_idx].clear()

                if is_frame:
                    if iteration == 0:
                        print("Data start streaming")
                        iteration = 1
                    if len(interface.devices) != 0:
                        all_device_data = interface.get_device_data(device_name="all", get_frame=False)
                        if not isinstance(all_device_data, list):
                            all_device_data = [all_device_data]
                        self.is_device_data[interface_idx].set()
                        for i in range(len(all_device_data)):
                            if self.devices[interface_idx][i].processing_method is not None:
                                self.device_queue_in[interface_idx][i].put_nowait(all_device_data[i])

                    if len(interface.marker_sets) != 0:
                        all_markers_tmp, _ = interface.get_marker_set_data(get_frame=False)
                        if not isinstance(all_markers_tmp, list):
                            all_markers_tmp = [all_markers_tmp]
                        self.is_kin_data[interface_idx].set()
                        for i in range(len(self.marker_sets[interface_idx])):
                            if self.marker_sets[interface_idx][i].kin_method is not None:
                                self.kin_queue_in[interface_idx][i].put_nowait(all_markers_tmp[i])
                    time_to_get_data = time() - tic

                    tic_process = time()
                    if len(interface.devices) != 0:
                        for i in range(len(interface.devices)):
                            if self.devices[interface_idx][i].processing_method is not None:
                                self.device_event[interface_idx][i].wait()
                                device_data = self.device_queue_out[interface_idx][i].get_nowait()
                                self.device_event[interface_idx][i].clear()
                                proc_device_data.append(
                                    np.around(device_data["processed_data"], decimals=self.device_decimals)
                                )
                            raw_device_data.append(np.around(all_device_data[i], decimals=self.device_decimals))
                        data_dic["proc_device_data"] = proc_device_data
                        data_dic["raw_device_data"] = raw_device_data

                    if len(interface.marker_sets) != 0:
                        for i in range(len(interface.marker_sets)):
                            if self.marker_sets[interface_idx][i].kin_method is not None:
                                self.kin_event[interface_idx][i].wait()
                                kin_data_proc = self.kin_queue_out[interface_idx][i].get_nowait()
                                self.kin_event[interface_idx][i].clear()
                                kin_data.append(np.around(kin_data_proc["kinematics_data"], decimals=self.kin_decimals))
                            raw_markers_data.append(np.around(all_markers_tmp[i], decimals=self.kin_decimals))
                        data_dic["kinematics_data"] = kin_data
                        data_dic["marker_set_data"] = raw_markers_data
                    process_time = time() - tic_process

                    if interface_idx != self.main_interface_idx:
                        dic_to_put = data_dic
                        dic_to_put["absolute_time_frame"] = absolute_time_frame_dic
                        dic_to_put["interface_latency"] = interface_latency
                        dic_to_put["process_time"] = process_time
                        dic_to_put["initial_time"] = initial_time
                        dic_to_put["time_to_get_data"] = time_to_get_data
                        dic_to_put["frame_number"] = frame_number
                        dic_to_put["lost_frames"] = lost_frames
                        dic_to_put["saving_time"] = saving_time
                        data_dic_all[interface_idx] = dic_to_put
                        self._put_in_queue(self.interfaces_data_queue[interface_idx], dic_to_put)
                    else:
                        data_dic_all[interface_idx] = data_dic
                        data_dic_all[interface_idx]["absolute_time_frame"] = absolute_time_frame_dic
                        data_dic_all[interface_idx]["interface_latency"] = interface_latency
                        data_dic_all[interface_idx]["process_time"] = process_time
                        data_dic_all[interface_idx]["initial_time"] = initial_time
                        data_dic_all[interface_idx]["time_to_get_data"] = time_to_get_data
                        data_dic_all[interface_idx]["frame_number"] = frame_number
                        data_dic_all[interface_idx]["lost_frames"] = lost_frames
                        data_dic_all[interface_idx]["saving_time"] = saving_time
                        for i in range(len(self.interfaces)):
                            if i != self.main_interface_idx:
                                data_tmp = self._get_from_queue(self.interfaces_data_queue[i], timeout=0.006)
                                if data_tmp is not None:
                                    if data_tmp["frame_number"] != last_frame_number_others[i] + 1:
                                        data_tmp["lost_frames"] = list(range(last_frame_number_others[i] + 1, data_tmp["frame_number"]))
                                        if len(data_tmp["lost_frames"]) == 0:
                                            data_tmp["lost_frames"] = data_tmp["frame_number"] - last_frame_number_others[i]
                                        print("Lost frames: ", data_tmp["lost_frames"])
                                    last_frame_number_others[i] = data_tmp["frame_number"]
                                    data_dic_all[i] = data_tmp

                    if interface_idx == self.main_interface_idx:
                        for i in range(len(self.ports)):
                            self._put_in_queue(self.server_queue[i], data_dic)

                        if len(self.plots) != 0:
                            size = 1 if not self.plots_multiprocess else len(self.plots)
                            for i in range(size):
                                self._put_in_queue(self.plots_queue[i], data_dic)

                    # Save data
                    if self.save_data is True:
                        tic_save = time()
                        # if save_count[interface_idx] == int(self.interfaces[interface_idx].system_rate / self.save_frequency):
                        path = self.save_path + str(interface_idx)
                        if interface_idx == self.main_interface_idx:
                            for i in range(len(self.interfaces)):
                                #dic_to_save[f"interface_{interface_idx}"] = dic_merger(data_dic, dic_to_save[interface_idx])
                                dic_to_save[f"interface_{i}"] = data_dic_all[i]
                            save(dic_to_save, path)
                        else:
                            save(data_dic_all[interface_idx], path)
                        save_count[interface_idx] = 0
                        save_count[interface_idx] += 1
                        saving_time = time() - tic_save

                    time_tmp = time() - tic + 0.0001
                    if time_tmp < 1 / self.interfaces[interface_idx].system_rate:
                        sleep(1 / self.interfaces[interface_idx].system_rate - time_tmp)
                    elif time_tmp - 0.0005 > 1 / self.interfaces[interface_idx].system_rate:
                        print(
                            f"WARNING: Stream rate ({self.interfaces[interface_idx].system_rate})"
                            f" is too high for the computer."
                            f"The actual stream rate is {1 / (time_tmp - 0.0001)}"
                        )
                    # if interface_idx == self.main_interface_idx:
                    #     print(1/(time() - tic))
                    iteration += 1

    @staticmethod
    def _put_in_queue(queue: mp.Queue, data: dict, keys: list = None) -> None:
        """
        Put data in a queue

        Parameters
        ----------
        queue : mp.Queue
            Queue to put data in
        data : dict
            Data to put in the queue
        keys : list
            Keys of the data to put in the queue
        """
        if keys:
            for key in data.keys():
                if key not in keys:
                    del data[key]
        try:
            queue.get_nowait()
        except Exception:
            pass
        queue.put_nowait(data)

    @staticmethod
    def _get_from_queue(queue: mp.Queue, timeout: float = 0) -> dict or None:
        """
        Get data from a queue

        Parameters
        ----------
        queue : mp.Queue
            Queue to get data from
        timeout : float
            Timeout of the queue

        Returns
        -------
        dict
            Data from the queue if data is available, else None
        """
        try:
            data = queue.get(timeout=timeout)
            return data
        except Exception:
            pass

    def stop(self):
        """
        Stop the stream
        """
        for process in self.processes:
            process.terminate()
            process.join()
