"""
This file contains a wrapper to use a tcp client more easily.
"""
from biosiglive import (
    GenericInterface,
    DeviceType,
)

from pyScienceMode2.rehastim_interface import Stimulator
from typing import Union
from enum import IntEnum
import numpy as np
import time


class DataType(IntEnum):
    Torque = 0
    Angle = 1,
    Speed = 2,
    ALL = -1


class MotomedInterface(GenericInterface):
    """
    Class for interfacing with the client.100
    """
    def __init__(self, port: str = "/dev/ttyUSB0", system_rate: int = 100, motomed_instance=None):
        """
        Initialize the client.

        Parameters
        ----------
        port: int
            Port of the server.
        """
        super(MotomedInterface, self).__init__(system_rate=system_rate)
        self.devices = []
        self.device_cmd_names = []
        self.port = port
        self.motomed = motomed_instance
        self.devices = []
        self._add_motomed()

    def init_motomed(self, arm_training=True, **kwargs):
        self.motomed = Stimulator(self.port, show_log=False, with_motomed=True).motomed
        self.motomed.init_phase_training(arm_training=arm_training)
        self.motomed.start_phase(**kwargs)
    def _add_motomed(
        self,
        nb_channels: int = 1,
        device_type: Union[DeviceType, str] = DeviceType.Generic,
        name: str = None,
        data_buffer_size: int = None,
        rate: float = 100,
    ):
        device_tmp = self._add_device(nb_channels, device_type, name, rate)
        if data_buffer_size:
            device_tmp.data_window = data_buffer_size
        self.devices.append(device_tmp)

    def _get_all_data(self, command: Union[list, DataType] = DataType.ALL):
        if command == DataType.ALL:
            data = [DataType.Torque, DataType.Angle, DataType.Speed]
        else:
            data = np.ndarray((len(command), 1))
        for c, cmd in enumerate(command):
            if DataType.Torque in command:
                data[c, :] = self.motomed.get_torque()
            elif cmd == DataType.Angle:
                data[c, :] = self.motomed.get_angle()
            elif cmd == DataType.Speed:
                data[c, :] = self.motomed.get_speed()
            else:
                raise ValueError("Invalid command.")
        return data

    def get_device_data(self, device_name="all", get_frame = False, data_type = DataType.ALL) -> np.ndarray:
        """
        Get the data from the server.

        Parameters
        ----------
        Returns
        -------
        data: dict
            Data from the server.
        """
        return self._get_all_data(command=data_type)


if __name__ == '__main__':
    interface = MotomedInterface(port="/dev/ttyUSB0", system_rate=100, motomed_instance=None)
    interface.init_motomed(speed=30, gear=5, active=False, go_forward=True, spasm_detection=False)
    while True:
        print(interface.get_device_data(data_type=DataType.ALL))
        time.sleep(1/100)