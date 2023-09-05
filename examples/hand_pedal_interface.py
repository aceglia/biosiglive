"""
This file contains a wrapper to use a tcp client more easily.
"""
from biosiglive import (
    InterfaceType,
    GenericInterface,
    Client,
)
from typing import Union
from enum import IntEnum
import struct
import numpy as np
import time


class DataType(IntEnum):
    Time = 0
    FxLeftPedal = 1,
    FyLeftPedal = 2,
    FzLeftPedal = 3,
    MxLeftPedal = 4,
    MyLeftPedal = 5,
    MzLeftPedal = 6,
    CoPAXLeftPedal = 7,
    CoPAYLeftPedal = 8,
    FxRightPedal = 9,
    FyRightPedal = 10,
    FzRightPedal = 11,
    MxRightPedal = 12,
    MyRightPedal = 13,
    MzRightPedal = 14,
    CoPAXRightPedal = 15,
    CoPAYRightPedal = 16,
    PositionLeftPedal = 17,
    PositionRightPedal = 18,
    CrankAngle = 19,
    TimeBis = 20,
    FxLeftPedalCrank = 21,
    FyLeftPedalCrank = 22,
    FzLeftPedalCrank = 23,
    MxLeftPedalCrank = 24,
    MyLeftPedalCrank = 25,
    MzLeftPedalCrank = 26,
    FxRightPedalCrank = 27,
    FyRightPedalCrank = 28,
    FzRightPedalCrank = 29,
    MxRightPedalCrank = 30,
    MyRightPedalCrank = 31,
    MzRightPedalCrank = 32,
    TorqueLeftPedalCrank = 33,
    TorqueRightPedalCrank = 34,
    TorqueNetPedalCrank = 35,
    SpeedCrank = 36,
    PowerLeftPedalCrank = 37,
    PowerRightPedalCrank = 38,
    PowerNetPedalCrank = 39,
    WorkLeftPedalCrank = 40,
    WorkRightPedalCrank = 41,
    WorkNetPedalCrank = 42,
    IndexEffectivenessLeftPedalCrank = 43,
    IndexEffectivenessRightPedalCrank = 44,
    IndexEffectivenessNetPedalCrank = 45
    AllData = -1
    AllLeftPedal = - 2
    AllRightPedal = -3
    AllPedal = -4
    AllLeftPedalCrank = -5
    AllRightPedalCrank = -6
    AllPedalCrank = -7
    AllCranks = -8


class CustomClient(Client):
    """
    Class for interfacing with the client.100
    """

    def __init__(self, server_ip, port, client_type="TCP"):
        super().__init__(server_ip, port, client_type)
        self.buff_size = 32767

    def connect(self):
        """
        Connect to the server.
        """
        self.client.connect((self.server_address, self.port))

    def _recv_all(self, buff_size: int = None):
        """
        Receive all data from the server and process it to return a tuple.
        Parameters
        ----------
        buff_size: int
            Size of the buffer.
        Returns
        -------
        data: tuple
            Tuple of data received.
        """
        msg_len = self.client.recv(4)
        msg_len = struct.unpack('!i', msg_len)[0]
        data = b''
        length_read = 0
        while length_read < msg_len:
            chunk = self.client.recv(self.buff_size)
            length_read += len(chunk)
            data += chunk
        frmt = '!' + str(int(msg_len)) + 'd'
        data = struct.unpack(frmt, data)
        return data

    def _send_message(self, message: list):
        """
        Convert message to bytes and send it.
        Parameters
        ----------
        message: list
            List that will be sent
        """
        b_message = b''
        for i in range(np.shape(message)[0]):
            for j in range(np.shape(message)[1]):
                b_message += struct.pack('!B', message[i][j])
        b_size = struct.pack('!i', np.shape(message)[0] * np.shape(message)[1] * 1)

        # Send size of the message
        self.client.sendall(b_size)

        # Send message
        self.client.sendall(b_message)

    def get_data(self, message: Union[list, DataType] = DataType.AllData, sample=None, buff: int = None):
        """
        Get the data from server using the command.
        Parameters
        ----------
        message
        buff: int
            Size of the buffer.
        Returns
        -------
        data: list
            Data from server.
        """
        if buff is None:
            buff = self.buff_size
        self._send_message(message)
        if isinstance(message, DataType):
            message = message.value
        if not isinstance(message, list):
            if message < 0:
                message = self._get_message_from_enum(message)
            else:
                message = [message]
        command = []
        for i in range(len(message)):
            for j in range(sample):
                command.append([i, j])
        # self._connect()
        return self._recv_all(buff)

    @staticmethod
    def _get_message_from_enum(message):
        if message == -1:
            return range(0, 46)
        elif message == -2:
            return [0] + list(range(1, 9)) + [17]
        elif message == -3:
            return [0] + list(range(9, 17)) + [18]
        elif message == -4:
            return [0] + list(range(1, 9)) + [17] + list(range(9, 17)) + [18]
        elif message == -5:
            return [0] + list(range(21, 27))+ [33] + [37] + [40] + [43]
        elif message == -6:
            return [0] + list(range(27, 33)) + [34] + [38] + [41] + [44]
        elif message == -7:
            return [0] + list(range(21, 27))+ [33] + [37] + [40] + [43] + list(range(27, 33)) + [34] + [38] + [41] + [44]
        elif message == -8:
            return [0, 19, 35, 36, 39, 42, 45]


class SensixInterface(GenericInterface):
    """
    Class for interfacing with the client.100
    """

    def __init__(self, ip: str = "127.0.0.1", port_tcp: int = 6000, read_frequency: int = 50, from_usb: bool = False):
        """
        Initialize the client.

        Parameters
        ----------
        ip: str
            IP address of the server.
        port: int
            Port of the server.
        client_type: str
            Type of the server.
        read_frequency: int
            Frequency of the reading of the data.
        """
        super(SensixInterface, self).__init__(ip, interface_type=InterfaceType.TcpClient)
        self.devices = []
        self.imu = []
        self.marker_sets = []
        self.read_frequency = read_frequency
        self.ip = ip
        self.port = port_tcp
        self.device_cmd_names = []
        self.last_server_data = None
        self.sample = int(250/read_frequency)
        self.from_usb = from_usb
        if from_usb:
            import nidaqmx
            local_system = nidaqmx.system.System.local()
            driver_version = local_system.driver_version
            print('DAQmx {0}.{1}.{2}'.format(driver_version.major_version, driver_version.minor_version,
                                             driver_version.update_version))
            for device in local_system.devices:
                print('Device Name: {0}, Product Category: {1}, Product Type: {2}'.format(
                    device.name, device.product_category, device.product_type))
            device_name = device.name
            self.task2 = nidaqmx.Task()
            self.task2.ai_channels.add_ai_voltage_chan(device_name + '/ai14')
            self.task2.start()

            self.min_voltage = 1.33
            max_voltage = 5
            self.origin = self.task2.read() - self.min_voltage
            self.angle_coeff = 360 / (max_voltage - self.min_voltage)
        else:
            self.client = CustomClient(server_ip=ip, port=port_tcp, client_type="TCP")
            self.client.connect()

    def get_crank_angle_from_usb(self):
        if not self.from_usb:
            raise Exception("This method is not available if from_usb is False")
        voltage = self.task2.read() - self.min_voltage
        self.actual_voltage = voltage - self.origin
        self.angle = 360 - (self.actual_voltage * self.angle_coeff) if 0 < self.actual_voltage <= 5 - self.origin else \
            abs(self.actual_voltage) * self.angle_coeff
        return self.angle

    def get_data_from_icrankset(self, command: Union[list, DataType] = DataType.AllData) -> np.ndarray:
        """
        Get the data from the server.

        Parameters
        ----------
        command: Union[str, list]
            Command to send to the server.
        Returns
        -------
        data: dict
            Data from the server.
        """
        if self.from_usb:
            raise Exception("This method is not available if from_usb is True")
        self.last_server_data = np.array(self.client.get_data(command, sample=self.sample)).reshape(-1, self.sample)
        return self.last_server_data


if __name__ == '__main__':
    from_usb =False
    freq = 50
    interface = SensixInterface(ip="127.0.0.1", port_tcp=6000, read_frequency=freq, from_usb=from_usb)
    while True:
        if from_usb:
            print(interface.get_crank_angle_from_usb())
        else:
            print(interface.get_data_from_icrankset(command=DataType.CrankAngle))
        time.sleep(1/freq)