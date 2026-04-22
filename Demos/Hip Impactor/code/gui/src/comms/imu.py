import serial
import serial.tools.list_ports
from multiprocessing import Process, Queue
import threading
from enum import Enum
import socket
import time
from commons.filters import Madgwick, ComplementaryFilter
import numpy as np
from comms.utl import Cluster


class IMUTools(object):
    def __init__(self):
        self.daport = None

    @staticmethod
    def search():
        ports = serial.tools.list_ports.comports()
        daport = []
        for i in range(0, len(ports)):
            print("device: "+str(ports[i].device))
            print("name: "+str(ports[i].name))
            print("description: "+str(ports[i].description))
            print("hwid: "+str(ports[i].hwid))
            print("vid: "+str(ports[i].vid))
            print("pid: "+str(ports[i].pid))
            print("serial_number: "+str(ports[i].serial_number))
            print("location: "+str(ports[i].location))
            print("manufacturer: "+str(ports[i].manufacturer))
            print("product: "+str(ports[i].product))
            print("interface: "+str(ports[i].interface))

            if "9D0F" in str(ports[i].hwid):
                daport.append(ports[i].device)
            if "USB VID:PID=10C4:EA60 SER=0001" in str(ports[i].hwid):
                print("hardcode know IMU")
                daport.append([ports[i].device, ports[i]])
            if "USB VID:PID=10C4:EA60 SER=01DAFFC8" in str(ports[i].hwid):
                print("hardcode known IMU Base")
                daport.append([ports[i].device, ports[i]])
            print()

        if len(daport) > 0:
            print("IMU found at " + str(daport))
        else:
            print("IMU no found")
        return daport


def io_processor(port, data_que, command_que):
    porta = serial.Serial(port, 115200, timeout=0.01)
    command = ImuCommands.run
    while command is ImuCommands.run:
        if command_que.qsize() > 0:
            command = command_que.get()
        try:
            e = porta.readline()
            if 250 > len(e) > 216:
                data_que.put(e)
        except serial.serialutil.SerialException:
            print("Input Error")
            command_que.put(ImuCommands.USBerror)
            break
    porta.close()
    data_que.put("Stopping")
    print("io_processor stopped")


def printlog(s):
    print(s)


class ImuFromServer(object):
    def __init__(self):
        self.server = None
        self.connect2sever()
        self.processor = None
        self.stopppppppp = False
        self.imu_fusion = ComplementaryFilter()
        self.data_que = Queue()
        self.command_que = Queue()

    def connect2sever(self):
        try:
            self.server = socket.socket()  # Create a socket object
            host = socket.gethostname()  # Get local machine name
            port = 12345  # Reserve a port for your service.

            self.server.connect((host, port))
            self.server.send(b'\nvi\n')
            print(self.server.recv(1024))
        except ConnectionRefusedError:
            pass

    def read_imu(self):
        print("imu_read")
        if self.command_que.qsize() > 0:
            if self.command_que.get() is ImuCommands.USBerror:
                print("Usb read error ... exiting ...")
                return None
        print("Get data, queue size: "+str(self.data_que.qsize()))
        if self.data_que.qsize() > 0:
            data = self.data_que.get()
            ret = {"time": data[0],
                   "eul": [int(data[1]), int(data[2]), int(data[3])]
                   }
            return ret
        else:
            print("End Read")
            return None

    def imu_server(self):
        while (1):
            if self.stopppppppp:
                break
            try:
                self.server.send(b"\nvi\n")
                databyte = self.server.recv(1024)
                data = databyte.decode("utf-8")
                data_line = data.split('\n')
                for l in data_line:
                    d = l.split(',')
                    if len(d) > 12:
                        e = []
                        for da in d:
                            try:
                                e.append(float(da))
                            except ValueError:
                                pass
                        if len(e) == 10:
                            acc = e[1:4]
                            gyr = e[4:7]
                            mag = e[7:]
                            self.imu_fusion.update(gyro=gyr, acc=acc)
                            imu_euler = self.imu_fusion.get_euler_earth()
                            #imu_euler = self.imu_fusion.madgwick_euler_earth()
                            ret = [e[0], imu_euler[0], imu_euler[1], imu_euler[2]]

                            self.data_que.put(ret)
                        print(e)
            except OSError:
                print("broke")
                break
            time.sleep(0.001)

    def run(self):
        self.processor = threading.Thread(target=self.imu_server, args=())
        self.processor.start()

    def stop_imucoms(self):
        self.stopppppppp = True


class Imuio(object):
    def __init__(self, comms, name="pooh"):
        self.porta = comms
        self.data_que = Queue()
        self.command_que = Queue()
        self.processor = None
        self.status = 1
        self.started = False
        self.gui = None

    def start(self):
        self.started = True
        if self.processor is None:
            self.start2()
            # self.processor = Process(target=io_processor, args=(self.porta, self.data_que, self.command_que))
            # self.processor.start()

    def start2(self):
        self.started = True
        if self.processor is None:
            self.processor = threading.Thread(target=io_processor, args=(self.porta, self.data_que, self.command_que))
            self.processor.start()

    def stop_imucoms(self):
        self.started = False
        print("stop_comms")
        self.command_que.put(ImuCommands.stop)

    def read_imu(self):
        if self.command_que.qsize() > 0:
            if self.command_que.get() is ImuCommands.USBerror:
                print("Usb read error ... exiting ...")
                return None
        data = self.data_que.get()
        read = data.decode("utf-8")
        reading = read.strip().split(",")
        if len(reading) == 7:
            ret = {"time": int(reading[0]),
                   "acc": [int(reading[1]), int(reading[2]), int(reading[3])],
                   "eul": [int(reading[4]), int(reading[5]), int(reading[6])]
                   }
            return ret
        else:
            ret = {}
            return ret


class ImuCommands(Enum):
    run = 1
    stop = 0
    USBerror = -1


class BaseSens(Imuio):
    def __init__(self, port, gui=None):
        super().__init__(port)
        print("Base-Station Connected connected at port: "+port)
        self.count = []
        self.gui = gui
        self.previous = np.array([0, 0, 0])
        self.prev_adj = 0
        self.offset = 0
        self.pt = 0
        self.prev_b = 0
        self.prev_yaw = {'1': 0, '2': 0, '3': 0}
        self.prev = None
        self.gui_note = False
        # self.start()

    def notify_gui(self):
        self.gui.connected = True
        self.gui.update()

    def read_imu(self, gui=True):
        if self.started:
            if gui and not self.gui_note:
                self.gui_note = False
                f = threading.Thread(target=self.notify_gui, args=())
                f.start()
                f.join()
            if self.command_que.qsize() > 0:
                if self.command_que.get() is ImuCommands.USBerror:
                    print("Usb read error ... exiting ...")
                    return None
            data = self.data_que.get()
            # c = threading.Thread(target=printlog, args=([data]))
            # c.start()
            # c.join()
            if isinstance(data, str):
                if data.lower() == "stopping":
                    return
            reader = data.decode("utf-8")
            reading = reader.strip().split(",")
            c = threading.Thread(target=printlog, args=([reading]))
            c.start()
            c.join()
            imus = Cluster.extract(reading)
            if imus is None:
                return None
            # print(imus)
            g = 9.80665
            a = None
            if len(reading) == 28:

                imu2_acc = [2 * (int(reading[20]) / 32767.0),
                            2 * (int(reading[21]) / 32767.0),
                            2 * (int(reading[22]) / 32767.0)]

                if len(self.count) < 200:
                    self.count.append(imu2_acc)
                else:
                    if a is None:
                        a = np.nanmean(self.count, axis=0)-np.array([0, 0, 1])
                    imu2_acc = np.array(imu2_acc)-a

                imu1_acc = BaseSens.raw_acc_to_float(g, reading[12:15])
                ret = {}
                bs = np.array([2 * int(reading[8]) / 32767.0,
                               2 * int(reading[9]) / 32767.0,
                               2 * int(reading[10]) / 32767.0])

                bsg = np.array([int(reading[15])/131, int(reading[16])/131, int(reading[17])/131])
            try:
                #for c in imus:
                ret = imus
                try:
                    if ret['c'] == 'bytes' or ret['c'] == '84':
                        return None
                except KeyError:
                    return None
                # ret = {"time": float(reading[0]),
                #        "hz": float(reading[1]),
                #        "IMU_boss": {"ypr": [float(reading[4]), float(reading[5]), float(reading[6])],
                #                     "acc": bs,
                #                     "norm": np.linalg.norm(np.array(bs))},
                #        "IMU_helper1": {"acc": imu1_acc,
                #                        "gyr": bsg,
                #                        "norm": np.linalg.norm(bsg)
                #                        },
                #        "IMU_helper2": {"acc": g*np.array(imu2_acc),
                #                        "gyr": [int(reading[24])/131,
                #                                int(reading[25])/131,
                #                                int(reading[26])/131],
                #                        "norm": np.linalg.norm(g*np.array(imu2_acc)),
                #                        }
                #        }
                try:
                    xmp = ret["imu1_ypr"]
                except KeyError:
                    print(ret)
                    return None

                if xmp[0] < self.previous[0]:
                    self.offset = 1
                elif xmp[0] > self.previous[0]:
                    self.offset = -1
                if abs(self.pt - ret["t"]) > 5*(1/ret["hz"]):
                    # print(str(ret["t"]) + " ||| " + str(ret["imu1_ypr"]) + " ||| " + str(ret["imu1_acc"]))
                    # print(self.prev)
                    # print(read)
                    ret["imu1_ypr"][0] = self.prev_yaw[ret['c']]
                else:
                    if ret["t"] > 10:
                        t = ret["t"]-self.pt
                        c = self.prev_adj + self.offset*(0.0051*t)
                        al = 0.9982
                        a = al * (ret["imu1_ypr"][0] - c) + (1 - al) * self.prev_yaw[ret['c']]
                        if ret["imu2_gyr_norm"] < 1.5:
                            a = self.prev_yaw[ret['c']]
                        self.prev_adj = c
                        self.previous = xmp
                        self.prev = ret
                        self.prev_yaw[ret['c']] = a
                        ret["imu1_ypr"][0] = a
                    else:
                        self.previous = xmp
                        self.prev_yaw[ret['c']] = xmp[0]
                        self.prev = ret

                self.pt = ret["t"]
                # if ret["time"] > 20:
                #     print(str(ret["time"])+","+str(ret["imu1_ypr"][0]))
                # print(self.previous[0])
                return ret
            except ValueError:
                pass

        return None

    @staticmethod
    def raw_acc_to_float(g, reading):
        imu1_acc = [g * 2 * (int(reading[0]) / 32767.0),
                    g * 2 * (int(reading[1]) / 32767.0),
                    g * 2 * (int(reading[2]) / 32767.0)]
        return imu1_acc


if __name__ == '__main__':
    pass
    # list_of_imu = IMUTools.search()
    # imu = []
    # for i in list_of_imu:
    #     imu.append(BaseSens(i[0]))
    # imu[0].start2()
    #
    # def runner():
    #     imu[0].read_imu(False)
    # while 1:
    #     c = threading.Thread(target=runner, args=())
    #     c.start()
    #     c.join()


