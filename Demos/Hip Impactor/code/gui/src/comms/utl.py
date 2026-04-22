from scipy.stats import mode
import pandas as pd
import numpy as np
from yatpkg.math.transformation import Quaternion
from enum import Enum


def simple_reader(filename, delimiter=",", headers=False):
    ret = []
    lengths = []
    stream = "hoi"
    read_successful = False
    try:
        f = open(filename, "r")
        while len(stream) > 0:
            stream = f.readline()
            elements = stream.strip().split(delimiter)
            ret.append(elements)
            lengths.append(len(elements))
        read_successful = True
    except OSError:
        print('cannot open', filename)
    finally:
        f.close()
    if read_successful:
        col = mode(lengths)[0][0]
        bounty = [b for b in range(0, len(lengths)) if lengths[b] != col]
        ret2 = [ret[i] for i in range(0, len(ret)) if i not in bounty]
        if headers:
            header = ret2.pop(0)
            try:
                data = [[float(i[j]) if j > 0 else int(i[j]) for j in range(0, col)] for i in ret2]
                return pd.DataFrame(data=data, columns=header)
            except ValueError:
                print("Why")
        else:
            try:
                data = [[float(i[j]) if j > 0 else int(i[j]) for j in range(0, col)] for i in ret2]
                return pd.DataFrame(data=data)
            except ValueError:
                print("Why")
        return ret
    return None


class Constants(Enum):
    impactor = 40


class IRScannerInfo:
    def __init__(self, data_file_name: str = None):
        self.matrix = None
        self.q: Quaternion = None
        self.euler = None
        self.euler_deg = None
        if data_file_name is not None:
            self.matrix = pd.read_csv(data_file_name, header=None)
            matrix_np = self.matrix.to_numpy()
            self.q: Quaternion = Quaternion.create_from_matrix(matrix_np)
            self.euler = self.q.to_euler()
            self.euler_deg = self.euler*(180/np.pi)


class Chippy:
    num_blocks = 22

    @staticmethod
    def board_id(s):
        b = s.split(" ")
        return b[2].split(":")[0]

    @staticmethod
    def board_time(s):
        b = s.split(":")
        return float(b[1])

    @staticmethod
    def board_hz(s):
        b = s.split(" ")
        return float(b[0])

    @staticmethod
    def board_imu1_ypr(s):
        ret = np.array([0., 0., 0.])
        for m in range(0, len(s)):
            r = s[m]
            if "ypr:" in r:
                start = r.index("ypr:")
                rs = r[start+4:]
            else:
                rs = r
            n = rs.strip().split(" ")
            # print(n)
            f = float(n[1])
            ret[m] = f
        return ret

    @staticmethod
    def board_imu1_acc(s):
        ret = np.array([0., 0., 0.])
        for m in range(0, len(s)):
            r = s[m]
            if "areal:" in r:
                start = r.index("areal:")
                rs = r[start + 6:]
            else:
                rs = r
            n = rs.strip()
            ret[m] = float(n) / 32767.0
        return ret

    @staticmethod
    def board_imu2_acc(s):
        ret = np.array([0., 0., 0.])
        for m in range(0, len(s)):
            r = s[m]
            if "MPU2:" in r:
                start = r.index("MPU2:")
                rs = r[start + 5:]
            else:
                rs = r
            n = rs.strip()
            ret[m] = float(n) / 32767.0
        return ret

    @staticmethod
    def board_imu_gyr(s):
        ret = np.array([0., 0., 0.])
        for m in range(0, len(s)):
            r = s[m]
            n = r.strip()
            ret[m] = float(n) / 131.0
        return ret

    @staticmethod
    def board_imu3_acc(s):
        ret = np.array([0., 0., 0.])
        for m in range(0, len(s)):
            r = s[m]
            if "MPU3:" in r:
                start = r.index("MPU3:")
                rs = r[start + 5:]
            else:
                rs = r
            n = rs.strip()
            ret[m] = float(n) / 32767.0
        return ret


class Cluster:
    @staticmethod
    def extract(e):
        imus = {}
        if len(e) == Chippy.num_blocks:
            try:
                c = Chippy.board_id(e[0])
            except IndexError:
                return None
            imus[c] = []
        if len(e) == Chippy.num_blocks:
            messagex = {
                "c": Chippy.board_id(e[0]),
                "t": Chippy.board_time(e[1]),
                "hz": Chippy.board_hz(e[2]),
                "imu1_ypr": Chippy.board_imu1_ypr(e[3:6]),
                "imu1_acc": Chippy.board_imu1_acc(e[6:9]),
                "imu2_acc": Chippy.board_imu2_acc(e[9:12]),
                "imu2_gyr": Chippy.board_imu_gyr(e[12:15]),
                "imu3_acc": Chippy.board_imu3_acc(e[15:18]),
                "imu3_gyr": Chippy.board_imu_gyr(e[18:21])
            }
            messagex["imu2_gyr_norm"] = np.linalg.norm(messagex["imu2_gyr"])
            return messagex
            # imus[message["c"]].append(message)
        return imus

# if __name__ == '__main__':
#     d = simple_reader("C:/Users/tyeu008/Downloads/IMU_Test2.txt")
#     imus = {}
#     for e in d:
#         if len(e) == Chippy.num_blocks:
#             c = Chippy.board_id(e[0])
#             imus[c] = []
#     for e in d:
#         if len(e) == Chippy.num_blocks:
#             message = {
#                 "c": Chippy.board_id(e[0]),
#                 "t": Chippy.board_time(e[1]),
#                 "hz": Chippy.board_hz(e[2]),
#                 "imu1_ypr": Chippy.board_imu1_ypr(e[3:6]),
#                 "imu1_acc": Chippy.board_imu1_acc(e[6:9]),
#                 "imu2_acc": Chippy.board_imu2_acc(e[9:12]),
#                 "imu2_gyr": Chippy.board_imu_gyr(e[12:15]),
#                 "imu3_acc": Chippy.board_imu3_acc(e[15:18]),
#                 "imu3_gyr": Chippy.board_imu_gyr(e[18:21])
#             }
#             imus[message["c"]].append(message)
#             pass
#     pass
