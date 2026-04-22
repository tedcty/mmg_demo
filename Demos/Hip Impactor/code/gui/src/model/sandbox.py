import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Constants(Enum):
    mpu6050_2g = 1 / 16384.0
    i4g = "4g"
    i16g = "16g"
    mpu6050_250dps = 1 / 131
    IMU0 = "imu0"
    IMU1 = "imu1"
    IMU2 = "imu2"
    acc = "acc"
    gyr = "gyro"
    ori = "orientation"
    time = "orientation"
    hz = "hz"


class Motion:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.d2x = 0
        self.d2y = 0
        self.d2z = 0

    @property
    def displacement(self):
        return np.array([self.x, self.y, self.z])

    @property
    def velocity(self):
        return np.array([self.dx, self.dy, self.dz])

    @property
    def acceleration(self):
        return np.array([self.d2x, self.d2y, self.d2z])

    @displacement.setter
    def displacement(self, e):
        self.x = e[0]
        self.y = e[1]
        self.z = e[2]

    @velocity.setter
    def velocity(self, e):
        self.dx = e[0]
        self.dy = e[1]
        self.dz = e[2]

    @acceleration.setter
    def acceleration(self, e):
        self.d2x = e[0]
        self.d2y = e[1]
        self.d2z = e[2]


class AngularData(Motion):
    def __init__(self):
        super().__init__()


class LinearData(Motion):
    def __init__(self):
        super().__init__()


class Messenger:
    def __init__(self):
        self.data = {}

    # gyro_range_divisor default for 245dps = 2
    @staticmethod
    def LSM6DS3_gyro(x, gyro_range_divisor=2):
        output = x * 4.375 * gyro_range_divisor / 1000
        return output

    # acc_range default 2g
    @staticmethod
    def LSM6DS3_acc(x, acc_range: int = 2):
        output = x * 0.061 * (acc_range >> 1) / 1000
        return output

    @staticmethod
    def create(s=None):
        # t Hz YPR Acc Acc Gyro Acc Gyro
        acc_convert = Constants.mpu6050_2g.value
        gyro_convert = Constants.mpu6050_250dps.value
        test = np.array([66.07, 30.30, 2.78, 0.04, 0.11, 0, -1, -24, 8, 64, 16396, 31, 5, -26, 1177, -893, -13430, -382,
                         278, 250])

        m = Messenger()
        m.data[Constants.time] = test[0]
        m.data[Constants.hz] = test[1]
        imu0_dmp = np.array(test[2:5])
        imu0_acc = (np.array(test[5:8])*acc_convert)*Gravity.earth
        m.data[Constants.IMU0] = {Constants.ori: imu0_dmp, Constants.acc: imu0_acc}
        a0 = test[8:11]
        imu1_acc = (a0 * acc_convert) * Gravity.earth
        g0 = test[11:14]
        imu1_gyr = (g0 * gyro_convert)

        m.data[Constants.IMU1] = {Constants.acc: imu1_acc, Constants.gyr: imu1_gyr}
        a1 = test[14:17]
        imu_a = np.array([Messenger.LSM6DS3_acc(a1[i]) for i in range(0, 3)])
        g1 = test[17:]
        imu_g = np.array([Messenger.LSM6DS3_gyro(g1[i]) for i in range(0, 3)])
        imu2_acc = imu_a * Gravity.earth
        imu2_gyr = imu_g
        m.data[Constants.IMU2] = {Constants.acc: imu2_acc, Constants.gyr: imu2_gyr}
        return m


class Chip:
    def __init__(self, chip_id):
        self.linear = LinearData()
        self.angular = AngularData()
        self.chip_id = chip_id

    def update(self, d):
        self.update_tilt()
        self.update_fusion()
        self.estimate_from_chip_data()
        pass

    def update_tilt(self):
        pass

    def update_fusion(self):
        pass

    def estimate_from_chip_data(self):
        self.estimate_linear_velocity()
        self.estimate_linear_displacement()
        self.estimate_angular_displacement()
        pass

    def estimate_linear_velocity(self, integration=True):
        pass

    def estimate_linear_displacement(self, integration=True):
        pass

    def estimate_angular_displacement(self, integration=True):
        pass


class ClusterModel:
    def __init__(self):
        # IMU 0, 1 = MPU6050
        # IMU 2 = LSM6DS1
        self.points = np.array([[0, 0, 0], [18, 0, 0], [-1.3, 0, -19.4]]).T
        self.distances = np.array([18, 20, 26])  # 0-1, 0-2, 1-2
        self.imu0 = Chip(Constants.IMU0)
        self.imu1 = Chip(Constants.IMU1)
        self.imu2 = Chip(Constants.IMU2)
        self.imus = [self.imu0, self.imu1, self.imu2]

    def update(self, d: Messenger):
        for imu in self.imus:
            imu.update(d.data[imu.chip_id])


class Stick:
    def __init__(self, r=np.array([0, 0, 0])):
        self.rotation_center = r
        self.vector = np.array([0, 0, -1])


class Stickman:
    def __init__(self):
        self.top = Stick()
        self.bottom = Stick(r=self.top.vector)


class Gravity:
    earth = 9.80665     # m/s^2

    def __init__(self, v=np.zeros([1, 3])):
        self.vector = v  # assuming [x, y, z]
        self.mag = np.linalg.norm(v)

    def g_vector_to_ground(self):
        v1 = self.vector
        m1 = np.linalg.norm(v1)
        v2 = np.array([self.vector[0], self.vector[1], 0])
        m2 = np.linalg.norm(v2)
        v1v2 = np.vdot(v1, v2)
        angle = np.arccos(v1v2/(m1 * m2))
        return angle*(180/np.pi)


if __name__ == '__main__':
    print(Gravity.earth)
    vg1 = Gravity(v=[1, 1, 1])
    vg2 = Gravity(v=[1, 1, 2])
    # a = np.sqrt(2)
    # o = 1
    # print(np.arctan2(o, a)*(180/np.pi))
    print(vg1.g_vector_to_ground())
    print(vg2.g_vector_to_ground())
    o1 = vg1.g_vector_to_ground()
    o2 = vg2.g_vector_to_ground()
    ok = (180 - o2 + o1)
    ok1 = (180 - o1 + o2)
    if ok > 180:
        ok = ok - 180
    if ok < -180:
        ok = ok + 180
        print()
    print(ok)
    print(ok1)
    cm = ClusterModel()
    print(cm.points)
    m = Messenger.create()
    print()
