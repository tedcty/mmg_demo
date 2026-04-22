import numpy as np
from commons.mymath import Quaternion, RotaasjeFolchoarder


class MyFilter(object):
    def update(self, Gyroscope, Accelerometer, Magnetometer):
        pass


class ComplementaryFilter(MyFilter):

    def __init__(self, buf=10, rate=1/100, alpha=0.98):
        print("ComplementaryFilter")
        print("|-> Requires 10s of static for filter to stabilise")
        print("|-> Only pitch and roll is calculated")
        print("|-> Current Settings:")
        print("|---> Buffer size:\t\t{:d} frames".format(buf))
        print("|---> sampling rate:\t{:.3f}s".format(rate))
        print("|---> alpha:\t\t\t{:.3f}".format(alpha))
        self.gyro_buffer = []
        self.acc_buffer = []
        self.buffer_size = buf
        self.ori_buffer = np.zeros([10, 3])
        self.rate = rate
        self.alpha = alpha

    def get_euler_earth(self):
        return self.ori_buffer[-1, :]*(180/np.pi)

    def update(self, gyro, acc, mag=None):
        self.gyro_buffer.append(gyro)
        self.acc_buffer.append(acc)
        if len(self.gyro_buffer) > self.buffer_size:
            self.gyro_buffer.pop(0)
        if len(self.acc_buffer) > self.buffer_size:
            self.acc_buffer.pop(0)
        self.__calculate_orientation__()
        return self.ori_buffer[-1, :]

    def __calculate_orientation__(self):
        acc = np.mean(np.array(self.acc_buffer), axis=0)
        gyro = np.mean(np.array(self.gyro_buffer), axis=0)
        prev_ori = np.mean(self.ori_buffer, axis=0)
        pitch_a = np.arctan2(acc[1], acc[2])
        pitch_b = prev_ori[0] + self.rate * gyro[0]

        roll_a = np.arctan2(acc[0], acc[2])
        roll_b = prev_ori[1] + self.rate * gyro[1]

        pitch = self.alpha * pitch_b + (1 - self.alpha) * pitch_a
        roll = self.alpha * roll_b + (1 - self.alpha) * roll_a
        tilt = np.array([[pitch, roll, 0]])
        self.ori_buffer = np.append(self.ori_buffer, tilt, axis=0)
        #print(self.ori_buffer[-1, :])

    def get_acc(self):
        return np.mean(self.acc_buffer, axis=0)



class Madgwick(MyFilter):
    # This is an implementation of the Madgwick filter adapted to use the Quaterion class.
    # Other changes:
    # * Combined update and updateIMU to one function
    # * Added to earth frame decomposition of quaternion to Euler angles
    #
    def __init__(self, sample_period=1/100, quaternion=None, beta=0.05):
        self.sample_period = sample_period
        self.quaternion = None
        if quaternion is None:
            self.quaternion = Quaternion()
        else:
            self.quaternion = quaternion
        self.beta = beta

    def update(self, gyro, acc, mag=None):
        q = self.quaternion.to_array()
        if np.linalg.norm(acc) == 0:
            return
        else:
            acc = acc / np.linalg.norm(acc)
        if mag is not None:
            if np.linalg.norm(mag) == 0:
                return
            else:
                mag = mag / np.linalg.norm(mag)
            magq = Quaternion([0, mag[0], mag[1], mag[2]])
            h = self.quaternion*(magq*self.quaternion.conjugate())
            hn = h.to_array()
            b = [0, np.linalg.norm([hn[1], hn[2]]), 0, hn[3]]

            # Gradient decent algorithm corrective step
            F = np.array([2 * (q[1] * q[3] - q[0] * q[2]) - acc[0],
                 2 * (q[0] * q[1] + q[2] * q[3]) - acc[1],
                 2 * (0.5 - q[1] * q[1] - q[2] * q[2]) - acc[2],
                 2 * b[1] * (0.5 - q[2] * q[2] - q[3] * q[3]) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - mag[0],
                 2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - mag[1],
                 2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] * q[1] - q[2] * q[2]) - mag[2]])

            J = np.array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                 [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                 [0, -4 * q[1], -4 * q[2], 0],
                 [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0],
                  -4 * b[1] * q[3] + 2 * b[3] * q[1]],
                 [-2 * b[1] * q[3] + 2 * b[3] * q[1], 2 * b[1] * q[2] + 2 * b[3] * q[0],
                  2 * b[1] * q[1] + 2 * b[3] * q[3], -2 * b[1] * q[0] + 2 * b[3] * q[2]],
                 [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2],
                  2 * b[1] * q[1]]])

        else:
            # Gradient decent algorithm corrective step
            F = np.array([2*(q[1]*q[3] - q[0]*q[2]) - acc[0],
                          2*(q[0]*q[1] + q[2]*q[3]) - acc[1],
                          2*(0.5 - q[1]*q[1] - q[2]*q[2]) - acc[2]])

            J = np.array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                          [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                          [0, -4 * q[1], -4 * q[2], 0]])
        step = np.dot(J.transpose(), F)
        step = step / np.linalg.norm(step)
        prod = self.quaternion * Quaternion([0, gyro[0], gyro[1], gyro[2]])
        qp = 0.5 * prod.to_array() - self.beta*step.transpose()
        qdot = Quaternion(qp)
        q = self.quaternion + qdot.fermannichfaldigje(self.sample_period)
        self.quaternion = q.unit()

    def get_euler_earth(self):
        # decompose as x is forward, y is to the left and up is z
        q = self.quaternion.conjugate()
        return q.toEuler(RotaasjeFolchoarder.zyx) * (180 / np.pi)

    def madgwick_euler_earth(self):
        qc = self.quaternion.conjugate()
        q = qc.to_array()
        R = np.zeros([3, 3])
        R[0, 0] = 2. * q[0]*q[0] - 1 + 2. * q[1]*q[1]
        R[1, 0] = 2. * (q[1] * q[2]-q[0]*q[3])
        R[2, 0] = 2. * (q[1] * q[3]+q[0]*q[2])
        R[2, 1] = 2. * (q[2] * q[3]-q[0]*q[1])
        R[2, 2] = 2. * q[0]*q[0] - 1 + 2. * q[3]*q[3]

        phi = np.arctan2(R[2, 1], R[2, 2]) * (180 / np.pi)
        theta = -np.arctan(R[2, 0] / np.sqrt(1 - R[2, 0]*R[2, 0])) * (180 / np.pi)
        psi = np.arctan2(R[1, 0], R[0, 0]) * (180 / np.pi)

        return [phi, theta, psi]
