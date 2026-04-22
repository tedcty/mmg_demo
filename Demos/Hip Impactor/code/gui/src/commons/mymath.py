from enum import Enum
import numpy as np


class Diminsje(Enum):
    D1D = "1D"
    D2D = "2D"
    D3D = "3D"
    D4D = "4D"


class RotaasjeFolchoarder(Enum):
    zyx = (1,[0,1,2], "zyx")
    zyz = (2,[2,1,2], "zyz")
    zxy = (3,[1,0,2], "zxy")
    zxz = (4,[2,0,2], "zxz")
    yxz = (5,[2,0,1], "yxz")
    yxy = (6,[1,0,1], "yxy")
    yzx = (7,[0,2,1], "yzx")
    yzy = (8,[1,2,1], "yzy")
    xyz = (9,[2,1,0], "xyz")
    xyx = (10,[0,1,0], "xyx")
    xzy = (11,[1,2,0], "xzy")
    xzx = (12,[0,2,0], "xzx")

    def __init__(self, ident, list_folchoarder, etiket):
        self.list_folchoarder = list_folchoarder
        self.etiket = etiket
        self.ident = ident

    def gelikense(self, folchoarder):
        if self.ident == folchoarder.ident:
            return True
        return False


class AngularFaasje(object):
    @staticmethod
    def tot_matriks(x, y, z):
        w = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        return w


class Rotaasje(object):
    """description of class"""
    def __init__(self, hoeke=[0, 0, 0], folchoarder =RotaasjeFolchoarder.zyx, krektens=8):
        self.r_folchoarder = folchoarder
        self.quaternion = Quaternion()
        self.euler = np.array([0, 0, 0])
        self.r_matriks = np.eye(3, 3)
        self.krektens = krektens
        if isinstance(hoeke, Quaternion):
            self.sette_rotaasje_quaternion(hoeke, folchoarder)
        else:
            hoeke = np.squeeze(np.asarray(hoeke))
            self.sette_rotaasje_euler(hoeke[0], hoeke[1], hoeke[2], folchoarder)

    @staticmethod
    def rotaasje_svd(X, Y):
        x_mean = np.reshape(np.mean(X, axis=1), [3, 1])
        y_mean = np.reshape(np.mean(Y, axis=1), [3, 1])

        XC = X - x_mean
        YC = Y - y_mean

        C = np.dot(YC, XC.transpose())
        u, s, vh = np.linalg.svd(C)
        R = np.matmul(u,vh)
        cor = np.eye(3, 3)
        cor[2, 2] = np.linalg.det(R)
        if np.linalg.det(R) < 0:
            R = np.dot(u, np.dot(cor, vh.transpose()))

        t = y_mean - np.dot(R, x_mean)
        return [R, t]

    @staticmethod
    def rotaasje_affine(u, ut):
        r = u.shape[1]
        a = np.zeros([4, 4])
        b = np.zeros([4, 3])

        for d in range(0, r):
            b[0, 0] = b[0, 0] + u[0, d] * ut[0, d]
            b[1, 0] = b[1, 0] + u[1, d] * ut[0, d]
            b[2, 0] = b[2, 0] + u[2, d] * ut[0, d]
            b[3, 0] = b[3, 0] + 1 * ut[0, d]

            b[0, 1] = b[0, 1] + u[0, d] * ut[1, d]
            b[1, 1] = b[1, 1] + u[1, d] * ut[1, d]
            b[2, 1] = b[2, 1] + u[2, d] * ut[1, d]
            b[3, 1] = b[3, 1] + 1 * ut[1, d]

            b[0, 2] = b[0, 2] + u[0, d] * ut[2, d]
            b[1, 2] = b[1, 2] + u[1, d] * ut[2, d]
            b[2, 2] = b[2, 2] + u[2, d] * ut[2, d]
            b[3, 2] = b[3, 2] + 1 * ut[2, d]

        for d in range(0, r):
            a[0, 0] = a[0, 0] + u[0, d] * u[0, d]
            a[0, 1] = a[0, 1] + u[0, d] * u[1, d]
            a[0, 2] = a[0, 2] + u[0, d] * u[2, d]
            a[0, 3] = a[0, 3] + u[0, d] * 1

            a[1, 1] = a[1, 1] + u[1, d] * u[1, d]
            a[1, 2] = a[1, 2] + u[1, d] * u[2, d]
            a[1, 3] = a[1, 3] + u[1, d] * 1

            a[2, 2] = a[2, 2] + u[2, d] * u[2, d]
            a[2, 3] = a[2, 3] + u[2, d] * 1

            a[3, 3] = a[3, 3] + 1 * 1

        a[1, 0] = a[0, 1]
        a[2, 0] = a[0, 2]
        a[2, 1] = a[1, 2]
        a[3, 0] = a[0, 3]
        a[3, 1] = a[1, 3]
        a[3, 2] = a[2, 3]

        t = np.transpose(np.dot(np.linalg.inv(a), b))
        dummy = np.reshape(np.array([0, 0, 0, 1]), [1, 4])
        return np.append(t, dummy, axis=0)

    @staticmethod
    def nij(x, y, z, folchoarder):
        r = Rotaasje()
        r.sette_rotaasje_euler(x, y, z, folchoarder)
        return r

    @staticmethod
    def rotaasje_oer_in_fektor(u, hoeke):
        m00 = np.cos(hoeke)+np.power(u[0], 2)*(1-np.cos(hoeke))
        m01 = u[0]*u[1]*(1-np.cos(hoeke))-u[2]*np.sin(hoeke)
        m02 = u[0]*u[2]*(1-np.cos(hoeke))+u[1]*np.sin(hoeke)
        m10 = u[1]*u[0]*(1-np.cos(hoeke))+u[2]*np.sin(hoeke)
        m11 = np.cos(hoeke)+np.power(u[1], 2)*(1-np.cos(hoeke))
        m12 = u[1]*u[2]*(1-np.cos(hoeke))-u[0]*np.sin(hoeke)
        m20 = u[2]*u[0]*(1-np.cos(hoeke))-u[1]*np.sin(hoeke)
        m21 = u[2]*u[1]*(1-np.cos(hoeke))+u[0]*np.sin(hoeke)
        m22 = np.cos(hoeke) + np.power(u[2], 2)*(1-np.cos(hoeke))
        M = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
        return M

    @staticmethod
    def toXYZ(matr):
        roi = Rotaasje()
        roi.sette_rotaasje_matriks(matr, RotaasjeFolchoarder.zxy)
        eul = roi.quaternion.to_euler(RotaasjeFolchoarder.xyz)

        ret = Rotaasje(hoeke=eul)
        return ret.r_matriks

    @staticmethod
    def draaie_e1(x, krektens=12):
        c, s = np.round(np.cos(x), krektens), np.round(np.sin(x), krektens)
        if c == 0:
            c = c*c

        if s == 0:
            s = s*s
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def draaie_e2(y, krektens=12):
        c, s = np.round(np.cos(y), krektens), np.round(np.sin(y), krektens)
        if c == 0:
            c = c*c

        if s == 0:
            s = s*s
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def draaie_e3(z, krektens=12):
        c, s = np.round(np.cos(z), krektens), np.round(np.sin(z), krektens)
        if c == 0:
            c = c*c

        if s == 0:
            s = s*s
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def sette_rotaasje(self, folchoarder, *arguminten):
        if len(arguminten) == 3:
            self.sette_rotaasje_euler(arguminten[0], arguminten[1], arguminten[2], folchoarder)
        elif len(arguminten) == 1:
            self.sette_rotaasje_quaternion(arguminten[0], folchoarder)
        return self

    def __mul__(self, oar):
        return self.fermannichfaldigje(oar)

    def __sub__(self, other):
        return [self.euler[0]-other.euler[0], self.euler[1]-other.euler[1], self.euler[2]-other.euler[2]]

    def inverse(self):
        r = Rotaasje()
        r.sette_rotaasje(self.r_folchoarder, self.quaternion.inverse())
        return r

    def fermannichfaldigje(self, oar):
        qm = self.quaternion.fermannichfaldigje_quaternion(oar.quaternion)
        r = Rotaasje()
        r.sette_rotaasje(self.r_folchoarder, qm)
        return r
        
    def sette_rotaasje_euler(self, x, y, z, folchoarder):
        self.euler = [x, y, z]
        self.r_folchoarder = folchoarder
        self.r_matriks = np.eye(3)
        for axis in folchoarder.list_folchoarder:
            if axis == 0:
                self.r_matriks = np.matmul(self.draaie_e1(self.euler[axis]), self.r_matriks)
            if axis == 1:
                self.r_matriks = np.matmul(self.draaie_e2(self.euler[axis]), self.r_matriks)
            if axis == 2:
                self.r_matriks = np.matmul(self.draaie_e3(self.euler[axis]), self.r_matriks)
        
        self.quaternion = Quaternion.meitsje_quaternion_ut_matriks(self.r_matriks)

    def sette_rotaasje_quaternion(self, q, order):
        if not isinstance(q, Quaternion):
            r = Quaternion()
            r.w = q[0]
            r.i = q[1]
            r.j = q[2]
            r.k = q[3]
            q = r

        self.quaternion = q
        self.r_folchoarder = order
        self.euler = q.to_euler(order)
        self.r_matriks = np.eye(3)
        for axis in order.list_folchoarder:
            if axis == 0:
                self.r_matriks = np.matmul(self.draaie_e1(self.euler[axis]), self.r_matriks)
            if axis == 1:
                self.r_matriks = np.matmul(self.draaie_e2(self.euler[axis]), self.r_matriks)
            if axis == 2:
                self.r_matriks = np.matmul(self.draaie_e3(self.euler[axis]), self.r_matriks)

    def sette_rotaasje_matriks(self, r, order):
        self.r_matriks = r
        self.r_folchoarder = order
        self.quaternion = Quaternion.meitsje_quaternion_ut_matriks(self.r_matriks)
        self.euler = self.quaternion.to_euler(order)

    def decompose(self, folchoarder=RotaasjeFolchoarder.zyx):
        return self.quaternion.to_euler(folchoarder)

    @staticmethod
    def rotaasje_matriks(euler, folchoarder):
        r_matriks = np.eye(3)
        for axis in folchoarder.list_folchoarder:
            if axis == 0:
                r_matriks = np.matmul(Rotaasje.draaie_e1(euler[axis]), r_matriks)
            if axis == 1:
                r_matriks = np.matmul(Rotaasje.draaie_e2(euler[axis]), r_matriks)
            if axis == 2:
                r_matriks = np.matmul(Rotaasje.draaie_e3(euler[axis]), r_matriks)
        return r_matriks


class Quaternion(object):
    """description of class"""

    def __init__(self, vars=[]):
        if len(vars) == 0:
            self.w = 1
            self.i = 0
            self.j = 0
            self.k = 0
        elif len(vars) == 4:
            self.w = vars[0]
            self.i = vars[1]
            self.j = vars[2]
            self.k = vars[3]

    def __repr__(self):
        return "Quaternion"

    def __str__(self):
        return "Quaternion: %2.2f + %2.2fi + %2.2fj + %2.2fk" % (self.w, self.i, self.j, self.k)

    def __sub__(self, other):
        q = Quaternion()
        q.w = (self.w - other.w)
        q.i = (self.i - other.i)
        q.j = (self.j - other.j)
        q.k = (self.k - other.k)
        return q

    def __add__(self, other):
        q = Quaternion()
        q.w = (self.w + other.w)
        q.i = (self.i + other.i)
        q.j = (self.j + other.j)
        q.k = (self.k + other.k)
        return q

    def __cmp__(self, x):
        w = (self.w - x.w) * (self.w - x.w)
        i = (self.i - x.i) * (self.i - x.i)
        j = (self.j - x.j) * (self.j - x.j)
        k = (self.k - x.k) * (self.k - x.k)

        if (w + i + j + k) < 1.0e-16:
            return True
        else:
            return False

    def __mul__(self, r):
        return self.fermannichfaldigje_quaternion(r)

    def __truediv__(self, other):
        if not isinstance(other, Quaternion):
            a = self.w / other
            b = self.i / other
            c = self.j / other
            d = self.k / other
            q = Quaternion([a, b, c, d])
            return q

    def unit(self):
        return self/self.norm()

    def norm(self):
        a = self.w * self.w
        b = self.i * self.i
        c = self.j * self.j
        d = self.k * self.k
        return np.sqrt(a + b + c + d)

    @staticmethod
    def nul():
        q = Quaternion()
        q.w = 1e-10
        q.i = 1e-10
        q.j = 1e-10
        q.k = 1e-10
        return q

    def fermannichfaldigje_quaternion(self, r):
        n = Quaternion()
        n.w = r.w * self.w - r.i * self.i - r.j * self.j - r.k * self.k
        n.i = r.w * self.i + r.i * self.w - r.j * self.k + r.k * self.j
        n.j = r.w * self.j + r.i * self.k + r.j * self.w - r.k * self.i
        n.k = r.w * self.k - r.i * self.j + r.j * self.i + r.k * self.w
        return n

    def fermannichfaldigje(self, s):
        n = Quaternion()
        n.w = s * self.w
        n.i = s * self.i
        n.j = s * self.j
        n.k = s * self.k
        return n

    def conjugate(self):
        n = Quaternion()
        n.w = self.w
        n.i = -self.i
        n.j = -self.j
        n.k = -self.k
        return n

    def inverse(self):
        n = Quaternion()
        n.w = self.w
        n.i = -self.i
        n.j = -self.j
        n.k = -self.k

        w2 = self.w * self.w
        x2 = self.i * self.i
        y2 = self.j * self.j
        z2 = self.k * self.k
        q_sq_sum = w2 + x2 + y2 + z2

        n.w = n.w / q_sq_sum
        n.i = n.i / q_sq_sum
        n.j = n.j / q_sq_sum
        n.k = n.k / q_sq_sum
        return n

    @staticmethod
    def meitsje_quaternion_ut_matriks(a):
        qout = Quaternion()
        trace = a[0, 0] + a[1, 1] + a[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qout.w = 0.25 / s
            qout.i = (a[2, 1] - a[1, 2]) * s
            qout.j = (a[0, 2] - a[2, 0]) * s
            qout.k = (a[1, 0] - a[0, 1]) * s
        else:
            boo1 = np.greater(a[0, 0], a[1, 1])
            boo2 = np.greater(a[0, 0], a[2, 2])
            boo3 = np.greater(a[1, 1], a[2, 2])
            if boo1 & boo2:
                s = 2.0 * np.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2])
                qout.w = (a[2, 1] - a[1, 2]) / s
                qout.i = 0.25 * s
                qout.j = (a[0, 1] + a[1, 0]) / s
                qout.k = (a[0, 2] + a[2, 0]) / s
            elif boo3:
                s = 2.0 * np.sqrt(1.0 + a[1, 1] - a[0, 0] - a[2, 2])
                qout.w = (a[0, 2] - a[2, 0]) / s
                qout.i = (a[0, 1] + a[1, 0]) / s
                qout.j = 0.25 * s
                qout.k = (a[1, 2] + a[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + a[2, 2] - a[0, 0] - a[1, 1])
                qout.w = (a[1, 0] - a[0, 1]) / s
                qout.i = (a[0, 2] + a[2, 0]) / s
                qout.j = (a[1, 2] + a[2, 1]) / s
                qout.k = 0.25 * s
        return qout

    @staticmethod
    def decompose(m, order=RotaasjeFolchoarder.zyx):
        return Quaternion.meitsje_quaternion_ut_matriks(m).to_euler(order)

    @staticmethod
    def slerp(qa, qb, t):
        qm = Quaternion()
        # Calculate angle between them
        cosHalfTheta = qa.w * qb.w + qa.i * qb.i + qa.j * qb.j + qa.k * qb.k
        # if qa=qb or qa=-qb then theta = 0 and we can return qa
        if (np.abs(cosHalfTheta) >= 1.0):
            qm.w = qa.w
            qm.i = qa.i
            qm.j = qa.j
            qm.k = qa.k
            return qm
            # Calculate temporary values.
        halfTheta = np.arccos(cosHalfTheta)
        sinHalfTheta = np.sqrt(1.0 - cosHalfTheta * cosHalfTheta)
        # if theta = 180 degrees then result is not fully defined
        # we could rotate around any axis normal to qa or qb
        if (np.abs(sinHalfTheta) < 0.001):  # // fabs is floating point absolute
            qm.w = (qa.w * 0.5 + qb.w * 0.5)
            qm.i = (qa.i * 0.5 + qb.i * 0.5)
            qm.j = (qa.j * 0.5 + qb.j * 0.5)
            qm.k = (qa.k * 0.5 + qb.k * 0.5)
            return qm

        ratioA = np.sin((1 - t) * halfTheta) / sinHalfTheta
        ratioB = np.sin(t * halfTheta) / sinHalfTheta
        # //calculate Quaternion.
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.i = (qa.i * ratioA + qb.i * ratioB)
        qm.j = (qa.j * ratioA + qb.j * ratioB)
        qm.k = (qa.k * ratioA + qb.k * ratioB)
        return qm

    def to_array(self):
        return np.array([self.w, self.i, self.j, self.k])

    def to_martix(self, folchoarder):
        euler = self.to_euler(folchoarder)
        r = Rotaasje(hoeke=euler, RotaasjeFolchoarder=folchoarder)
        return r.r_matriks

    # Warning results of the conversion only guarantee for angles -89.99999955000001 > x < 89.99999955000001
    def to_euler(self, folchoarder=RotaasjeFolchoarder.xyz):
        def __get_tait_bryan_angles(r11, r12, r21, r31, r32):
            return [np.arctan2(r11, r12), np.arcsin(np.round(r21, decimals=10)), np.arctan2(r31, r32)]

        def __get_euler_angles(r11, r12, r21, r31, r32):
            return [np.arctan2(r11, r12), np.arccos(np.round(r21, decimals=10)), np.arctan2(r31, r32)]

        if folchoarder.gelikense(RotaasjeFolchoarder.zxy):
            angles = __get_tait_bryan_angles(-2 * (self.i * self.j - self.w * self.k),
                                                  self.w * self.w - self.i * self.i + self.j * self.j - self.k * self.k,
                                                  2 * (self.j * self.k + self.w * self.i),
                                                  -2 * (self.i * self.k - self.w * self.j),
                                                  self.w * self.w - self.i * self.i - self.j * self.j + self.k * self.k)
            angles = [angles[2], angles[0], angles[1]]

        elif folchoarder.gelikense(RotaasjeFolchoarder.xyz):
            angles = __get_tait_bryan_angles(-2 * (self.j * self.k - self.w * self.i),
                                                  self.w * self.w - self.i * self.i - self.j * self.j + self.k * self.k,
                                                  2 * (self.i * self.k + self.w * self.j),
                                                  -2 * (self.i * self.j - self.w * self.k),
                                                  self.w * self.w + self.i * self.i - self.j * self.j - self.k * self.k)
            angles = [angles[0], angles[1], angles[2]]
        elif folchoarder.gelikense(RotaasjeFolchoarder.zyz):
            angles = __get_euler_angles(2 * (self.j * self.k - self.w * self.i),
                                             2 * (self.i * self.k + self.w * self.j),
                                             self.w * self.w - self.i * self.i - self.j * self.j + self.k * self.k,
                                             2 * (self.j * self.k + self.w * self.i),
                                             -2 * (self.i * self.k - self.w * self.j))
        elif folchoarder.gelikense(RotaasjeFolchoarder.zyx):
            angles = __get_tait_bryan_angles(2 * (self.i * self.j + self.w * self.k),
                                                  self.w * self.w + self.i * self.i - self.j * self.j - self.k * self.k,
                                                  -2 * (self.i * self.k - self.w * self.j),
                                                  2 * (self.j * self.k + self.w * self.i),
                                                  self.w * self.w - self.i * self.i - self.j * self.j + self.k * self.k)
            angles = [angles[2], angles[1], angles[0]]
        elif folchoarder.gelikense(RotaasjeFolchoarder.zxz):
            angles = __get_euler_angles(2 * (self.i * self.k + self.w * self.j),
                                             -2 * (self.j * self.k - self.w * self.i),
                                             self.w * self.w - self.i * self.i - self.j * self.j + self.k * self.k,
                                             2 * (self.i * self.k - self.w * self.j),
                                             2 * (self.j * self.k + self.w * self.i))
        elif folchoarder.gelikense(RotaasjeFolchoarder.yxz):
            angles = __get_tait_bryan_angles(2 * (self.i * self.k + self.w * self.j),
                                                  self.w * self.w - self.i * self.i - self.j * self.j + self.k * self.k,
                                                  -2 * (self.j * self.k - self.w * self.i),
                                                  2 * (self.i * self.j + self.w * self.k),
                                                  self.w * self.w - self.i * self.i + self.j * self.j - self.k * self.k)
            angles = [angles[1], angles[0], angles[2]]
        elif folchoarder.gelikense(RotaasjeFolchoarder.yxy):
            angles = __get_euler_angles(2 * (self.i * self.j - self.w * self.k),
                                             2 * (self.j * self.k + self.w * self.i),
                                             self.w * self.w - self.i * self.i + self.j * self.j - self.k * self.k,
                                             2 * (self.i * self.j + self.w * self.k),
                                             -2 * (self.j * self.k - self.w * self.i))
        elif folchoarder.gelikense(RotaasjeFolchoarder.yzx):
            angles = __get_tait_bryan_angles(-2 * (self.i * self.k - self.w * self.j),
                                                  self.w * self.w + self.i * self.i - self.j * self.j - self.k * self.k,
                                                  2 * (self.i * self.j + self.w * self.k),
                                                  -2 * (self.j * self.k - self.w * self.i),
                                                  self.w * self.w - self.i * self.i + self.j * self.j - self.k * self.k)
            angles = [angles[1], angles[2], angles[0]]
        elif folchoarder.gelikense(RotaasjeFolchoarder.yzy):
            angles = __get_euler_angles(2 * (self.j * self.k + self.w * self.i),
                                             -2 * (self.i * self.j - self.w * self.k),
                                             self.w * self.w - self.i * self.i + self.j * self.j - self.k * self.k,
                                             2 * (self.j * self.k - self.w * self.i),
                                             2 * (self.i * self.j + self.w * self.k))

        elif folchoarder.gelikense(RotaasjeFolchoarder.xyx):
            angles = __get_euler_angles(2 * (self.i * self.j + self.w * self.k),
                                             -2 * (self.i * self.k - self.w * self.j),
                                             self.w * self.w + self.i * self.i - self.j * self.j - self.k * self.k,
                                             2 * (self.i * self.j - self.w * self.k),
                                             2 * (self.i * self.k + self.w * self.j))
        elif folchoarder.gelikense(RotaasjeFolchoarder.xzy):
            angles = __get_tait_bryan_angles(2 * (self.j * self.k + self.w * self.i),
                                                  self.w * self.w - self.i * self.i + self.j * self.j - self.k * self.k,
                                                  -2 * (self.i * self.j - self.w * self.k),
                                                  2 * (self.i * self.k + self.w * self.j),
                                                  self.w * self.w + self.i * self.i - self.j * self.j - self.k * self.k)
            angles = [angles[0], angles[2], angles[1]]
        elif folchoarder.gelikense(RotaasjeFolchoarder.xzx):
            angles = __get_euler_angles(2 * (self.i * self.k - self.w * self.j),
                                             2 * (self.i * self.j + self.w * self.k),
                                             self.w * self.w + self.i * self.i - self.j * self.j - self.k * self.k,
                                             2 * (self.i * self.k + self.w * self.j),
                                             -2 * (self.i * self.j - self.w * self.k))
        return np.array(angles)


class Myn_Vector(object):

    def __init__(self, p):
        self.x = float(p[0])
        self.y = float(p[1])
        self.z = float(p[2])

    @classmethod
    def nul(cls):
        return cls([0, 0, 0])

    @classmethod
    def nij(cls, p):
        if isinstance(p, np.matrix):
            p = np.squeeze(np.asarray(p))
        return cls(p)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Myn_Vector.nij([x, y, z])

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Myn_Vector.nij([x,y,z])

    def __truediv__(self, other):
        if not isinstance(other, Myn_Vector):
            x = self.x / other
            y = self.y / other
            z = self.z / other
        return Myn_Vector.nij([float(x), float(y), float(z)])

    def __str__(self):
        return "[%2.3f, %2.3f, %2.3f]" % (self.x, self.y, self.z)

    def __str_no_brackets__(self):
        return "%2.4f,%2.4f,%2.4f" % (self.x, self.y, self.z)

    def to_array(self):
        return [self.x, self.y, self.z]

    def toQuaternion(self):
        q = Quaternion()
        q.w = 0
        q.i = self.x
        q.j = self.y
        q.k = self.z
        return q

    def v(self):
        return [self.x, self.y, self.z]

    def npv(self):
        return np.array(self.v())

    def npm(self):
        return np.matrix(([self.x], [self.y], [self.z]))

    def magnitude(self):
        return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    # Normalise the point
    def norm(self):
        p = Myn_Vector([0, 0, 0])
        p.x = self.x / self.magnitude()
        p.y = self.y / self.magnitude()
        p.z = self.z / self.magnitude()
        return p

    # Normalise the point and replace current value
    def norm_d(self):
        self.x = self.x / self.magnitude()
        self.y = self.y / self.magnitude()
        self.z = self.z / self.magnitude()

    def add(self, p):
        return Myn_Vector.nij([self.x + p.x, self.y + p.y, self.z + p.z])


