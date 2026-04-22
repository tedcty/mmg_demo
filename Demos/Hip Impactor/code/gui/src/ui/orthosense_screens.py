from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from commons.lang import CommonSymbols
from comms.utl import IRScannerInfo, Constants
# noinspection PyUnresolvedReferences
from kivy.graphics import Color, Ellipse, Line, Rectangle  # there is a bug in pycharm, the added line above
from kivy.uix.checkbox import CheckBox
import threading
import time
from kivy.uix.spinner import Spinner
from datetime import datetime
from ui.orthosense_widget import BoxsensInfo
from kivy.uix.widget import WidgetException

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import numpy as np


class TargetScreen(Screen):
    def __init__(self, name="Target", myparent=None, ws=[1280, 720], imucup=None, settingscreen=None):
        self.pelvis_info = IRScannerInfo()
        super(TargetScreen, self).__init__(name=str(name))
        center = [ws[0] * 0.75 + (0.001 * ws[0]), 0.45 * ws[1] + (0.010 * ws[1]) / 2]
        self.currentwindow = ws
        self.setting_screen = settingscreen
        self.radius = ws[0] * 0.75 - ws[0] * 0.55
        self.boxme = [[center[0], center[0] - self.radius, center[0] + self.radius],
                      [center[1], center[1] - self.radius, center[1] + self.radius]]
        self.cup_data = [0, 0]
        self.cup_target_data = [45, 25]
        self.cup_target = [self.cup_target_data[0] / 90 * self.radius, self.cup_target_data[1] / 90 * self.radius]
        self.imupu = None
        self.isclosing = False
        self.draw_targets(ws)
        self.imucupo = imucup
        self.myparent = myparent
        self.labels_holder = Widget()
        self.left_offset = 0.0234375 * ws[0]
        self.left_offset_radio = 0.1171875 * ws[0]
        self.top_offset = 0.118055556 * ws[1]
        self.top_offset_title = 0.040611111 * ws[1]
        self.top_offset_refplane = 0.416666667 * ws[1]
        self.refang = None
        self.capture = Button(text=CommonSymbols.camera.value[1], pos=(ws[0] - 198, ws[1] - 100))
        self.capture.font_name = CommonSymbols.camera.value[2]
        self.capture.font_size = '50sp'
        # self.capture.bind(on_release=self.switch2setting)
        self.menu = Button(text=CommonSymbols.menu.value[1], pos=(ws[0] - 100, ws[1] - 100))
        self.menu.font_name = "DejaVuSans"
        self.menu.font_size = '50sp'
        self.menu.bind(on_release=self.switch2setting)
        self.cuplabel = Label(text='Cup Orientation', font_size='40sp',
                              pos=(self.left_offset + 0.09375 * ws[0], 0.833333333 * ws[1] - self.top_offset),
                              halign='left')

        self.cuplabelInclination = Label(text='Inclination', font_size='25sp',
                                         pos=(self.left_offset + 45, 525 - self.top_offset),
                                         halign='left')
        self.inclination_data = '-45' + CommonSymbols.Degrees.value[1]
        self.cuplabelInclinationData = Label(text=self.inclination_data, font_size='60sp',
                                             pos=(self.left_offset + 47, 450 - self.top_offset),
                                             halign='left')
        self.cuplabelAnteversion = Label(text='Anteversion', font_size='25sp',
                                         pos=(self.left_offset + 320, 525 - self.top_offset),
                                         halign='left')
        self.anteversionData = '-30' + CommonSymbols.Degrees.value[1]
        self.cuplabelAnteversionData = Label(text=self.anteversionData, font_size='60sp',
                                             pos=(self.left_offset + 322, 450 - self.top_offset),
                                             halign='left')

        self.Reflabel = Label(text='Reference Plane', font_size='40sp',
                              pos=(self.left_offset + 125, 600 - self.top_offset_refplane - self.top_offset),
                              halign='left')
        self.ReflabelInclination = Label(text='Inclination', font_size='25sp',
                                         pos=(self.left_offset + 45, 525 - self.top_offset_refplane - self.top_offset),
                                         halign='left')
        self.Refinclination_data = '75' + CommonSymbols.Degrees.value[1]
        self.ReflabelInclinationData = Label(text=self.Refinclination_data, font_size='60sp',
                                             pos=(
                                                 self.left_offset + 47,
                                                 450 - self.top_offset_refplane - self.top_offset),
                                             halign='left')
        self.ReflabelAnteversion = Label(text='Anteversion', font_size='25sp',
                                         pos=(self.left_offset + 320, 525 - self.top_offset_refplane - self.top_offset),
                                         halign='left')
        self.RefanteversionData = '10' + CommonSymbols.Degrees.value[1]
        self.ReflabelAnteversionData = Label(text=self.RefanteversionData, font_size='60sp',
                                             pos=(
                                                 self.left_offset + 322,
                                                 450 - self.top_offset_refplane - self.top_offset),
                                             halign='left')

        self.cupCheck = CheckBox(active=True, group='target', pos=(self.left_offset_radio + 200, 600 - self.top_offset))
        self.refCheck = CheckBox(active=False, group='target',
                                 pos=(self.left_offset_radio + 200, 300 - self.top_offset))
        self.boo = False
        model_text = 'Mode: ' + 'Target Mode'
        self.modelabel = Label(text=model_text, font_size='18sp',
                               pos=(self.left_offset + 650, -30),
                               halign='left')

        target_text = 'Target (Cup): ' + 'Inclination: ' + str(self.cup_target_data[0]) + CommonSymbols.Degrees.value[
            1] + ', ' + 'Anteversion: ' + str(self.cup_target_data[0]) + CommonSymbols.Degrees.value[1]
        self.targetlabel = Label(text=target_text, font_size='18sp',
                                 pos=(self.left_offset + 950, -30),
                                 halign='left')

        self.titlelabel = Label(text='Target', font_size='60sp',
                                halign='left', color=(200 / 255, 200 / 255, 200 / 255, 1))
        self.titlelabel.pos = (self.left_offset + 65, 0.85 * ws[1])

        self.add_widget(self.labels_holder)
        self.buffer_pipe = []

    def update_labels_postion(self, ws):
        self.left_offset = 0.0234375 * ws[0]
        self.left_offset_radio = 0.1171875 * ws[0]
        self.top_offset = 0.118055556 * ws[1]
        self.top_offset_title = 0.040611111 * ws[1]
        self.top_offset_refplane = 0.416666667 * ws[1]
        self.titlelabel.pos = (self.left_offset + 0.05625 * ws[0], 0.85 * ws[1] + 0.001388888888888889 * 15 * ws[1])
        self.menu.pos = (ws[0] - 100, ws[1] - 100)
        self.capture.pos = (ws[0] - 198, ws[1] - 100)

    def set_imu_info(self, info):
        self.pelvis_info = info

    def enter_stuff(self, e):
        if self.setting_screen is not None:
            boo = False
            try:
                a = int(self.setting_screen.cuplabelInclinationData.text.strip())

            except ValueError:
                boo = True
                a = 0

            boo2 = False
            try:
                b = int(self.setting_screen.cuplabelAnteversionData.text.strip())
            except ValueError:
                boo2 = True
                b = 0
            if not boo:
                target_text = 'Target (Cup): ' + 'Inclination: ' + str(a) + \
                              CommonSymbols.Degrees.value[1]
            else:
                target_text = 'Target (Cup): ' + 'Inclination: INPUT ERROR'
            if not boo2:
                target_text = target_text + ', ' + 'Anteversion: ' + str(b) + CommonSymbols.Degrees.value[1]
            else:
                target_text = target_text + ', ' + 'Anteversion: INPUT ERROR'
            self.targetlabel.text = target_text
            self.cup_target_data = [a, b]
            self.cup_target = [self.cup_target_data[0] / 90 * self.radius, self.cup_target_data[1] / 90 * self.radius]

    def switch2setting(self, e):
        self.setting_screen.enter_stuff(e)
        self.parent.transition.direction = "right"
        self.parent.current = 'Info'

    def draw_targets(self, windows):
        with self.canvas:
            w = windows[0]
            h = windows[1]

            color = (0, 0, 1)
            Color(*color, mode='hsv')
            d = 500
            Ellipse(pos=((0.75 - ((d / 2) / w) + 0.0025) * w, (0.45 - ((d / 2) / h) + 0.005) * h), size=(d, d))

            color = (0, 0, 0.157)
            Color(*color, mode='hsv')
            d = 496
            Ellipse(pos=((0.75 - ((d / 2) / w) + 0.0025) * w, (0.45 - ((d / 2) / h) + 0.005) * h), size=(d, d))

            color = (0, 0, 1)
            Color(*color, mode='hsv')
            d = 250
            Ellipse(pos=((0.75 - ((d / 2) / w) + 0.0025) * w, (0.45 - ((d / 2) / h) + 0.005) * h), size=(d, d))

            color = (0, 0, 0.157)
            Color(*color, mode='hsv')
            d = 246
            Ellipse(pos=((0.75 - ((d / 2) / w) + 0.0025) * w, (0.45 - ((d / 2) / h) + 0.005) * h), size=(d, d))

            color = (200 / 360, 0.498, 1)
            Color(*color, mode='hsv')
            Rectangle(pos=(0.5 * w, 0.45 * h), size=(w, 0.010 * h))

            Color(*color, mode='hsv')
            Rectangle(pos=(0.75 * w, 0.0 * h), size=(0.005 * w, h))

            # center dot
            color = (0, 0, 1)
            Color(*color, mode='hsv')
            d = 40
            Ellipse(pos=((0.75 - ((d / 2) / w) + 0.0025) * w, (0.45 - ((d / 2) / h) + 0.005) * h), size=(d, d))

            color = (200 / 360, 0.498, 1)
            Color(*color, mode='hsv')
            d = 30
            Ellipse(pos=((0.75 - ((d / 2) / w) + 0.0025) * w, (0.45 - ((d / 2) / h) + 0.005) * h), size=(d, d))

            color = (0, 0, 0.10)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.85 * h), size=(w, 0.2 * h))

            color = (0, 0, 0.05)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.85 * h), size=(w, 0.01 * h))

            color = (0, 0, 0.10)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.00 * h), size=(w, 0.07 * h))

            color = (0, 0, 0.05)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.06 * h), size=(w, 0.01 * h))

            # target ----------------------------------------------------------------------------------------#
            color = (0, 0, 1)
            Color(*color, mode='hsv')
            d = 50
            wc = self.boxme[0]
            hc = self.boxme[1]
            Ellipse(pos=(wc[0] + self.cup_target[1] - d / 2, hc[0] + self.cup_target[0] - d / 2), size=(d, d))
            color = (120 / 360, 0.75, 0.8)
            Color(*color, mode='hsv')
            d = 40
            Ellipse(pos=(wc[0] + self.cup_target[1] - d / 2, hc[0] + self.cup_target[0] - d / 2), size=(d, d))
            # ------------------------------------------------------------------------------------------------#

            # the dot ---------------------------------------------------------------------------------------#
            color = (12 / 360, 0.8, 1)
            Color(*color, mode='hsv')
            d = 40
            # Ellipse(pos=((0.70 - ((d / 2) / w) + 0.0025) * w, (0.275 - ((d / 2) / h) + 0.005) * h), size=(d, d))
            wc = self.boxme[0]
            hc = self.boxme[1]
            Ellipse(pos=(wc[0] - d / 2 + self.cup_data[1], hc[0] - d / 2 + self.cup_data[0]), size=(d, d))
            # ------------------------------------------------------------------------------------------------#
            color = (120 / 360, 0.75, 0.8)
            Color(*color, mode='hsv')
            d = 20
            Ellipse(pos=((0.975 - ((d / 2) / w) + 0.0025) * w, (0.025 - ((d / 2) / h) + 0.005) * h), size=(d, d))

    def update(self, windows):
        self.canvas.clear()
        self.clear_widgets()
        self.draw_targets(windows)
        self.update_labels_postion(windows)
        self.add_widget(self.labels_holder)

    def conect2server(self):
        self.imucupo.connect2sever()
        self.imucupo.run()
        self.imupu = threading.Thread(target=self.updatedata, args=())
        self.imupu.start()

    def connect_imu(self):
        if self.imucupo is not None:
            self.imucupo.start2()
        self.imupu = threading.Thread(target=self.updatedata, args=())
        self.imupu.start()

    def disconnect_imu(self):
        self.isclosing = True
        if self.imucupo is not None:
            self.imucupo.stop_imucoms()

    def updater(self, d):
        try:
            ang = d["imu1_ypr"]
            # print(d)
            # print(str(d["t"])+","+str(d["hz"])+","+'{0:3.2f}'.format(ang[0]))
            if self.refang is None:
                # print("hello moto")
                ia = ang[1]
                if ia > 180:
                    ia = ia - 360
                aa = ang[0]
                if aa > 180:
                    aa = aa - 360
                # self.cup_data = [ia / 90 * self.radius - Constants.impactor.value/90 * self.radius, aa / 90 * self.radius]
                self.cup_data = [ia / 90 * self.radius,
                                 aa / 90 * self.radius]
                self.cuplabelInclinationData.text = '{0:3.0f}'.format(ia) + CommonSymbols.Degrees.value[1]
                self.cuplabelAnteversionData.text = '{0:3.0f}'.format(aa) + CommonSymbols.Degrees.value[1]
                self.ReflabelInclinationData.text = '{0:3.0f}'.format(0) + CommonSymbols.Degrees.value[1]
                self.ReflabelAnteversionData.text = '{0:3.0f}'.format(0) + CommonSymbols.Degrees.value[1]
                self.refang = [0, 0, 0]
                self.cup_data = [0 / 90 * self.radius, 0 / 90 * self.radius]
            else:
                print(d)
                if d['c'] == '1':
                    #print("hello roboto: "+str(ang[0]))
                    offsets = [0, 0]
                    if self.pelvis_info is not None:
                        print(self.pelvis_info.euler_deg)
                        if self.pelvis_info.euler_deg is not None:
                            offsets[0] = self.pelvis_info.euler_deg[0]
                            offsets[1] = self.pelvis_info.euler_deg[2]
                    iaa = ang[2] - offsets[1]
                    if iaa > 180:
                        iaa = iaa - 360
                    self.refang[0] = iaa
                    aaa = ang[0] - offsets[0]
                    print(ang)
                    if aaa > 180:
                        aaa = aaa - 360
                    self.refang[1] = aaa
                    # self.cup_data = [ia / 90 * self.radius, aa / 90 * self.radius]
                    self.ReflabelInclinationData.text = '{0:3.0f}'.format(iaa) + CommonSymbols.Degrees.value[1]
                    self.ReflabelAnteversionData.text = '{0:3.0f}'.format(aaa) + CommonSymbols.Degrees.value[1]

                    c = threading.Thread(target=self.update, args=(self.currentwindow,))
                    c.start()

                elif d['c'] == '2':
                    #print("Domo arigato: "+str(ang[0]))
                    ia = ang[1] - self.refang[0]
                    if ia > 180:
                        ia = ia - 360
                    aa = ang[0] - self.refang[1]
                    if aa > 180:
                        aa = aa - 360
                    ia = -ia
                    aa = -aa
                    #print([ia, aa])
                    self.cup_data = [ia / 90 * self.radius, aa / 90 * self.radius]
                    self.cuplabelInclinationData.text = '{0:3.0f}'.format(ia) + CommonSymbols.Degrees.value[1]
                    self.cuplabelAnteversionData.text = '{0:3.0f}'.format(aa) + CommonSymbols.Degrees.value[1]

                    c = threading.Thread(target=self.update, args=(self.currentwindow,))
                    c.start()
            #c.join()
        except KeyError:
            pass
        pass

    def updatedata(self):
        time.sleep(1)
        while True:
            if self.isclosing:
                break
            if self.imucupo is not None:
                d = self.imucupo.read_imu()
                if d is None:
                    continue
                # print("screen updatedata - "+str(d))
                if d is None:
                    break
                if self.pelvis_info is not None:
                    print(self.pelvis_info.euler_deg)
                self.updater(d)
                # c = threading.Thread(target=self.updater, args=(d,))
                # c.start()
                # c.join()
                # try:
                #     ang = d["imu1_ypr"]
                #     #print(d)
                #     #print(str(d["t"])+","+str(d["hz"])+","+'{0:3.2f}'.format(ang[0]))
                #     if self.refang is None:
                #         #print("hello moto")
                #         ia = ang[1]
                #         if ia > 180:
                #             ia = ia - 360
                #         aa = ang[0]
                #         if aa > 180:
                #             aa = aa - 360
                #         #self.cup_data = [ia / 90 * self.radius - Constants.impactor.value/90 * self.radius, aa / 90 * self.radius]
                #         self.cup_data = [ia / 90 * self.radius,
                #                          aa / 90 * self.radius]
                #         self.cuplabelInclinationData.text = '{0:3.0f}'.format(ia) + CommonSymbols.Degrees.value[1]
                #         self.cuplabelAnteversionData.text = '{0:3.0f}'.format(aa) + CommonSymbols.Degrees.value[1]
                #         self.ReflabelInclinationData.text = '{0:3.0f}'.format(0) + CommonSymbols.Degrees.value[1]
                #         self.ReflabelAnteversionData.text = '{0:3.0f}'.format(0) + CommonSymbols.Degrees.value[1]
                #         self.refang = [0, 0]
                #         self.cup_data = [0 / 90 * self.radius, 0 / 90 * self.radius]
                #     else:
                #         print(d)
                #         if d['c'] == '1':
                #             print("hello roboto")
                #             ia = ang[1]
                #             if ia > 180:
                #                 ia = ia - 360
                #             self.refang[0] = ia
                #             aa = ang[0]
                #             if aa > 180:
                #                 aa = aa - 360
                #             self.refang[1] = aa
                #             # self.cup_data = [ia / 90 * self.radius, aa / 90 * self.radius]
                #             self.ReflabelInclinationData.text = '{0:3.0f}'.format(ia) + CommonSymbols.Degrees.value[1]
                #             self.ReflabelAnteversionData.text = '{0:3.0f}'.format(aa) + CommonSymbols.Degrees.value[1]
                #
                #         elif d['c'] == '2':
                #             print("Domo arigato")
                #             ia = ang[1] - self.refang[0]
                #             if ia > 180:
                #                 ia = ia - 360
                #             aa = ang[0] # - self.refang[1]
                #             if aa > 180:
                #                 aa = aa - 360
                #             self.cup_data = [ia / 90 * self.radius, aa / 90 * self.radius]
                #             self.cuplabelInclinationData.text = '{0:3.0f}'.format(ia) + CommonSymbols.Degrees.value[1]
                #             self.cuplabelAnteversionData.text = '{0:3.0f}'.format(aa) + CommonSymbols.Degrees.value[1]
                #     c = threading.Thread(target=self.update, args=(self.currentwindow,))
                #     c.start()
                #     c.join()
                # except KeyError:
                #     pass

    def add_labels(self):
        self.labels_holder.add_widget(self.cuplabel)
        self.labels_holder.add_widget(self.cupCheck)
        self.labels_holder.add_widget(self.refCheck)
        self.labels_holder.add_widget(self.cuplabelInclination)
        self.labels_holder.add_widget(self.cuplabelAnteversion)
        self.labels_holder.add_widget(self.cuplabelInclinationData)
        self.labels_holder.add_widget(self.cuplabelAnteversionData)

        self.labels_holder.add_widget(self.Reflabel)
        self.labels_holder.add_widget(self.ReflabelInclination)
        self.labels_holder.add_widget(self.ReflabelAnteversion)
        self.labels_holder.add_widget(self.ReflabelInclinationData)
        self.labels_holder.add_widget(self.ReflabelAnteversionData)

        self.labels_holder.add_widget(self.titlelabel)
        self.labels_holder.add_widget(self.modelabel)
        self.labels_holder.add_widget(self.targetlabel)
        self.labels_holder.add_widget(self.menu)
        self.labels_holder.add_widget(self.capture)


class InfoScreen(Screen):
    def __init__(self, name='Info', myparent=None, ws=[1280, 720], targetpage=None):
        super(InfoScreen, self).__init__(name=str(name))
        self.wins = ws
        self.draw_background(ws)
        self.target_page = targetpage
        self.myparent = myparent
        self._holder = Widget()
        self.left_offset = 0.0234375 * ws[0]
        self.left_offset_radio = 0.1171875 * ws[0]
        self.top_offset = 0.118055556 * ws[1]
        self.top_offset_title = 0.040611111 * ws[1]
        self.top_offset_refplane = 0.416666667 * ws[1]

        self.capture = Button(text=CommonSymbols.hammer.value[1], pos=(ws[0] - 198, ws[1] - 100))
        self.capture.font_name = CommonSymbols.hammer.value[2]
        self.capture.font_size = '50sp'
        self.capture.bind(on_release=self.switch2start)

        self.menu = Button(text=CommonSymbols.bullseye.value[0], pos=(ws[0] - 100, ws[1] - 100))
        self.menu.font_name = CommonSymbols.bullseye.value[2]
        self.menu.font_size = '50sp'
        self.menu.bind(on_release=self.switch2target)
        self.add_widget(self._holder)

        self.titlelabel = Label(text='Sys Info Menu', font_size='60sp',
                                halign='left', color=(200 / 255, 200 / 255, 200 / 255, 1))
        self.titlelabel.pos = (self.left_offset + 155, 0.85 * ws[1])

        self.cuplabel = Label(text='Set Target Cup Orientation', font_size='40sp',
                              pos=(self.left_offset + 0.165 * ws[0], 0.833333333 * ws[1] - self.top_offset),
                              halign='left')

        self.cuplabelInclination = Label(text='Inclination', font_size='25sp',
                                         pos=(self.left_offset + 45, 525 - self.top_offset),
                                         halign='left')
        self.inclination_data = '00'
        self.cuplabelInclinationData = TextInput(text=self.inclination_data, font_size='55sp',
                                                 pos=(self.left_offset + 47, 450 - self.top_offset),
                                                 halign='center')
        self.cuplabelInclinationData.multiline = False
        padd = self.cuplabelInclinationData.padding
        y_top = self.cuplabelInclinationData.height / 4.0 - (self.cuplabelInclinationData.line_height / 2.0) * len(
            self.cuplabelInclinationData._lines)
        padd[1] = y_top
        padd[3] = 0
        self.cuplabelInclinationData.padding = padd
        # self.spinner_imu1.padding_y = [self.spinner_imu1.height / 4.0 - (self.spinner_imu1.line_height / 2.0) * len(self.spinner_imu1._lines), 0]
        self.cuplabelInclinationDeg = Label(text=CommonSymbols.Degrees.value[1], font_size='55sp',
                                            pos=(self.left_offset + 115, 480 - self.top_offset),
                                            halign='left')

        self.cuplabelAnteversion = Label(text='Anteversion', font_size='25sp',
                                         pos=(self.left_offset + 320, 525 - self.top_offset),
                                         halign='left')
        self.anteversionData = '00'
        self.cuplabelAnteversionData = TextInput(text=self.anteversionData, font_size='55',
                                                 pos=(self.left_offset + 322, 450 - self.top_offset),
                                                 halign='center')
        self.cuplabelAnteversionData.multiline = False
        padd = self.cuplabelAnteversionData.padding
        y_top = self.cuplabelAnteversionData.height / 4.0 - (self.cuplabelAnteversionData.line_height / 2.0) * len(self.cuplabelAnteversionData._lines)
        padd[1] = y_top
        padd[3] = 0
        self.cuplabelAnteversionData.padding = padd
        self.cuplabelAnteversionDeg = Label(text=CommonSymbols.Degrees.value[1], font_size='55sp',
                                            pos=(self.left_offset + 390, 480 - self.top_offset),
                                            halign='left')

        self.Reflabel = Label(text='Override Reference Plane', font_size='40sp',
                              pos=(self.left_offset + 205, 600 - self.top_offset_refplane - self.top_offset),
                              halign='left')
        self.ReflabelInclination = Label(text='Inclination', font_size='25sp',
                                         pos=(self.left_offset + 45, 525 - self.top_offset_refplane - self.top_offset),
                                         halign='left')
        self.Refinclination_data = '00'
        self.ReflabelInclinationData = TextInput(text=self.Refinclination_data, font_size='55sp',
                                                 pos=(
                                                     self.left_offset + 47,
                                                     450 - self.top_offset_refplane - self.top_offset),
                                                 halign='center')
        self.ReflabelInclinationData.multiline = False
        padd = self.ReflabelInclinationData.padding
        y_top = self.ReflabelInclinationData.height / 4.0 - (self.ReflabelInclinationData.line_height / 2.0) * len(
            self.ReflabelInclinationData._lines)
        padd[1] = y_top
        padd[3] = 0
        self.ReflabelInclinationData.padding = padd
        self.ReflabelInclinationDeg = Label(text=CommonSymbols.Degrees.value[1], font_size='55sp',
                                            pos=(
                                            self.left_offset + 115, 480 - self.top_offset - self.top_offset_refplane),
                                            halign='left')
        self.ReflabelAnteversion = Label(text='Anteversion', font_size='25sp',
                                         pos=(self.left_offset + 320, 525 - self.top_offset_refplane - self.top_offset),
                                         halign='left')
        self.RefanteversionData = '00'
        self.ReflabelAnteversionData = TextInput(text=self.RefanteversionData, font_size='55sp',
                                                 pos=(
                                                     self.left_offset + 322,
                                                     450 - self.top_offset_refplane - self.top_offset),
                                                 halign='center')
        self.ReflabelAnteversionData.multiline = False
        padd = self.ReflabelAnteversionData.padding
        y_top = self.ReflabelAnteversionData.height / 4.0 - (self.ReflabelInclinationData.line_height / 2.0) * len(
            self.ReflabelAnteversionData._lines)
        padd[1] = y_top
        padd[3] = 0
        self.ReflabelAnteversionData.padding = padd
        self.ReflabelAnteversionDeg = Label(text=CommonSymbols.Degrees.value[1], font_size='55sp',
                                            pos=(
                                            self.left_offset + 390, 480 - self.top_offset - self.top_offset_refplane),
                                            halign='center')

        self.add_labels()

    def switch2target(self, e):
        self.target_page.enter_stuff(e)
        self.parent.transition.direction = "left"
        self.parent.current = 'Target'

    def switch2start(self, e):
        self.target_page.enter_stuff(e)
        self.parent.transition.direction = "right"
        self.parent.current = 'Start'

    def enter_stuff(self, e):
        if self.target_page is not None:
            print("ahfguiasehghueiargauirhgheruighuiargheighrhegeiau")
            self.cuplabelInclinationData.text = str(self.target_page.cup_target_data[0])
            self.cuplabelAnteversionData.text = str(self.target_page.cup_target_data[1])

        # self.update()

    def draw_background(self, windows):
        with self.canvas:
            w = windows[0]
            h = windows[1]

            color = (0, 0, 0.10)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.85 * h), size=(w, 0.2 * h))

            color = (0, 0, 0.05)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.85 * h), size=(w, 0.01 * h))

            color = (0, 0, 0.10)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.00 * h), size=(w, 0.07 * h))

            color = (0, 0, 0.05)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.06 * h), size=(w, 0.01 * h))

    def update(self):
        self.canvas.clear()
        self.clear_widgets()
        self.draw_background(self.wins)
        self.add_widget(self._holder)

    def add_labels(self):
        self._holder.add_widget(self.menu)
        self._holder.add_widget(self.titlelabel)
        self._holder.add_widget(self.capture)

        self._holder.add_widget(self.cuplabel)
        self._holder.add_widget(self.cuplabelInclination)
        self._holder.add_widget(self.cuplabelAnteversion)
        self._holder.add_widget(self.cuplabelInclinationData)
        self._holder.add_widget(self.cuplabelInclinationDeg)
        self._holder.add_widget(self.cuplabelAnteversionData)
        self._holder.add_widget(self.cuplabelAnteversionDeg)

        self._holder.add_widget(self.Reflabel)
        self._holder.add_widget(self.ReflabelInclination)
        self._holder.add_widget(self.ReflabelAnteversion)
        self._holder.add_widget(self.ReflabelInclinationData)
        self._holder.add_widget(self.ReflabelInclinationDeg)
        self._holder.add_widget(self.ReflabelAnteversionData)
        self._holder.add_widget(self.ReflabelAnteversionDeg)


class StartScreen(Screen):
    def __init__(self, name='Start', myparent=None, ws=[1280, 720], targetpage=None):
        super(StartScreen, self).__init__(name=str(name))
        self.wins = ws
        self.draw_background(ws)
        self.target_page: TargetScreen = targetpage

        self.myparent = myparent
        self._holder = Widget()
        self.left_offset = 0.08 * ws[0]
        self.left_offset_radio = 0.1171875 * ws[0]
        self.top_offset = 0.118055556 * ws[1]
        self.top_offset_title = 0.040611111 * ws[1]
        self.top_offset_refplane = 0.356666667 * ws[1]

        self.target = Button(text=CommonSymbols.bullseye.value[0], pos=(ws[0] - 100, ws[1] - 100))
        self.target.font_name = CommonSymbols.bullseye.value[2]
        self.target.font_size = '50sp'
        self.target.bind(on_release=self.switch2target)

        self.menu = Button(text=CommonSymbols.menu.value[0], pos=(ws[0] - 200, ws[1] - 100))
        self.menu.font_name = CommonSymbols.menu.value[2]
        self.menu.font_size = '50sp'
        self.menu.bind(on_release=self.switch2info)

        self.set_arm = Button(text=CommonSymbols.set_square.value[0], pos=(ws[0] / 2 - 100, ws[1] - 500))
        self.set_arm.font_name = CommonSymbols.set_square.value[2]
        self.set_arm.font_size = '50sp'
        self.set_arm.bind(on_release=self.guide_mode)

        self.scan = Button(text=CommonSymbols.scan_icon.value[0], pos=(ws[0] - 410, ws[1] - 100))
        self.scan.background_color = (0.28, 0.28, 0.28, 1)
        self.scan.font_name = CommonSymbols.scan_icon.value[2]
        self.scan.font_size = '50sp'
        self.scan.bind(on_release=self.scanning_for_boxsens)

        self.showterminal = Button(text=CommonSymbols.computer.value[0], pos=(ws[0] - 310, ws[1] - 100))
        self.showterminal.background_color = (0.28, 0.28, 0.28, 1)
        self.showterminal.font_name = CommonSymbols.computer.value[2]
        self.showterminal.font_size = '50sp'
        self.showterminal.bind(on_release=self.show_log)

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y - %H:%M:%S")

        self.console = TextInput(text="Welcome to Exosense Desktop\nVersion: 0.0.2 alpha\n" + date_time + "\n\n>>",
                                 size_hint=(None, None),
                                 size=(530, 250),
                                 pos=(self.left_offset + ws[0] / 2 - 50, 0.25 * ws[1] - self.top_offset)
                                 )
        self.console.readonly = True

        self.box1 = BoxsensInfo(name="BaseSens", pos=(102+ws[0]/2, 515), console=self.console)
        self.target_page.imucupo.gui = self.box1
        #self.box2 = BoxsensInfo(name="BaseSens 02", pos=(102+ws[0]/2, 515), console=self.console)

        self.box_label = Label(text=CommonSymbols.MINIDISC.value[0], font_size='92sp',
                                halign='left', pos=(self.left_offset - 55, ws[1]/2+93))
        self.box_label.font_name = CommonSymbols.MINIDISC.value[2]
        self.current_box = self.box1

        self.spinner_imu1 = Spinner(
            # default value shown
            text='BaseSens',
            font_size='30sp',
            # available values
            #values=('Scan', self.box1.name, self.box2.name),
            values=('Scan', self.box1.name),
            # just for positioning in our example
            size_hint=(None, None),
            size=(225, 80),
            pos=(self.left_offset + 55, ws[1]/2+100))
        self.spinner_imu1.bind(text=self.active_boxsens)

        self.add_widget(self._holder)

        self.titlelabel = Label(text='Exosens Desktop', font_size='56sp',
                                halign='left', color=(200 / 255, 200 / 255, 200 / 255, 1))
        self.titlelabel.pos = (self.left_offset + 105, 0.85 * ws[1])

        self.Reflabel = Label(text='Pelvis Offset', font_size='40sp',
                              pos=(self.left_offset + 25, 600 - self.top_offset_refplane - self.top_offset),
                              halign='left')
        self.ReflabelInclination = Label(text='X', font_size='25sp',
                                         pos=(self.left_offset-42.5, 525 - self.top_offset_refplane - self.top_offset),
                                         halign='left')
        self.Refinclination_data = '00'
        self.ReflabelInclinationData = TextInput(text=self.Refinclination_data, font_size='55sp',
                                                 pos=(
                                                     self.left_offset - 45,
                                                     450 - self.top_offset_refplane - self.top_offset),
                                                 halign='center')
        self.ReflabelInclinationData.multiline = False
        padd = self.ReflabelInclinationData.padding
        y_top = self.ReflabelInclinationData.height / 4.0 - (self.ReflabelInclinationData.line_height / 2.0) * len(
            self.ReflabelInclinationData._lines)
        padd[1] = y_top
        padd[3] = 0
        self.ReflabelInclinationData.padding = padd
        self.ReflabelInclinationDeg = Label(text=CommonSymbols.Degrees.value[1], font_size='55sp',
                                            pos=(
                                            self.left_offset + 25, 480 - self.top_offset - self.top_offset_refplane),
                                            halign='left')

        self.ReflabelAnteversion = Label(text='Y', font_size='25sp',
                                         pos=(self.left_offset + 120, 525 - self.top_offset_refplane - self.top_offset),
                                         halign='left')
        self.RefanteversionData = '00'
        self.ReflabelAnteversionData = TextInput(text=self.RefanteversionData, font_size='55sp',
                                                 pos=(
                                                     self.left_offset + 122,
                                                     450 - self.top_offset_refplane - self.top_offset),
                                                 halign='center')
        self.ReflabelAnteversionData.multiline = False
        padd = self.ReflabelAnteversionData.padding
        y_top = self.ReflabelAnteversionData.height / 4.0 - (self.ReflabelInclinationData.line_height / 2.0) * len(
            self.ReflabelAnteversionData._lines)
        padd[1] = y_top
        padd[3] = 0
        self.ReflabelAnteversionData.padding = padd
        self.ReflabelAnteversionDeg = Label(text=CommonSymbols.Degrees.value[1], font_size='55sp',
                                            pos=(
                                            self.left_offset + 190, 480 - self.top_offset - self.top_offset_refplane),
                                            halign='center')

        self.ReflabelZ = Label(text='Z', font_size='25sp',
                                         pos=(self.left_offset + 300, 525 - self.top_offset_refplane - self.top_offset),
                                         halign='left')
        self.RefZData = '00'
        self.ReflabelZData = TextInput(text=self.RefanteversionData, font_size='55sp',
                                       pos=(
                                           self.left_offset + 302,
                                           450 - self.top_offset_refplane - self.top_offset),
                                       halign='center')

        self.ReflabelZData.multiline = False
        padd = self.ReflabelZData.padding
        y_top = self.ReflabelZData.height / 4.0 - (self.ReflabelInclinationData.line_height / 2.0) * len(
            self.ReflabelZData._lines)
        padd[1] = y_top
        padd[3] = 0
        self.ReflabelZData.padding = padd
        self.ReflabelZDeg = Label(text=CommonSymbols.Degrees.value[1], font_size='55sp',
                                   pos=(
                                       self.left_offset + 370,
                                       480 - self.top_offset - self.top_offset_refplane),
                                   halign='center')

        self.add_labels()
        self.pelvis = None

    def switch2target(self, e):
        self.target_page.enter_stuff(e)
        self.target_page.pelvis_info = self.pelvis
        self.parent.transition.direction = "left"
        self.parent.current = 'Target'

    def switch2info(self, e):
        self.target_page.enter_stuff(e)
        self.parent.transition.direction = "left"
        self.parent.current = 'Info'

    def active_boxsens(self, spinner, text):
        if text is "Scan":
            self.console.text = self.console.text + " Scanning for BaseSens!\n>>"
        elif text is self.box1.name:
            # self._holder.remove_widget(self.box2)
            try:
                self._holder.add_widget(self.box1)
            except WidgetException:
                pass
            self.console.text = self.console.text + " Displaying "+self.box1.name+"!\n>>"
        # elif text is self.box2.name:
        #     self._holder.remove_widget(self.box1)
        #     try:
        #         self._holder.add_widget(self.box2)
        #     except WidgetException:
        #         pass
        #     self.console.text = self.console.text + " Displaying "+self.box2.name+"!\n>>"

    def show_log(self, e):
        self.console.disabled = not self.console.disabled
        if self.console.disabled:
            self.console.opacity = 0

        else:
            self.console.opacity = 1

    def scanning_for_boxsens(self, e):
        self.console.text = self.console.text + " Scanning ports for BaseSens ... No BaseSens found!\n>>"

    def guide_mode(self, e):
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        if len(filename) > 0:
            self.pelvis = IRScannerInfo(filename)
            self.Refinclination_data = str(self.pelvis.euler_deg[0])
            self.ReflabelInclinationData.text = self.Refinclination_data
            self.RefanteversionData = str(self.pelvis.euler_deg[1])
            self.ReflabelAnteversionData.text = self.RefanteversionData
            self.ReflabelZData.text = str(self.pelvis.euler_deg[2])

        self.console.text = self.console.text + " Entering guide mode ... Cannot find pelvis rig!\n>>" + filename

    def enter_stuff(self, e):
        if self.target_page is not None:
            print("ahfguiasehghueiargauirhgheruighuiargheighrhegeiau")
            self.spinner_imu1.text = str(self.target_page.cup_target_data[0])
            self.IMU_1_indicator.text = str(self.target_page.cup_target_data[1])

        # self.update()

    def draw_background(self, windows):
        with self.canvas:
            w = windows[0]
            h = windows[1]

            color = (0, 0, 0.10)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.85 * h), size=(w, 0.2 * h))

            color = (0, 0, 0.05)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.85 * h), size=(w, 0.01 * h))

            color = (0, 0, 0.10)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.00 * h), size=(w, 0.07 * h))

            color = (0, 0, 0.05)
            Color(*color, mode='hsv')
            Rectangle(pos=(0, 0.06 * h), size=(w, 0.01 * h))

    def update(self):
        self.canvas.clear()
        self.clear_widgets()
        self.draw_background(self.wins)
        self.add_widget(self._holder)

    def add_labels(self):
        self._holder.add_widget(self.target)
        self._holder.add_widget(self.menu)
        self._holder.add_widget(self.titlelabel)
        self._holder.add_widget(self.console)
        self._holder.add_widget(self.scan)
        self._holder.add_widget(self.set_arm)
        self._holder.add_widget(self.showterminal)
        self._holder.add_widget(self.spinner_imu1)
        self._holder.add_widget(self.current_box)
        self._holder.add_widget(self.box_label)

        self._holder.add_widget(self.Reflabel)
        self._holder.add_widget(self.ReflabelInclination)
        self._holder.add_widget(self.ReflabelAnteversion)
        self._holder.add_widget(self.ReflabelZ)

        self._holder.add_widget(self.ReflabelInclinationData)
        self._holder.add_widget(self.ReflabelInclinationDeg)
        self._holder.add_widget(self.ReflabelAnteversionData)
        self._holder.add_widget(self.ReflabelAnteversionDeg)
        self._holder.add_widget(self.ReflabelZData)
        self._holder.add_widget(self.ReflabelZDeg)
        # self._holder.add_widget(self.box1)
        # self._holder.add_widget(self.box2)
