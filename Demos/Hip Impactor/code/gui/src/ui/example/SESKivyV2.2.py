import numpy as np
from random import random
from enum import Enum

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
# noinspection PyUnresolvedReferences
from kivy.graphics import Color, Ellipse, Line, Rectangle  # there is a bug in pycharm, the added line above
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image

from playsound import playsound     # using playsound instead of kivy because there is a bug in the kivy gstreamer lib

import time
import threading
import _thread

##
#   This app is a modification of the example app from https://kivy.org/docs/tutorials/firstwidget.html
#   to record touch data and saving the data to a file with the trial configuration
#   Requires Kivy Python 3.6
#   Kivy installation guide - https://kivy.org/docs/installation/installation-windows.html

##
#   Playing Sound will require playsound module


class Tools(object):
    # General tools used in the script
    @staticmethod
    def time_millie():
        return int(round(time.time() * 1000))


###
#  global variables


start_time = Tools.time_millie()  # start time rounded
startTime = time.time()  # start time

####


class Sides(Enum):
    left = 1
    right = 2


class MyPaintWidget(Widget):

    def __init__(self):
        super(MyPaintWidget, self).__init__()
        self.mydict = {}
        self.hand = {Sides.left:[], Sides.right:[]}
        self.line = {}
        self.on = 0
        self.hz = 1

    def clear(self):
        self.mydict = {}
        self.hand = {Sides.left: [], Sides.right: []}
        self.line = {}
        self.on = 0
        self.hz = 1

    def on_touch_down(self, touch):
        winme = Window.size
        width = winme[0]
        if self.on:
            color = (random(), 1, 1)
            with self.canvas:
                Color(*color, mode='hsv')
                d = 30.
                Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
                touch.ud['line'] = Line(points=(touch.x, touch.y))
                t = Tools.time_millie()-start_time
                if touch.x < width/2:
                    self.line[touch] = Sides.left
                    self.hand[Sides.left].append([t, self.hz, touch.x, touch.y])
                else:
                    self.line[touch] = Sides.right
                    self.hand[Sides.right].append([t, self.hz, touch.x, touch.y])
                self.mydict[touch] = [[t, self.hz, touch.x, touch.y]]

    def on_touch_move(self, touch):
        if self.on:
            touch.ud['line'].points += [touch.x, touch.y]
            side = self.line[touch]
            t = Tools.time_millie() - start_time
            self.hand[side].append([t, self.hz, touch.x, touch.y])
            self.mydict[touch].append([t, self.hz, touch.x, touch.y])


class ButtonBar(Widget):
    # This class manages the button, the status of the trial and the metronome
    #
    def __init__(self, myParent):
        super(ButtonBar, self).__init__()
        self.wind = Window.size
        self.sound = 'ElevatorDing2.wav'
        self.Openb = Button(text='Open/Create Trial', size=(200, 50))  # not used yet
        self.Openb.bind(on_release=self.OpenTrial)
        self.play = Button(text='Start', size=(100, 50))  # , pos=(205,0))
        self.play.bind(on_release=self.PlayTrial)
        self.startLabel = Label(text='', font_size='30sp', pos=(self.wind[0]*0.875, int(self.wind[1] * 0.95)), halign='left')
        self.stateLabel = Label(text='', font_size='30sp', pos=(self.wind[0]/2, int(self.wind[1] * 1.55)),
                                halign='left')

        self.add_widget(self.play)
        self.add_widget(self.stateLabel)
        self.add_widget(self.startLabel)

        self.myParent = myParent
        self.settings = None
        self.tickCount = 0
        self.isPlay = 1
        self.hz = 1
        self.sibling = None
        self.index = 0
        self.count = 0
        self.boo = True
        self.red_dot = True

    def OpenTrial(self, event):
        print("config")

    def update(self):
        self.wind = Window.size
        self.startLabel.x = self.wind[0] * 0.45
        self.startLabel.y = int(self.wind[1] * 0.9)

    def clear_status(self):
        self.startLabel.text = ""
        self.update()

    def tictok(self):
        global startTime

        if self.settings is not None:
            if startTime < 0:
                t_start = time.time()
                time_diff = int(time.time() - t_start)
                while (5 - time_diff) > 0:
                    time_diff = int(time.time() - t_start)
                    self.startLabel.text = "Ready ... " + str(5 - time_diff)
            print("starting")
            if startTime < 0:
                startTime = time.time()

            _thread.start_new_thread(playsound, (self.sound,))
            if "Ready" in self.startLabel.text:
                self.startLabel.text = "Recording"
            time_diff = int(time.time() - startTime)

            remainder = time_diff % float(1)
            if self.startLabel.text == "Recording" and remainder == 0.0 and time_diff > 0:
                if self.red_dot:
                    self.draw_red_dot()
                    self.red_dot = False
                else:
                    self.draw_red_dot(brightness=0)
                    self.red_dot = True

            self.index += 1

            self.stateLabel.text = "Freq: " + '{0:.02f}'.format(self.hz) + " Hz" + "     Time: " + '{0:02d}'.format(time_diff) + " s"
            self.stateLabel.x = self.wind[0]*0.9 - self.stateLabel.width
            self.stateLabel.y = self.play.y - self.play.height / 2

            if self.index > 19:
                self.index = 0

            remainder = time_diff % float(self.settings.duration)
            print(time_diff)
            if remainder == 0 and self.boo and self.isPlay == 1 and time_diff > 1:
                if self.tickCount < self.settings.increment:
                    self.hz = self.settings.init_freq + self.settings.step*(self.tickCount + 1)
                    self.sibling.hz = self.hz
                    self.tickCount += 1
                    self.boo = False
                else:
                    self.isPlay = 0

            else:
                self.boo = True

        if self.isPlay is 1:
            next = 1 / self.hz
            t = threading.Timer(next, self.tictok)
            t.start()
            t.join()
        else:
            self.draw_red_dot(brightness=0)
            self.stateLabel.text = ""
            self.stateLabel.x = self.play.x + self.play.width + self.play.width * 0.12
            self.stateLabel.y = int(self.wind[1] * -0.02)
            if self.play.text is "Processing":
                self.startLabel.text = "Saving ..."
            else:
                self.startLabel.text = "Saved"
                t = threading.Timer(2, self.clear_status)
                t.start()
            self.startLabel.x = self.wind[0] * 0.45
            self.startLabel.y = int(self.wind[1] * 0.9)

    def PlayTrial(self, event):
        global startTime

        if self.play.text is "Start":
            startTime = -1
            self.update()
            self.startLabel.text = "Ready ...  "
            self.isPlay = 1
            self.tickCount = 0
            self.play.text = 'Save'
            self.myParent.StartTrial()
            self.myParent.painter.on = 1
            self.hz = self.settings.init_freq
            #t = threading.Timer(1 / self.settings.init_freq, self.tictok)
            t = threading.Timer(0, self.tictok)
            t.start()
        else:
            self.draw_progress_bar(percent=0.1)
            self.draw_red_dot(brightness=0)
            self.stateLabel.text = ""
            self.startLabel.text = "Saving ..."
            self.startLabel.x = int(self.wind[0] * 0.92)
            self.startLabel.y = int(self.wind[1] * 0.52)
            self.myParent.painter.on = 0
            self.play.text = 'Processing'
            self.play.disabled = True

            self.isPlay = 0
            self.myParent.clear_canvas(self)

    def draw_progress_bar(self, percent=0.5, brightness=1):
        with self.canvas:
            w = Window.size[0]
            h = Window.size[1]
            color = (0.5, 1, brightness)
            Color(*color, mode='hsv')
            Rectangle(pos=(0.07*w, 0.005*h), size=(percent*w-0.06*w, 0.041*h))

    def draw_red_dot(self, brightness=0.68):
        with self.canvas:
            color = (0.97, 0.91, brightness)
            Color(*color, mode='hsv')
            Ellipse(pos=(self.startLabel.x+self.startLabel.width*1.25, self.startLabel.y+self.startLabel.height*0.39), size=(25, 25))

    def set_play_button(self, text="Start"):
        self.play.text = text
        self.play.disabled = False
        self.startLabel.text = ""
        self.draw_progress_bar(percent=1, brightness=0)
        self.myParent.painter.on = 0


class Popupwig(Widget):
    # This class contains the UI elements for
    def __init__(self, myParent):
        super(Popupwig, self).__init__()
        w = Window.size[0]
        h = Window.size[1]

        self.title_label = Label(text='Options', pos=(0, h*0.88571))

        line_height = 565
        self.initFeqLabel = Label(text='Initial Frequency:', pos=(30, line_height), halign='left')
        self.initFeqInput = TextInput(text='1', pos=(200,line_height+35), size=(200,30))
        self.initFeqUnit = Label(text='Hz', pos=(400, line_height))

        line_height = 535
        self.incrementLabel = Label(text='Increment @ Step:', pos=(30, line_height), halign='left')
        self.incrementInput = TextInput(text='4 @ 0.25', pos=(200, line_height+35), size=(200, 30))
        self.IncrementUnit = Label(text='Steps / Hz', pos=(400, line_height))

        line_height = 505
        self.incrementDLabel = Label(text='Duration @ Freq:', pos=(30, line_height), halign='left')
        self.incrementDInput = TextInput(text='10', pos=(200, line_height + 35), size=(200, 30))
        self.IncrementDUnit = Label(text='Seconds', pos=(400, line_height))

        line_height = 475
        self.ODLabel = Label(text='Output Directory:', pos=(30, line_height), halign='left')
        self.ODInput = TextInput(text='N/A', pos=(200, line_height + 35), size=(300, 30))
        self.title_label2 = Label(text='Participant and Experimenter', pos=(70, 420))

        line_height = 370
        self.subjectLabel = Label(text='Participant:', pos=(30, line_height), halign='left')
        self.subjectName = TextInput(text='Who?', pos=(200, line_height + 35), size=(300, 30))

        line_height = 340
        self.epLabel = Label(text='Experimenter:', pos=(30, line_height), halign='left')
        self.epName = TextInput(text='Who?', pos=(200, line_height + 35), size=(300, 30))

        self.title_label3 = Label(text='Trial Information', pos=(30, 280))

        line_height = 230
        self.tnLabel = Label(text='Trial Number:', pos=(30, line_height), halign='left')
        self.tnName = TextInput(text='1', pos=(200, line_height + 35), size=(300, 30))

        line_height = 200
        self.tdLabel = Label(text='Trial Description:', pos=(30, line_height), halign='left')
        self.tdName = TextInput(text='What?', pos=(200, line_height - 35), size=(300, 100))

        #self.wimg = Image(source='mylogo.png', pos=(700, 300), size=(300, 300))
        self.wimg = Image(source='abi-vrw-rgb.png', pos=(700, 300), size=(300, 300))

        self.add_widget(self.wimg)

        self.add_widget(self.title_label)
        self.add_widget(self.initFeqLabel)
        self.add_widget(self.initFeqUnit)
        self.add_widget(self.initFeqInput)

        self.add_widget(self.incrementLabel)
        self.add_widget(self.IncrementUnit)
        self.add_widget(self.incrementInput)

        self.add_widget(self.incrementDLabel)
        self.add_widget(self.IncrementDUnit)
        self.add_widget(self.incrementDInput)

        self.add_widget(self.ODLabel)
        self.add_widget(self.ODInput)
        #self.add_widget(self.openDir)

        self.add_widget(self.title_label2)
        self.add_widget(self.subjectLabel)
        self.add_widget(self.subjectName)

        self.add_widget(self.epLabel)
        self.add_widget(self.epName)

        self.add_widget(self.tdLabel)
        self.add_widget(self.tdName)

        self.add_widget(self.tnLabel)
        self.add_widget(self.tnName)

        self.add_widget(self.title_label3)
        #self.draw_divider()

    def getSetup(self):
        inFq = float(self.initFeqInput.text)
        icre_data = str(self.incrementInput.text).split("@")
        count = float(icre_data[0].strip())
        step_size = float(icre_data[1].strip())
        duration = float(self.incrementDInput.text)

        outdir = str(self.ODInput.text)

        subject_name = self.subjectName.text
        experimenter = self.epName.text

        trial_num = float(self.tnName.text)
        trial_desc = self.tdName.text

        return DataSummary(inFq, count, step_size, duration, outdir, subject_name, experimenter, trial_num, trial_desc)

    def update(self):
        self.clear_widgets()
        self.canvas.clear()
        w = Window.size[0]
        h = Window.size[1]

        self.title_label = Label(text='Options', pos=(0, h * 0.8857142857142857+0.0071428571428571*h))

        line_height = 0.8071428571428571*h
        self.initFeqLabel = Label(text='Initial Frequency:', pos=(30, line_height), halign='left')
        self.initFeqInput = TextInput(text='1', pos=(200, line_height + 35), size=(200, 30))
        self.initFeqUnit = Label(text='Hz', pos=(400, line_height))

        line_height = 0.7642857142857143*h
        self.incrementLabel = Label(text='Increment @ Step:', pos=(30, line_height), halign='left')
        self.incrementInput = TextInput(text='4 @ 0.25', pos=(200, line_height + 35), size=(200, 30))
        self.IncrementUnit = Label(text='Steps / Hz', pos=(400, line_height))

        line_height = 0.7214285714285714*h
        self.incrementDLabel = Label(text='Duration @ Freq:', pos=(30, line_height), halign='left')
        self.incrementDInput = TextInput(text='10', pos=(200, line_height + 35), size=(200, 30))
        self.IncrementDUnit = Label(text='Seconds', pos=(400, line_height))

        line_height = 0.6785714285714286*h
        self.ODLabel = Label(text='Output Directory:', pos=(30, line_height), halign='left')
        self.ODInput = TextInput(text='N/A', pos=(200, line_height + 35), size=(300, 30))
        # self.openDir = Button(text='+', size=(30, 30), pos=(400, line_height + 34))
        # self.openDir.bind(on_release=self.openDirEvent)

        self.title_label2 = Label(text='Participant and Experimenter', pos=(70, 0.6*h+0.0071428571428571*h))

        line_height = 0.5285714285714286*h
        self.subjectLabel = Label(text='Participant:', pos=(30, line_height), halign='left')
        self.subjectName = TextInput(text='Who?', pos=(200, line_height + 35), size=(300, 30))

        line_height = 0.4857142857142857*h
        self.epLabel = Label(text='Experimenter:', pos=(30, line_height), halign='left')
        self.epName = TextInput(text='Who?', pos=(200, line_height + 35), size=(300, 30))

        self.title_label3 = Label(text='Trial Information', pos=(30, 0.4*h+0.0071428571428571*h))

        line_height = 0.3285714285714286*h
        self.tnLabel = Label(text='Trial Number:', pos=(30, line_height), halign='left')
        self.tnName = TextInput(text='1', pos=(200, line_height + 35), size=(300, 30))

        line_height = 0.2857142857142857*h
        self.tdLabel = Label(text='Trial Description:', pos=(30, line_height), halign='left')
        self.tdName = TextInput(text='What?', pos=(200, line_height - 35), size=(300, 100))

        #self.wimg = Image(source='mylogo.png', pos=(0.5833333333333333*w, 0.4285714285714286*h), size=(300, 300))
        self.wimg = Image(source='uoa-vrw-rgb.png', pos=(0.5833333333333333 * w, 0.4285714285714286 * h), size=(500, 500))

        self.add_widget(self.wimg)

        self.add_widget(self.title_label)
        self.add_widget(self.initFeqLabel)
        self.add_widget(self.initFeqUnit)
        self.add_widget(self.initFeqInput)

        self.add_widget(self.incrementLabel)
        self.add_widget(self.IncrementUnit)
        self.add_widget(self.incrementInput)

        self.add_widget(self.incrementDLabel)
        self.add_widget(self.IncrementDUnit)
        self.add_widget(self.incrementDInput)

        self.add_widget(self.ODLabel)
        self.add_widget(self.ODInput)
        # self.add_widget(self.openDir)

        self.add_widget(self.title_label2)
        self.add_widget(self.subjectLabel)
        self.add_widget(self.subjectName)

        self.add_widget(self.epLabel)
        self.add_widget(self.epName)

        self.add_widget(self.tdLabel)
        self.add_widget(self.tdName)

        self.add_widget(self.tnLabel)
        self.add_widget(self.tnName)

        self.add_widget(self.title_label3)
        self.draw_divider()

    def openDirEvent(self,obj):
        f = FileChooserListView(pos=(550,100), size=(500,600))
        self.add_widget(f)

    def draw_divider(self):
        with self.canvas:
            w = Window.size[0]
            h = Window.size[1]
            color = (0.5, 1, 1)
            Color(*color, mode='hsv')
            Rectangle(pos=(0.0166666666666667*w, 0.9285714285714286*h), size=(0.4166666666666667*w, 5))
            Rectangle(pos=(0.0166666666666667*w, 0.6428571428571429*h), size=(0.4166666666666667*w, 5))
            Rectangle(pos=(0.0166666666666667*w, 0.4428571428571429*h), size=(0.4166666666666667*w, 5))


class DataSummary(object):
    def __init__(self, ifq, icr, ste, dur, odi, subj, experi,tn,td):
        self.init_freq = ifq
        self.increment = icr
        self.step = ste
        self.duration = dur

        self.outdir = odi

        self.participant = subj
        self.experimenter = experi

        self.trial_number = tn
        self.trial_description = td

    def __str__(self):
        outstr = "<---Trial_Configuration--->\n"
        outstr += "Trial Number: " + str(self.trial_number) + "\n"
        outstr += "Trial Description: " + str(self.trial_description) + "\n"
        outstr += "Participant: " + str(self.participant) + "\n"
        outstr += "Experimenter: " + str(self.experimenter) + "\n"
        outstr += "Inital Frequency: " + str(self.init_freq) + "\n"
        outstr += "Step: " + str(self.step) + "\n"
        outstr += "Number of Increments: " + str(self.increment) + "\n"
        outstr += "Increment Duration: " + str(self.duration) + "\n"
        outstr += "<----------------------->\n"
        return outstr


class Carrier(object):
    def __init__(self, key, data):
        self.keys = key
        self.data = data


class MyTouchApp(App):
    def __init__(self, w=1200, h=700):
        super(MyTouchApp, self).__init__()
        self.parent = None

        self.painter = None
        self.settingPage = None
        self.bar = None
        self.leftHand = []
        self.righHand = []
        Window.maximize()
        w = Window.size[0]
        h = Window.size[1]
        r = (h * 0.75)
        self.center_left = [w * 0.01 + r / 2.0, r / 4.5 + r / 2]
        self.center_right = [w * 0.55 + r / 2.0, r / 4.5 + r / 2]
        self.update(w, h)
        self.length_of_cycle = 20  # min number of points per cycle to be include

    def update(self, w, h):
        r = (h * 0.75)
        self.center_left = [w * 0.01 + r / 2.0, r / 4.5 + r / 2]
        self.center_right = [w * 0.55 + r / 2.0, r / 4.5 + r / 2]

    def on_stop(self):
        self.bar.isPlay = 0

    # Resize Window Handler
    def win_cb(self, window, width, height):
        self.update(width, height)
        self.settingPage.update()

    def build(self):
        self.parent = Widget()

        self.painter = MyPaintWidget()
        self.settingPage = Popupwig(self)
        self.bar = ButtonBar(self)

        Window.bind(on_resize=self.win_cb)

        startbtn = Button(text='Start')
        startbtn.bind(on_release=self.StartTrial)

        clearbtn = Button(text='Save')
        clearbtn.bind(on_release=self.clear_canvas)

        self.parent.add_widget(self.painter)
        self.parent.add_widget(self.bar)
        self.parent.add_widget(self.settingPage)

        return self.parent

    def Config(self):
        print("config")

    def StartTrial(self):
        self.remove_form()
        self.DrawCanvas()

    def remove_form(self):
        data = self.settingPage.getSetup()
        self.bar.settings = data
        self.bar.sibling = self.painter
        self.parent.remove_widget(self.settingPage)

    def DrawCanvas(self):
        with self.painter.canvas:
            color = (0, 0, 1)
            Color(*color, mode='hsv')
            # Scaling for the circles may need to be readjusted for screen smaller then 1920 by 1080
            r = (Window.size[1])*0.75
            w = (Window.size[0])
            Ellipse(pos=((w-r)*0.05, r/4.5), size=(r, r))
            Ellipse(pos=((w-r)*0.95, r/4.5), size=(r, r))
            color = (0.8, 1, 1)
            Color(*color, mode='hsv')
            Rectangle(pos=((w-r)*0.05, r/1.4), size=(r, 10))
            Rectangle(pos=((w-r)*0.95, r/1.4), size=(r, 10))

    def processing_tagging_cycles(self, hand, side):
        print("processing data")

        center = self.center_left
        if side is Sides.right:
            center = self.center_right
        if len(hand) > 0:
            num_of_samples = len(hand)
            start = np.array(hand[0][2:4])
            t = [hand[i][0] for i in range(0, num_of_samples)]
            hz = [hand[i][1] for i in range(0, num_of_samples)]
            threshold = np.linalg.norm(start - center)
            pixel = [hand[i][2:4] for i in range(0, num_of_samples)]
            mags = [np.linalg.norm(np.array(hand[i][2:4]) - start) for i in range(0, num_of_samples)]
            mags_filter = [i if i > threshold else 0 for i in mags]
            min_filter = [0] * num_of_samples
            gradient = [0] * num_of_samples
            sliding_window = 10
            for i in range(int(sliding_window/2), len(hand)-int(sliding_window/2)):
                current_data_window = mags_filter[i-int(sliding_window/2):i+int(sliding_window/2)]
                min_filter[i] = min(current_data_window)

            for i in range(1, num_of_samples - 1):
                g = min_filter[i+1] - min_filter[i-1]
                a = t[i+1] - t[i-1]
                gradient[i] = (g/a)
                pass

            mask = [1 if i == 0 else 0 for i in gradient]
            stands_count = []
            stands = []
            stand = 0
            for i in range(0, len(mask)):
                if i > 0 and mask[i] == 1:
                    stands_count.append(stands_count[len(stands_count)-1]+mask[i])
                    stand += mask[i]
                else:
                    if stand > 10:
                        stands.append([t[round(i-stand/2)], hz[round(i-stand/2)], stand])
                        stand = 0
                    stands_count.append(mask[i])

                pass
            data = {"time":t,
                    "hz": hz,
                    "raw_data": pixel,
                    "cycles": stands}

            return data

    def processing_ratios(self, data):
        if data is not None:
            hz = list(set(data["hz"]))
            hz.sort()
            sort_data_by_hz = {}
            for i in hz:
                sort_data_by_hz[i] = {"time": [],
                                      "points": []}
            for i in range(0, len(data["hz"])):
                h_key = data["hz"][i]
                sort_data_by_hz[h_key]["time"].append(data["time"][i])
                sort_data_by_hz[h_key]["points"].append(data["raw_data"][i])
            # calculate overall ratio
            for i in hz:
                x = [point[0] for point in sort_data_by_hz[i]["points"]]
                y = [point[1] for point in sort_data_by_hz[i]["points"]]
                pair = [abs(max(x)-min(x)), abs(max(y)-min(y))]
                sort_data_by_hz[i]["overall_ratio"] = min(pair)/max(pair)
                sort_data_by_hz[i]["freq"] = []

            # ratio per cycle
            start = data["time"][0]
            cycle_ratios = []
            for c in data["cycles"]:
                end_time = c[0]
                hz_data = sort_data_by_hz[c[1]]
                cycle = []
                for h in range(0, len(hz_data["time"])):
                    if hz_data["time"][h] >= start and hz_data["time"][h] < end_time:
                        cycle.append(hz_data["points"][h])
                x = [point[0] for point in cycle]
                y = [point[1] for point in cycle]
                ratio = -1
                try:
                    pair = [abs(max(x) - min(x)), abs(max(y) - min(y))]
                    ratio = min(pair) / max(pair)
                except ValueError:
                    pass
                dt = float(end_time - start)/1000
                f = 1/dt
                sort_data_by_hz[c[1]]["freq"].append(float(f))
                cyc = {"start_time": start,
                       "end_time": start,
                       "Metronome_hz": c[1],
                       "user_hz": f,
                       "ratio": ratio}
                start = end_time
                cycle_ratios.append(cyc)
            ret = {"by_hz": sort_data_by_hz,
                   "by_cycles": cycle_ratios}
            return ret

    def processing_analysis(self, trigger=None):
        if trigger is not None:
            trigger.draw_progress_bar(percent=0.1)

        left = self.processing_tagging_cycles(self.painter.hand[Sides.left], Sides.left)
        right = self.processing_tagging_cycles(self.painter.hand[Sides.right], Sides.right)

        if trigger is not None:
            trigger.draw_progress_bar(percent=0.3)

        out_left = self.processing_ratios(left)
        out_right = self.processing_ratios(right)

        if trigger is not None:
            trigger.draw_progress_bar(percent=0.7)

        return {"raw left": left,
                "raw right": right,
                "result left": out_left,
                "result right": out_right,
                }

    def print_results(self, data):
        ret_left = "<---Trial_Summary--->\n"
        ret_left += "Metronome_hz, user_hz, ratio \n"
        if data["result left"] is not None:
            for d in data["result left"]["by_hz"]:
                ret_left += str(d)+","
                f = data["result left"]["by_hz"][d]["freq"]
                if len(f) > 0:
                    ret_left += str(sum(f) / len(f))+","
                    ret_left += str(data["result left"]["by_hz"][d]["overall_ratio"])
                    ret_left += "\n"
            ret_left += "<---Trial_Result_by_cycles--->\n"
            ret_left += "start_time, end_time, Metronome_hz, user_hz, ratio\n"
            for d in data["result left"]["by_cycles"]:
                ret_left += str(d["start_time"])+","
                ret_left += str(d["end_time"])+","
                ret_left += str(d["Metronome_hz"]) + ","
                ret_left += str(d["user_hz"]) + ","
                ret_left += str(d["ratio"])
                ret_left += "\n"
            ret_left += "<---Trial_Data--->\n"
            ret_left += "time, Metronome_hz, x, y\n"
            for d in range(0, len(data["raw left"]["time"])):
                ret_left += str(data["raw left"]["time"][d]) + ","
                ret_left += str(data["raw left"]["hz"][d]) + ","
                ret_left += str(data["raw left"]["raw_data"][d][0]) + ","
                ret_left += str(data["raw left"]["raw_data"][d][1]) + "\n"

        ret_right = "<---Trial_Summary--->\n"
        ret_right += "Metronome_hz, user_hz, ratio \n"
        if data["result right"] is not None:
            for d in data["result right"]["by_hz"]:
                ret_right += str(d) + ","
                f = data["result right"]["by_hz"][d]["freq"]
                ret_left += str(sum(f) / len(f)) + ","
                ret_right += str(data["result right"]["by_hz"][d]["overall_ratio"])
                ret_right += "\n"
            ret_right += "<---Trial_Result_by_cycles--->\n"
            ret_right += "start_time, end_time, Metronome_hz, user_hz, ratio\n"
            for d in data["result right"]["by_cycles"]:
                ret_right += str(d["start_time"]) + ","
                ret_right += str(d["end_time"]) + ","
                ret_right += str(d["Metronome_hz"]) + ","
                ret_right += str(d["user_hz"]) + ","
                ret_right += str(d["ratio"])
                ret_right += "\n"
            ret_right += "<---Trial_Data--->\n"
            ret_right += "time, Metronome_hz, x, y\n"
            for d in range(0, len(data["raw right"]["time"])):
                ret_right += str(data["raw right"]["time"][d]) + ","
                ret_right += str(data["raw right"]["hz"][d]) + ","
                ret_right += str(data["raw right"]["raw_data"][d][0]) + ","
                ret_right += str(data["raw right"]["raw_data"][d][1]) + "\n"
        return {"left": ret_left, "right": ret_right}

    def writeout(self, data):
        out_dir = self.bar.settings.outdir + "\\"
        if self.bar.settings.outdir == "N/A":
            out_dir = ""
        filename = out_dir + "Trial" + str(int(self.bar.settings.trial_number)) + ".Left.txt"
        try:
            file = open(filename, "w")
            file.write("Hand: Left Hand\n\n")
            file.write(self.bar.settings.__str__())
            file.write(data["left"])
            file.close()
        except PermissionError:
            print("cannot not save file is open")

        filename = out_dir + "Trial" + str(int(self.bar.settings.trial_number)) + ".Right.txt"
        try:
            file = open(filename, "w")
            file.write("Right Hand\n\n")
            file.write(self.bar.settings.__str__())
            file.write(data["right"])
            file.close()
        except PermissionError:
            print("cannot not save file is open")

    def clear_canvas(self, trigger=None):
        data = self.processing_analysis(trigger)
        if trigger is not None:
            trigger.draw_progress_bar(percent=0.8)
        str_data = self.print_results(data)
        self.writeout(str_data)
        if trigger is not None:
            trigger.draw_progress_bar(percent=1)
        trigger.set_play_button(text="Start")
        self.painter.canvas.clear()
        self.painter.clear()
        self.parent.add_widget(self.settingPage)
        # if trigger is not None:
        #     trigger.draw_progress_bar(percent=0.2)
        # spike = self.processing_tagging_cycles(self.painter.hand[Sides.left], Sides.left)
        # if trigger is not None:
        #     trigger.draw_progress_bar(percent=0.4)
        #
        #
        # dir = self.bar.settings.outdir+"\\"
        # if self.bar.settings.outdir == "N/A":
        #     dir = ""
        #
        # filename = dir+"Trial"+str(int(self.bar.settings.trial_number))+".Left.txt"
        # file = open(filename, "w")
        # file.write(self.bar.settings.__str__())
        # file.write("Left Hand\n\n")
        #
        # L = len(self.painter.hand[Sides.left])
        # dataL = self.analysis(Sides.left,spike,L)
        #
        # file.write("\n\nSummary: \n\n")
        # for key in dataL.keys:
        #     file.write(str(key)+": "+str(dataL.data[key]))
        #     file.write("\n")
        #
        # file.write("\n\nData: \n\n")
        # for i in range(0,L):
        #     l = self.painter.hand[Sides.left][i]
        #     sp = 0
        #     if i >= 3 and i < L-3:
        #         sp = spike[i-3]
        #     file.write(str(l[0]) + "," + str(l[1]) + "," + str(l[2])+ "," + str(l[3]) + ", " + str(sp)+"\n")
        #
        # file.close()
        # print("done - left hand")
        # if trigger is not None:
        #     trigger.draw_progress_bar(percent=0.5)
        # spike = self.processing_tagging_cycles(self.painter.hand[Sides.right], Sides.right)
        # if trigger is not None:
        #     trigger.draw_progress_bar(percent=0.7)
        # filename = dir + "Trial" + str(int(self.bar.settings.trial_number)) + ".Right.txt"
        # file = open(filename, "w")
        # file.write(self.bar.settings.__str__())
        # file.write("Right Hand\n\n")
        #
        # R = len(self.painter.hand[Sides.right])
        # dataR = self.analysis(Sides.right, spike, R)
        #
        # file.write("\n\nSummary: \n\n")
        # for key in dataR.keys:
        #     file.write(str(key)+": "+str(dataR.data[key]))
        #     file.write("\n")
        #
        # file.write("\n\nData: \n\n")
        # if trigger is not None:
        #     trigger.draw_progress_bar(percent=0.8)
        # for i in range(0, R):
        #     r = self.painter.hand[Sides.right][i]
        #     sp = 0
        #     if i >= 3 and i < R - 3:
        #         sp = spike[i - 3]
        #     file.write( str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "," + str(r[3]) + ", " + str(sp) + "\n")
        #
        # file.close()

    #
    # def processing(self, hand, side):
    #     print("processing data")
    #     L = len(hand)
    #     start = [0, 0]
    #     angles = []
    #     center = self.center_left
    #     if side is Sides.right:
    #         center = self.center_right
    #     for i in range(0, L):
    #         l = hand[i]
    #         c = np.array(center)
    #         v = np.array(l[2:4])
    #         e = v-c
    #         if i == 0:
    #             start = e
    #         else:
    #             ab = np.dot(start, e)
    #             angle = np.arccos(ab/(np.linalg.norm(start)*np.linalg.norm(e)))
    #             angles.append(angle)
    #
    #     angular_v = []
    #     for i in range(0,len(angles)-2):
    #         angular_v.append(angles[i+2]-angles[i])
    #
    #     pulse = []
    #     for i in range(0,len(angular_v)):
    #         if angular_v[i] > 0:
    #             pulse.append(3.1415)
    #         else:
    #             pulse.append(-3.1415)
    #
    #     spike = []
    #     for i in range(0,len(pulse)-1):
    #         #print (pulse[i+1]-pulse[i])
    #         spike.append(abs(pulse[i+1]-pulse[i]))
    #
    #     return spike
    #
    # def analysis(self,side,spikes, L):
    #     start_hand_time = 0
    #     average_time = 0
    #     max_x = -10000000.0
    #     min_x = 100000000.0
    #
    #     max_y = -10000000.0
    #     min_y = 100000000.0
    #
    #     spike_count = 0
    #     cycles = 0
    #     id = 0
    #     sp = 0
    #     ratio_avg = []
    #     ratio = 1
    #     data_left = {}
    #     data_keys = []
    #     count_time = []
    #     #print ("start of loop")
    #     for i in range(0, L):
    #         l = self.painter.hand[side][i]
    #         if i == 0:
    #             start_hand_time = l[0]
    #             max_x = l[2]
    #             min_x = l[2]
    #             max_y = l[3]
    #             min_y = l[3]
    #             id = float(l[1])
    #         else:
    #             if l[2] > max_x:
    #                 max_x = l[2]
    #             elif l[2] < min_x:
    #                 min_x = l[2]
    #
    #             if l[3] > max_y:
    #                 max_y = l[3]
    #             elif l[3] < min_y:
    #                 min_y = l[3]
    #
    #         if i >= 3 and i < L - 3:
    #             sp = abs(spikes[i - 3])
    #
    #         if sp > 0:
    #             spike_count += 1
    #
    #         if spike_count >= 2:
    #             spike_count = 0
    #             cycles += 1
    #             average_time += (l[0] - start_hand_time) / 1000.0
    #             count_time.append((l[0] - start_hand_time) / 1000.0)
    #             numerator = abs(max_x - min_x)
    #             y = abs(max_y - min_y)
    #             denominator = y
    #             if y < numerator:
    #                 denominator = numerator
    #                 numerator = y
    #             r = numerator / denominator
    #             print (r)
    #             ratio += r
    #             ratio_avg.append(r)
    #             start_hand_time = l[0]
    #             max_x = l[2]
    #             min_x = l[2]
    #
    #             max_y = l[3]
    #             min_y = l[3]
    #         if (float(l[1]) - id) > 0 or (i == int(L - 1)):
    #             print("print_summary")
    #             if cycles == 0:
    #                 cycles = 1
    #             if average_time < 0.00000001:
    #                 average_time = -1
    #             d = len(count_time)
    #             if d == 0:
    #                 d = 1
    #             data_keys.append("average_participant_freq_@<" + str(id) + "_Hz>")
    #             if (sum(count_time) / d) == 0:
    #                 data_left["average_participant_freq_@<" + str(id) + "_Hz>"] = -1000000000.000
    #             else:
    #                 data_left["average_participant_freq_@<" + str(id) + "_Hz>"] = 1.0 / (sum(count_time) / d)
    #             d = len(ratio_avg)
    #             if d == 0:
    #                 d = 1
    #             data_keys.append("average_participant_ratio_@<" + str(id) + "_Hz>")
    #             data_left["average_participant_ratio_@<" + str(id) + "_Hz>"] = sum(ratio_avg)/d
    #
    #             id = l[1]
    #             cycles = 0
    #             ratio_avg = []
    #             count_time = []
    #
    #     return Carrier(data_keys, data_left)



if __name__ == '__main__':
    m = MyTouchApp()
    m.run()
    #MyPaintApp().run()