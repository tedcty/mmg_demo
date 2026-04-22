from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from commons.lang import CommonSymbols
from kivy.uix.spinner import Spinner


class BoxsensInfo(FloatLayout):
    def __init__(self, name: str = "Boxsen", pos=(0, 0), console=None):
        super(BoxsensInfo, self).__init__()
        self.console = console
        self.name = name
        self.pos = pos
        self.left_offset = 0
        self.left_offset_radio = 0
        self.top_offset = 0
        self.top_offset_title = 0
        self.top_offset_refplane = 0
        self.connected = False

        self.name = name
        self.IMU1label = Label(text=self.name, font_size='40sp',
                               pos=(pos[0], pos[1]),
                               halign='left')

        self.CaliIMU1 = Button(text="Calibrate",
                               size_hint=(None, None),
                               size=(120, 50),
                               pos=(pos[0] + 300, pos[1]+25))
        self.CaliIMU1.font_size = '18sp'
        self.CaliIMU1.bind(on_release=self.calibration_boxsens01)

        self.IMU1_batterylabel = Label(text=CommonSymbols.battery.value[0], font_size='40sp',
                                       pos=(pos[0] + 400, pos[1]),
                                       halign='left')
        self.IMU1_batterylabel.font_name = CommonSymbols.battery.value[2]
        self.IMU1_batterylabel.color = (0.5, 0.5, 0.5, 1)

        self.ConnectIMU1 = Button(text="Connect",
                                  size_hint=(None, None),
                                  size=(120, 50),
                                  pos=(pos[0] + 175, pos[1]+25))
        self.ConnectIMU1.font_size = '18sp'
        self.ConnectIMU1.bind(on_release=self.connecting_boxsens01)

        # self.RoleIMU_1 = Label(text='Role', font_size='25sp',
        #                        pos=(pos[0] - 80, pos[1]-75),
        #                        halign='left')
        #
        # self.spinner_imu1 = Spinner(
        #     # default value shown
        #     text='None',
        #     # available values
        #     values=('None', 'Cup', 'Reference', "Other"),
        #     # just for positioning in our example
        #     size_hint=(None, None),
        #     size=(100, 44),
        #     pos=(pos[0] - 55, pos[1]-120))
        # self.spinner_imu1.bind(text=self.setting_role)

        self.IMU1_connection = Label(text='Connection', font_size='25sp',
                                     pos=(pos[0] + 140, pos[1]-75),
                                     halign='left')
        self.IMU_1_connection_status = CommonSymbols.bars.value[1]
        self.IMU_1_indicator = Label(text=self.IMU_1_connection_status, font_size='55',
                                     pos=(pos[0] + 140, pos[1] - 141),
                                     halign='center')

        self.IMU_1_indicator.font_name = CommonSymbols.camera.value[2]
        self.IMU_1_indicator.color = (1, 0, 0, 1)

        self.IMU1_Status = Label(text='Status', font_size='25sp',
                                 pos=(pos[0] + 340, pos[1]-75),
                                 halign='left')

        imu1stat = CommonSymbols.flag.value[0]
        self.IMU1_Status_icon = Label(text=imu1stat, font_size='25sp',
                                      pos=(pos[0] + 400, pos[1]-75),
                                      halign='left')
        self.IMU1_Status_icon.font_name = CommonSymbols.flag.value[2]
        self.IMU1_Status_icon.color = (0, 1, 0, 1)

        self.IMU1_status_message = Label(text="Not Connected", font_size='18sp',
                                         pos=(pos[0] + 350, pos[1]-150),
                                         halign='center')
        self.add_labels()

    def update(self):
        if self.connected:
            self.IMU_1_indicator.color = (0, 1, 0, 1)
            self.IMU1_status_message.text = "Connected"

    def add_labels(self):
        self.add_widget(self.IMU1_Status_icon)
        self.add_widget(self.IMU1label)
        # self.add_widget(self.RoleIMU_1)
        self.add_widget(self.IMU1_connection)
        # self.add_widget(self.spinner_imu1)
        self.add_widget(self.IMU_1_indicator)
        self.add_widget(self.IMU1_Status)
        self.add_widget(self.ConnectIMU1)
        self.add_widget(self.CaliIMU1)
        self.add_widget(self.IMU1_batterylabel)
        self.add_widget(self.IMU1_status_message)

    def connecting_boxsens01(self, e):
        self.console.text = self.console.text + " Connecting " + self.name + " ... Cannot connect to " + self.name + "!\n>>"

    def calibration_boxsens01(self, e):
        self.console.text = self.console.text + " Calibrating " + self.name + " ... Cannot Calibrate to " + self.name + "!\n>>"

    def setting_role(self, spinner, text):
        self.console.text = self.console.text + " Setting Role " + self.name + " to "+text+"!\n>>"



