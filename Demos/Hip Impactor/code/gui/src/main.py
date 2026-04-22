from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from ui.orthosense_screens import TargetScreen, InfoScreen, StartScreen
from kivy.uix.screenmanager import ScreenManager
from comms.imu import IMUTools, Imuio, BaseSens, ImuFromServer


class ExosensApp(App):
    def __init__(self, w=1280, h=720, imucup=None):
        super(ExosensApp, self).__init__()
        self.parent = Widget()
        Window.size = [w, h]
        self.sm = ScreenManager()
        self.my_screen = TargetScreen(myparent=self.sm, imucup=imucup)
        self.my_screen_setting = InfoScreen(myparent=self.sm, targetpage=self.my_screen)
        self.my_screen_start = StartScreen(myparent=self.sm, targetpage=self.my_screen)
        self.my_screen.setting_screen = self.my_screen_setting
        self.sm.add_widget(self.my_screen_start)
        self.sm.add_widget(self.my_screen)
        self.sm.add_widget(self.my_screen_setting)

        self.but = Button(text='Switch', font_size=14)

    def win_cb(self, window, width, height):
        self.my_screen.canvas.clear()
        self.my_screen.clear_widgets()
        self.my_screen.update(Window.size)

    def switch(self, e):
        self.sm.current = self.my_screen.name

    def closeme(self):
        self.my_screen.disconnect_imu()

    def build(self):
        Window.clearcolor = (40 / 255, 40 / 255, 40 / 255, 1)
        Window.bind(on_resize=self.win_cb)
        Window.bind(on_close=self.closeme)
        self.but.bind(on_release=self.switch)
        self.my_screen.add_labels()
        self.my_screen.connect_imu()
        # self.my_screen.conect2server()
        return self.sm


if __name__ == '__main__':
    list_of_imu = IMUTools.search()
    imu = []
    for i in list_of_imu:
        imu.append(BaseSens(i[0]))
    imu_d = None
    if len(imu) > 0:
        imu_d = imu[0]
    m = ExosensApp(imucup=imu_d)
    # m = ExosensApp(imucup=ImuFromServer())
    m.run()
    m.closeme()
