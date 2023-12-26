# This Python file uses the following encoding: utf-8
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QSlider
from equalizer_bar import EqualizerBar

import sys
import random
import time

from Tools import Tools
import constants

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
#from ui_form import Ui_Widget

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('form.ui', self) # Load the .ui file
        self.tools = Tools()
        self.setup_equalizers()
        self.setup_sliders()

        self.tools.read_file(r"D:\\Emir\\School\\Semester 6\\EE 473\\project\\audio\\Je te laisserai des mots.wav")
        self.current_second = 0
        self.setup_timer()
    
    def setup_timer(self):
        self._timer = QtCore.QTimer()
        self._timer.setInterval(int(100))
        self._timer.timeout.connect(self.play_song)
        self._timer.start()

    def play_song(self):
        if (self.current_second < float(self.tools.duration)):
            self.play_song_part(self.current_second)
            self.current_second += constants.timestep
            print(self.current_second)

    def play_song_part(self, current_second):
        fft_magnitude, fft_frequencies = self.tools.fourier_transform_range(int(current_second * self.tools.getSampleRate()),  int((current_second + constants.timestep) * self.tools.getSampleRate()))
        self.update_old_equalizer(fft_magnitude, fft_frequencies)
        # time.sleep(constants.timestep)

    def update_old_equalizer(self, fft_magnitude, fft_frequencies):
        i = 0
        values = []
        for max_value in constants.equalizer_divisions:
            fft_mag_total = 0
            fft_freq_num = 0
            while i < len(fft_frequencies) and fft_frequencies[i] < max_value + 1:
                fft_mag_total += fft_magnitude[i] 
                fft_freq_num += 1
                i += 1

            values.append(min(100, (fft_mag_total / (fft_freq_num + 0.0001)) * constants.magnitude_gain)) 
        self.old_equalizer.setValues(values)


    def setup_equalizers(self):
        self.old_equalizer = EqualizerBar(12, ['#0C0786', '#40039C', '#6A00A7', '#8F0DA3', '#B02A8F', '#CA4678', '#E06461',
                                          '#F1824C', '#FCA635', '#FCCC25', '#EFF821'])
        self.new_equalizer = EqualizerBar(12, ['#0C0786', '#40039C', '#6A00A7', '#8F0DA3', '#B02A8F', '#CA4678', '#E06461',
                                        '#F1824C', '#FCA635', '#FCCC25', '#EFF821'])
        self.horizontalLayout_2.addWidget(self.old_equalizer)
        self.horizontalLayout_2.addWidget(self.new_equalizer)

    def setup_sliders(self):
        self.sliders = []
        i = 0
        while(i < self.horizontalLayout.count()):
            self.sliders.append(self.findChild(QSlider, f"verticalSlider_{i}"))
            self.horizontalLayout.itemAt(i).widget().setValue(50)
            i +=1


    # def update_values_random(self):
    #     self.old_equalizer.setValues([
    #         min(100, v+random.randint(0, 50) if random.randint(0, 5) > 2 else v)
    #         for v in self.old_equalizer.values()
    #         ])
    #     self.new_equalizer.setValues([
    #         min(100, v+random.randint(0, 50) if random.randint(2, 5) > 2 else v)
    #         for v in self.new_equalizer.values()
    #         ])



if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.showMaximized()
    sys.exit(app.exec_())
