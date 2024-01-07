from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QListWidget, QListWidgetItem, QPushButton, QLineEdit
from equalizer_bar import EqualizerBar

import sys
import time
import math
import os

from Tools import Tools
import constants


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('form.ui', self) # Load the .ui file
        self.setup_equalizers()
        self.setup_sliders()
        self.setup_songs_list()
        self.setup_noise_reduction()
        self.play_button = self.findChild(QPushButton, f"pushButton")
        self.play_button.pressed.connect(self.play_button_pressed)
        self.tools = Tools()

        self.tools.read_file("Je te laisserai des mots.wav")
        self.tools.gain_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_second = 0
        self.setup_timer()
    
    def setup_timer(self):
        self._timer = QtCore.QTimer()
        self._timer.setInterval(constants.timestep * 1000)
        self._timer.timeout.connect(self.play_song)
        self._timer.start()

    def play_song(self):
        if (not self.tools.paused):
            if (self.current_second < float(self.tools.duration)):
                self.play_song_part(self.current_second)
                self.current_second += constants.timestep
                print(self.current_second)

    def play_song_part(self, current_second):
        # divided_data = self.tools.data[int(current_second * self.tools.sample_rate): int((current_second + constants.timestep) * self.tools.sample_rate)]
        # fft_magnitude, fft_frequencies = self.tools.fourier_transform(divided_data)
        if (self.tools.input_freq is not None):
            self.update_old_equalizer(self.tools.input_mag, self.tools.input_freq)
            self.update_new_equalizer(self.tools.output_mag, self.tools.input_freq)

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

            values.append(min(100, math.log(abs(fft_mag_total / (fft_freq_num + 0.0001))) ** 2 * constants.magnitude_gain)) 
        self.old_equalizer.setValues(values)

    def update_new_equalizer(self, fft_magnitude, fft_frequencies):
        i = 0
        values = []
        for max_value in constants.equalizer_divisions:
            fft_mag_total = 0
            fft_freq_num = 0
            while i < len(fft_frequencies) and fft_frequencies[i] < max_value + 1:
                fft_mag_total += fft_magnitude[i] 
                fft_freq_num += 1
                i += 1

            values.append(min(100, math.log(abs(fft_mag_total / (fft_freq_num + 0.0001))) ** 2 * constants.magnitude_gain)) 
        self.new_equalizer.setValues(values)

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
            self.horizontalLayout.itemAt(i).widget().valueChanged.connect(self.update_sliders)
            i +=1

    def setup_songs_list(self):
        self.songs_list = self.findChild(QtWidgets.QListWidget, "listWidget")
        songs = os.listdir('./songs')
        for song in songs:
            if song.endswith(".wav"):
                self.songs_list.addItem(QListWidgetItem(song))
        self.songs_list.setCurrentItem(self.songs_list.item(0))
        self.current_song = self.songs_list.currentItem().text()

    def setup_noise_reduction(self):
        self.noise_sliders = []
        i = 0
        while(i < self.horizontalLayout_6.count()):
            self.sliders.append(self.findChild(QSlider, f"NverticalSlider_{i}"))
            self.horizontalLayout.itemAt(i).widget().setValue(50)
            self.horizontalLayout.itemAt(i).widget().valueChanged.connect(self.update_sliders)
            i +=1
        
        self.noise_max_boxes = []
        i = 0
        while(i < self.horizontalLayout_7.count()):
            self.noise_max_boxes.append(self.findChild(QLineEdit, f"lineEdit_{5 + i}"))
            self.horizontalLayout_7.itemAt(i).widget().setText("0")
            self.horizontalLayout_7.itemAt(i).widget().setValidator(QtGui.QIntValidator())
            i +=1

        self.noise_min_boxes = []
        i = 0
        while(i < self.horizontalLayout_5.count()):
            self.noise_min_boxes.append(self.findChild(QLineEdit, f"lineEdit_{i}"))
            self.horizontalLayout_5.itemAt(i).widget().setText("0")
            self.horizontalLayout_5.itemAt(i).widget().setValidator(QtGui.QIntValidator())
            i +=1

    def update_sliders(self):
        i = 0
        while(i < len(self.sliders)):
            self.tools.gain_list[i] = (self.sliders[i].value() - 50) / 2
            i += 1
    
    def play_button_pressed(self):
        if (self.current_song != self.songs_list.currentItem().text()):
            self.tools.stop()
            self.tools.read_file(self.songs_list.currentItem().text())
            self.current_song = self.songs_list.currentItem().text()
            self.current_second = 0
        if (self.tools.paused):
            self.tools.play(int(self.tools.sample_rate * constants.timestep))
        else:
            self.tools.paused = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.showMaximized()
    sys.exit(app.exec_())