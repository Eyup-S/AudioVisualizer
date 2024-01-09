from PyQt6 import QtWidgets, uic, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QListWidget, QListWidgetItem, QPushButton, QLineEdit, QLabel, QRadioButton
from PyQt6.QtGui import QPixmap, QImage
from equalizer_bar import EqualizerBar

import sys
import time
import math
import os
import numpy as np

from Tools import Tools
from NoiseReduction import NoiseReduction
import constants


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('form.ui', self) # Load the .ui file
        self.setup_equalizers()
        self.setup_sliders()
        self.setup_songs_list()
        self.setup_plots()
        self.setup_preset_buttons()
        self.tools = Tools()
        self.setup_noise_reduction()
        self.play_button = self.findChild(QPushButton, f"pushButton")
        self.play_button.pressed.connect(self.play_button_pressed)
        self.tools.paused = True

        self.current_second = 0
        self.current_song = None
        self.setup_timer()
    
    def setup_timer(self):
        self._timer = QtCore.QTimer()
        self._timer.setInterval(int(constants.timestep * 1000))
        self._timer.timeout.connect(self.play_song)
        self._timer.start()

    def play_song(self):
        if (not self.tools.paused):
            if (self.current_second < float(self.tools.duration)):
                self.play_song_part(self.current_second)
                self.current_second += constants.timestep
                print(self.current_second)

    def play_song_part(self, current_second):
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
                fft_mag_total += abs(fft_magnitude[i]) 
                fft_freq_num += 1
                i += 1

            values.append(min(100, math.log(fft_mag_total / (fft_freq_num + 0.0001)) ** 2 * constants.magnitude_gain)) 
        self.old_equalizer.setValues(values)

    def update_new_equalizer(self, fft_magnitude, fft_frequencies):
        i = 0
        values = []
        for max_value in constants.equalizer_divisions:
            fft_mag_total = 0
            fft_freq_num = 0
            while i < len(fft_frequencies) and fft_frequencies[i] < max_value + 1:
                fft_mag_total += abs(fft_magnitude[i]) 
                fft_freq_num += 1
                i += 1
            values.append(min(100, math.log(fft_mag_total / (fft_freq_num + 0.0001) + 1) ** 2 * constants.magnitude_gain)) 
        self.new_equalizer.setValues(values)

    def setup_equalizers(self):
        self.old_equalizer = EqualizerBar(18, [
                                            '#0C0786', '#2B0593', '#47029E', '#6001A4', '#7705A6', 
                                            '#8D0CA3', '#A01C98', '#B32D8C', '#C23E7F', '#D04F71', 
                                            '#DD6064', '#E87257', '#F2844B', '#F8993D', '#FCAF31', 
                                            '#FCC528', '#F7DE23', '#EFF821'
                                        ])
        self.new_equalizer = EqualizerBar(18, [
                                            '#0C0786', '#2B0593', '#47029E', '#6001A4', '#7705A6', 
                                            '#8D0CA3', '#A01C98', '#B32D8C', '#C23E7F', '#D04F71', 
                                            '#DD6064', '#E87257', '#F2844B', '#F8993D', '#FCAF31', 
                                            '#FCC528', '#F7DE23', '#EFF821'
                                        ])
        self.horizontalLayout_2.addWidget(self.old_equalizer)
        self.horizontalLayout_2.addWidget(self.new_equalizer)

    def setup_sliders(self):
        self.sliders = []
        i = 0
        while(i < self.horizontalLayout.count()):
            self.sliders.append(self.tabWidget.findChild(QSlider, f"verticalSlider_{i}"))
            self.sliders[i].setValue(50)
            self.sliders[i].sliderReleased.connect(self.update_sliders)
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
            self.noise_sliders.append(self.findChild(QSlider, f"NverticalSlider_{i}"))
            self.horizontalLayout_6.itemAt(i).widget().setValue(0)
            self.noise_sliders[i].sliderReleased.connect(self.update_noise_sliders)
            i +=1

        self.noise_max_boxes = []
        i = 0
        while(i < self.horizontalLayout_7.count()):
            self.noise_max_boxes.append(self.findChild(QLineEdit, f"lineEdit_{5 + i}"))
            self.horizontalLayout_7.itemAt(i).widget().setText(str(constants.default_nr_freq[i][1]))
            self.horizontalLayout_7.itemAt(i).widget().setValidator(QtGui.QIntValidator())
            self.noise_max_boxes[i].editingFinished.connect(self.update_noise_sliders)
            i +=1

        self.noise_min_boxes = []
        i = 0
        while(i < self.horizontalLayout_5.count()):
            self.noise_min_boxes.append(self.findChild(QLineEdit, f"lineEdit_{i}"))
            self.horizontalLayout_5.itemAt(i).widget().setText(str(constants.default_nr_freq[i][0]))
            self.horizontalLayout_5.itemAt(i).widget().setValidator(QtGui.QIntValidator())
            self.noise_min_boxes[i].editingFinished.connect(self.update_noise_sliders)
            i +=1

        self.tools.nr_freq_bands = constants.default_nr_freq
        self.tools.nr_thresholds = [0] * len(constants.default_nr_freq)

    def setup_plots(self):
        i = 0
        self.plots = []
        while(i < 3):
            self.plots.append(self.findChild(QLabel, f"label_{i}"))
            i += 1
        self.play_button = self.findChild(QPushButton, f"plotButton")
        self.play_button.pressed.connect(self.plot_button_pressed)

    def setup_preset_buttons(self):
        self.preset_buttons = []
        i = 0
        while(i < self.horizontalLayout_3.count()):
            self.preset_buttons.append(self.findChild(QRadioButton, f"radioButton_{i}"))
            self.preset_buttons[i].pressed.connect(self.preset_button_pressed)
            i += 1

    def preset_button_pressed(self):
        i = 0
        while(i < len(self.sliders)):
            self.sliders[i].setValue(constants.presets[self.preset_buttons.index(self.sender())][i] / 12 * 50 + 50)
            i += 1
        self.update_sliders()

    def update_noise_sliders(self):
        i = 0
        while(i < len(self.noise_sliders)):
            self.tools.nr_thresholds[i] = self.noise_sliders[i].value()
            i += 1
        i = 0
        while(i < len(self.noise_max_boxes)):
            self.tools.nr_freq_bands[i] = (int(self.noise_min_boxes[i].text()), int(self.noise_max_boxes[i].text()))
            i += 1

    def update_sliders(self):
        i = 0
        while(i < len(self.sliders)):
            self.tools.gain_list[i] = (self.sliders[i].value() - 50) / 2
            i += 1
        self.preset_buttons[-1].setChecked(True)
    
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

    def plot_button_pressed(self):
        if (self.current_song is not None):
            fft_mag, fft_freq = self.tools.fourier_transform(self.tools.normalized_data)
            img_array, width, height = Tools.plot_fft(fft_mag, fft_freq, 0, 2000, self.tools.duration)
            q_img = QImage(img_array, width, height, QImage.Format.Format_RGB32)
            pixmap = QPixmap.fromImage(q_img)
            self.plots[0].setPixmap(pixmap)
            self.plots[0].setScaledContents(True)

            _, fft_mag = self.tools.equalizer(fft_mag, fft_freq, self.tools.band_list, self.tools.gain_list)
            fft_mag = self.tools.spectral_gate(fft_mag, fft_freq)
            img_array, width, height = Tools.plot_fft(fft_mag, fft_freq, 0, 2000, self.tools.duration)
            q_img = QImage(img_array, width, height, QImage.Format.Format_RGB32)
            pixmap = QPixmap.fromImage(q_img)
            self.plots[2].setPixmap(pixmap)
            self.plots[2].setScaledContents(True)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.showMaximized()
    sys.exit(app.exec())