# This Python file uses the following encoding: utf-8
import sys

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QSlider
from equalizer_bar import EqualizerBar
import random

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
#from ui_form import Ui_Widget

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.ui = Ui_Widget()
        #self.ui.setupUi(self)
        uic.loadUi('form.ui', self) # Load the .ui file
        self.old_equalizer = EqualizerBar(80, ['#0C0786', '#40039C', '#6A00A7', '#8F0DA3', '#B02A8F', '#CA4678', '#E06461',
                                          '#F1824C', '#FCA635', '#FCCC25', '#EFF821'])
        self.new_equalizer = EqualizerBar(80, ['#0C0786', '#40039C', '#6A00A7', '#8F0DA3', '#B02A8F', '#CA4678', '#E06461',
                                        '#F1824C', '#FCA635', '#FCCC25', '#EFF821'])
        self.horizontalLayout_2.addWidget(self.old_equalizer)
        self.horizontalLayout_2.addWidget(self.new_equalizer)

        self.sliders = []
        i = 0
        while(i < self.horizontalLayout.count()):
            self.sliders.append(self.findChild(QSlider, f"verticalSlider_{i}"))
            self.horizontalLayout.itemAt(i).widget().setValue(50)
            i +=1
        print(self.sliders)

        self._timer = QtCore.QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.update_values)
        self._timer.start()

    def update_values(self):
        self.old_equalizer.setValues([
            min(100, v+random.randint(0, 50) if random.randint(0, 5) > 2 else v)
            for v in self.old_equalizer.values()
            ])
        self.new_equalizer.setValues([
            min(100, v+random.randint(0, 50) if random.randint(0, 5) > 2 else v)
            for v in self.new_equalizer.values()
            ])



if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.showMaximized()
    sys.exit(app.exec_())
