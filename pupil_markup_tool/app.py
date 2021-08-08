from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie
import sys

import pupil_markup
from pupil_markup import Ui_Dialog


class guiApp(QtWidgets.QMainWindow, pupil_markup.Ui_Dialog):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButtonOpenImage.clicked.connect(self.open_image_clicked)

    def open_image_clicked(self):
        dialog = QtWidgets.QFileDialog(self, options=QtWidgets.QFileDialog.DontUseNativeDialog)
        dialog.setNameFilter("Images (*.png *.bmp *.jpg)")
        dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        path = dialog.getOpenFileName(self, "Open source image", "/home")[0]
        self.labelSourceImagePath.setText(path)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = guiApp()
    window.show()
    sys.exit(app.exec_())
