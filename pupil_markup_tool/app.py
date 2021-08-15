import cv2
import numpy as np
import os, sys

ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")

from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from pupil_markup import Ui_Dialog


class guiApp(QtWidgets.QMainWindow, Ui_Dialog):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButtonOpenImage.clicked.connect(self.open_image_clicked)

        self.loaded_img = None

    def draw_image(self, cv_image):
        if not isinstance(cv_image, np.ndarray):
            return
        img_ysz, img_xsz = cv_image.shape[:2]
        widget_xsz, widget_ysz = self.labelImage.width(), self.labelImage.height()
        scaled_img = np.zeros(shape=(widget_ysz, widget_xsz, 3), dtype=np.uint8)
        k_x, k_y = widget_xsz / img_xsz, widget_xsz / img_ysz
        if k_x < k_y:
            scaled_img_ysz = int(k_x * img_ysz)
            img = cv2.resize(cv_image, (widget_xsz, scaled_img_ysz))
            offset = int(0.5 * (widget_ysz - scaled_img_ysz))
            scaled_img[offset:offset + scaled_img_ysz, :, :] = img
        else:
            scaled_img_xsz = int(k_y * img_xsz)
            img = cv2.resize(cv_image, (scaled_img_xsz, widget_ysz))
            offset = int(0.5 * (widget_xsz - scaled_img_xsz))
            scaled_img[:, offset:offset + scaled_img_xsz, :] = img
        rgb_image = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        qimg = QtGui.QImage(rgb_image.data, width, height, channels * width, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(qimg))
        self.labelImage.setPixmap(pixmap.scaled(self.labelImage.size(), QtCore.Qt.KeepAspectRatio))
        self.labelImage.setAlignment(QtCore.Qt.AlignCenter)
        self.labelImage.setScaledContents(True)
        self.labelImage.setMinimumSize(1, 1)
        self.labelImage.show()

    def open_image_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image to markup", directory="/home/igor",
                                                        filter="Images (*.png), *.bmp, *.jpg",
                                                        options=QtWidgets.QFileDialog.DontUseNativeDialog)
        self.labelSourceImagePath.setText(path)
        self.loaded_img = cv2.imread(path)
        self.draw_image(self.loaded_img)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = guiApp()
    window.setWindowTitle("Pupil markup tool")
    window.show()
    sys.exit(app.exec_())
