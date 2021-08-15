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
from utils.ellipse_equation_solver import EllipseEquationSolver


class Markup:

    def __init__(self):
        self.src_image = None
        self.transform = None
        self.inv_transform = None
        self.image = None
        self.visualization = None


class guiApp(QtWidgets.QMainWindow, Ui_Dialog):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButtonOpenImage.clicked.connect(self.open_image_clicked)
        self.labelImage.mousePressEvent = self.label_image_mouse_pressed

        self.markup = Markup()
        self.loaded_img = None
        self.scaled_img = None
        self.loaded_img_scale = None
        self.loaded_img_offset = None
        self.selected_points = []

    def set_source_image(self, cv_image):
        self.markup.src_image = cv_image.copy()
        img_ysz, img_xsz = cv_image.shape[:2]
        widget_xsz, widget_ysz = self.labelImage.width(), self.labelImage.height()
        self.markup.image = np.zeros(shape=(widget_ysz, widget_xsz, 3), dtype=np.uint8)
        k_x, k_y = widget_xsz / img_xsz, widget_xsz / img_ysz
        offset_x, offset_y = 0, 0
        if k_x < k_y:
            scale = k_x
            scaled_img_xsz, scaled_img_ysz = widget_xsz, int(scale * img_ysz)
            offset_y = int(0.5 * (widget_ysz - scaled_img_ysz))
        else:
            scale = k_y
            scaled_img_xsz, scaled_img_ysz = int(scale * img_xsz), widget_ysz
            offset_x = int(0.5 * (widget_xsz - scaled_img_xsz))
        self.markup.image[offset_y:offset_y + scaled_img_ysz, offset_x:offset_x + scaled_img_xsz, :] = \
            cv2.resize(cv_image, (scaled_img_xsz, scaled_img_ysz))
        self.markup.transform = np.asarray([
            [scale, 0, offset_x],
            [0, scale, offset_y]
        ], dtype=np.float32)
        self.markup.inv_transform = np.asarray([
            [1. / scale, 0, -1 * offset_x],
            [0, 1. / scale, -1 * offset_y]
        ], dtype=np.float32)

    def update_image(self, points=[]):
        self.markup.visualization = self.markup.image.copy()
        scale = self.markup.transform[0, 0]
        if len(points) >= 5:
            solver = EllipseEquationSolver()
            equation = solver.solve(points)
            a, b, x, y, teta = solver.equation2canonical(equation)
            scale = self.markup.transform[0, 0]
            center_coordinates = (int(x * scale), int(y * scale))
            axesLength = (int(a * scale), int(b * scale))
            angle = teta * 180. / np.pi
            startAngle = 0
            endAngle = 360
            color = (0, 0, 255)
            thickness = 5
            self.markup.visualization = cv2.ellipse(self.markup.visualization, center_coordinates, axesLength, angle, startAngle, endAngle,
                              color, thickness)
        for x, y in points:
            self.markup.visualization = cv2.circle(self.markup.visualization, (int(x * scale), int(y * scale)), 5, (0, 255, 255), 3)
        rgb_image = cv2.cvtColor(self.markup.visualization, cv2.COLOR_BGR2RGB)
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
        self.set_source_image(cv2.imread(path))
        self.update_image()

    def label_image_mouse_pressed(self, event: QtGui.QMouseEvent):
        widget_coord = np.asarray([[event.x(), event.y()]])
        x, y = np.dot(widget_coord, self.markup.inv_transform)[0][:2]
        self.selected_points.append((x, y))
        if len(self.selected_points) > 5:
            self.selected_points.pop(0)
        self.update_image(self.selected_points)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = guiApp()
    window.setWindowTitle("Pupil markup tool")
    window.show()
    sys.exit(app.exec_())
