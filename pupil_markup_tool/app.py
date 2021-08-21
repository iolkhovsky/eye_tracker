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
from os.path import basename, dirname, isfile, join, splitext
import sys

from pupil_markup import Ui_Dialog
from utils.ellipse import solve_ellipse_equation, ellipse_equation_to_canonical, visualize_ellipse
from utils.file_utils import write_yaml, read_yaml
from utils.pose_estimation import find_normal, normal2angles


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
        self.pushButtonSaveMarkup.clicked.connect(self.save_markup_clicked)

        self.markup = Markup()
        self.selected_points = []
        self.points_max_cnt = 6
        self.equation = None
        self.canonical = None

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
        self.labelEllipseEquation.clear()
        self.label_semi_a.clear()
        self.label_semi_b.clear()
        self.label_center_x.clear()
        self.label_center_y.clear()
        self.label_teta.clear()
        self.label_yaw.clear()
        self.label_pitch.clear()
        if len(points) >= self.points_max_cnt:
            self.equation = solve_ellipse_equation(points)
            if self.equation is not None:
                a, b, c, d, e, f = self.equation
                self.labelEllipseEquation.setText(f"{'{:.3f}'.format(a)}x2 + {'{:.3f}'.format(b)}xy + "
                                                  f"{'{:.3f}'.format(c)}y2 + {'{:.3f}'.format(d)}x + "
                                                  f"{'{:.3f}'.format(e)}y + {'{:.3f}'.format(f)} = 0")
            self.canonical = ellipse_equation_to_canonical(self.equation)
            if self.canonical is not None:
                a, b, x, y, teta = self.canonical
                yaw, pitch = normal2angles(find_normal(self.canonical))
                self.label_center_x.setText("{:.2f}".format(x))
                self.label_center_y.setText("{:.2f}".format(y))
                self.label_semi_a.setText("{:.2f}".format(a))
                self.label_semi_b.setText("{:.2f}".format(b))
                self.label_teta.setText("{:.2f}".format(teta * 180. / np.pi))
                self.label_yaw.setText("{:.2f}".format(yaw))
                self.label_pitch.setText("{:.2f}".format(pitch))
                vis_img = visualize_ellipse(self.canonical, self.markup.visualization, self.markup.transform)
                if vis_img is not None:
                    self.markup.visualization = vis_img

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
        markup_path = f"{splitext(path)[0]}.yml"
        self.labelOutputMarkupPath.setText(markup_path)
        self.set_source_image(cv2.imread(path))
        if isfile(markup_path):
            data = read_yaml(markup_path)
            self.selected_points = data["markup"]["keypoints"]
            self.update_image(self.selected_points)
        else:
            self.update_image()

    def label_image_mouse_pressed(self, event: QtGui.QMouseEvent):
        widget_coord = np.asarray([[event.x(), event.y()]])
        x, y = np.dot(widget_coord, self.markup.inv_transform)[0][:2]
        self.selected_points.append((x, y))
        if len(self.selected_points) > self.points_max_cnt:
            self.selected_points.pop(0)
        self.update_image(self.selected_points)

    def save_markup_clicked(self):
        target_path = self.labelOutputMarkupPath.text()
        a, b, c, d, e, f = self.equation
        a_axis, b_axis, x, y, teta = self.canonical
        data = {
            "hint": "pupil_markup",
            "source_image_path": self.labelSourceImagePath.text(),
            "image_width": self.markup.src_image.shape[1],
            "image_height": self.markup.src_image.shape[0],
            "image_channels": self.markup.src_image.shape[2],
            "markup": {
                "keypoints": [(float(x), float(y)) for x, y in self.selected_points],
                "equation": {
                    "a": float(a), "b": float(b), "c": float(c), "d": float(d), "e": float(e), "f": float(f),
                },
                "canonical": {
                    "a": float(a_axis), "b": float(b_axis), "x": float(x), "y": float(y), "teta": float(teta)
                }
            }
        }
        write_yaml(target_path, data)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = guiApp()
    window.setWindowTitle("Pupil markup tool")
    window.show()
    sys.exit(app.exec_())
