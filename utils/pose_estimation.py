import numpy as np


def find_normal(ellipse_canonical):
    assert len(ellipse_canonical) == 5
    a_axis, b_axis, x, y, teta = ellipse_canonical
    # x right, y up, z us
    if a_axis > b_axis:
        major_vector = np.asarray([
            a_axis * np.cos(teta), a_axis * np.sin(teta), 0.
        ])
        minor_axis_x = b_axis * np.sin(teta)
        minor_axis_y = b_axis * np.cos(teta)
        minor_axis_z = np.sqrt(a_axis ** 2 - minor_axis_x ** 2 - minor_axis_y ** 2)
        minor_vector = np.asarray([
            minor_axis_x, minor_axis_y, minor_axis_z
        ])
    else:
        major_vector = np.asarray([
            -1. * b_axis * np.sin(teta), b_axis * np.cos(teta), 0.
        ])
        minor_axis_x = a_axis * np.cos(teta)
        minor_axis_y = b_axis * np.sin(teta)
        minor_axis_z = np.sqrt(b_axis ** 2 - minor_axis_x ** 2 - minor_axis_y ** 2)
        minor_vector = np.asarray([
            minor_axis_x, minor_axis_y, minor_axis_z
        ])
    circle_normal = np.cross(major_vector, minor_vector)
    normal = circle_normal / np.linalg.norm(circle_normal)
    normal *= np.sign(normal[2])
    return normal


def normal2angles(normal):
    assert isinstance(normal, np.ndarray)
    assert len(normal) == 3
    x, y, z = normal
    yaw = np.abs(np.arcsin(x))
    pitch = np.abs(np.arcsin(y))
    return 180. * yaw / np.pi, 180. * pitch / np.pi
