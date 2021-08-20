import numpy as np


def find_normal(ellipse_canonical):
    assert len(ellipse_canonical) == 5
    a_axis, b_axis, x, y, teta = ellipse_canonical
    # x right, y up, z us
    if a_axis > b_axis:
        minor_axis_angle = np.arccos(b_axis / a_axis)
    else:
        minor_axis_angle = np.arccos(a_axis / b_axis)
    major_vector = np.asarray([
        a_axis * np.cos(teta), a_axis * np.sin(teta), 0.
    ])
    minor_vector = np.asarray([
        b_axis * np.sin(teta), b_axis * np.cos(teta), a_axis * np.sin(minor_axis_angle)
    ])
    circle_normal = np.cross(major_vector, minor_vector)
    if circle_normal[2] < 0:
        circle_normal *= -1.
    norm = circle_normal / np.linalg.norm(circle_normal)
    return norm


def normal2angles(normal):
    assert isinstance(normal, np.ndarray)
    assert len(normal) == 3
    x, y, z = normal
    yaw = np.abs(np.arcsin(x))
    pitch = np.abs(np.arcsin(y))
    return 180. * yaw / np.pi, 180. * pitch / np.pi


canonical = np.asarray([35.40, 30.53, 12.05, 13.07, 54.09 * np.pi / 180.])
normal = find_normal(canonical)
print(f"Normal: {normal}")
x_angle, y_angle = normal2angles(normal)
print((f"Angles: {x_angle}, {y_angle}"))
