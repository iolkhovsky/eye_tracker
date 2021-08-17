from matplotlib import pyplot as plt
import numpy as np
from os import makedirs
from os.path import isdir, join
import pytest

from utils.ellipse import solve_ellipse_equation, ellipse_equation_to_canonical


TEST_OUTPUT_DIR = "test_output"


def test_ellipse_equation_from_points():
    points = [
        [-2, 0],
        [2, 0],
        [0, 1],
        [0, -1],
        [1.5, 0.6],
        [-1.5, -0.6]
    ]
    equation = solve_ellipse_equation(points)
    print(equation)
    assert equation is not None and len(equation) == 6
    target = [1.73765574e-01, 5.98525867e-02, 6.95062297e-01, 1.14690723e-16, -6.27463386e-16, -6.9506e-1]
    for val, target_val in zip(equation, target):
        assert target_val == pytest.approx(val, 1e-2)


def test_canonical_from_equation():
    equation = [1/16., 0., 1/4., 0, 0, -1]
    canonical = ellipse_equation_to_canonical(equation)
    assert canonical is not None and len(canonical) == 5
    target = (4, 2, 0., 0., 0.)
    for val, target_val in zip(canonical, target):
        assert target_val == pytest.approx(val, 1e-3)


def test_ellipse_solver():
    test_point_set = [
        [[-2, 0], [2, 0], [0, 1], [0, -1], [1.5, 0.6], [-1.5, -0.6]],
    ]
    if not isdir(TEST_OUTPUT_DIR):
        makedirs(TEST_OUTPUT_DIR)

    for idx, points in enumerate(test_point_set):
        equation = solve_ellipse_equation(points)
        a, b, x, y, teta = ellipse_equation_to_canonical(equation)
        points_cnt = 100
        t = np.linspace(0, 2 * np.pi, points_cnt)
        ellipse = np.array([a * np.cos(t), b * np.sin(t)])
        rotation_matrix = np.asarray([[np.cos(teta), -1 * np.sin(teta)], [np.sin(teta), np.cos(teta)]], dtype=np.float32)
        rotated_ellipse = np.zeros(ellipse.shape, dtype=np.float32)
        for i in range(points_cnt):
            rotated_ellipse[:, i] = np.dot(rotation_matrix, ellipse[:, i])
        rotated_ellipse[0, :] += x
        rotated_ellipse[1, :] += y
        plt.plot(rotated_ellipse[0, :], rotated_ellipse[1, :], 'blue')
        xref, yref = [list(x) for x in zip(*points)]
        plt.scatter(x=xref, y=yref, marker="^", color="red")
        plt.grid(color='gray', linestyle='--')
        plt.savefig(join(TEST_OUTPUT_DIR, f"test_{idx}_ellipse_solver.jpg"))
        plt.clf()
