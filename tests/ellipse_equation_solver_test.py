from matplotlib import pyplot as plt
import numpy as np
from os import makedirs
from os.path import isdir, join
import pytest

from utils.ellipse_equation_solver import EllipseEquationSolver


TEST_OUTPUT_DIR = "test_output"


def test_ellipse_equation_from_points():
    points = [
        [-2, 0],
        [2, 0],
        [0, 1],
        [0, -1],
        [1.5, 0.6]
    ]
    solver = EllipseEquationSolver()
    equation = solver.solve(points)
    assert equation is not None and len(equation) == 5
    target = (-0.25, -1.0, -0.0861111, 0.0, -0.0, 1.0)
    for val, target_val in zip(equation, target):
        assert target_val == pytest.approx(val, 1e-3)


def test_canonical_from_equation():
    points = [
        [-2, 0],
        [2, 0],
        [0, 1],
        [0, -1],
        [1.5, 0.6]
    ]
    solver = EllipseEquationSolver()
    equation = solver.solve(points)
    canonical = solver.equation2canonical(equation)
    assert canonical is not None and len(canonical) == 5
    target = (0.9987704640773339, 2.0083868868770423, 0.0, -0.0, 1.5136392076492708)
    for val, target_val in zip(canonical, target):
        assert target_val == pytest.approx(val, 1e-3)


def test_ellipse_solver():
    test_point_set = [
        [[-2, 0], [2, 0], [0, 1], [0, -1], [1.5, 0.6], [-1.5, -0.6]],
    ]
    solver = EllipseEquationSolver()
    if not isdir(TEST_OUTPUT_DIR):
        makedirs(TEST_OUTPUT_DIR)

    for idx, points in enumerate(test_point_set):
        equation = solver.solve(points)
        a, b, x, y, teta = solver.equation2canonical(equation)
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
