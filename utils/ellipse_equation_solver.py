import numpy as np


class EllipseEquationSolver:

    """
    Finds  coefficients of the general ellipse equation using the set of input points:
    A'x2 + B'y2 + C'xy + D'x + E'y + F' = 0 -> (* 1./F')
    Ax2 + By2 + Cxy + Dx + Ey = -1
    in
    returns: (A, B, C, D, E)
    """

    @staticmethod
    def solve(points):
        points_cnt = len(points)
        assert points_cnt >= 5
        matA = np.zeros(shape=(points_cnt, 5), dtype=np.float32)
        matB = np.full(shape=(points_cnt, 1), fill_value=-1., dtype=np.float32)
        for idx, p in enumerate(points):
            assert len(p) == 2
            x, y = p
            matA[idx] = np.asarray([x ** 2, y ** 2, x * y, x, y], dtype=np.float32)
        try:
            return np.linalg.solve(matA, matB).flatten()
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def equation2canonical(equation_coeffs):
        assert len(equation_coeffs) == 5
        a, c, b, d, e = equation_coeffs
        f = 1.
        denum = b ** 2 - 4 * a * c
        x = (2 * c * d - b * e) / denum
        y = (2 * a * e - b * d) / denum
        teta = None
        if b == 0:
            teta = 0. if a < c else 0.5 * np.pi
        else:
            teta = np.arctan((1 / b) * (c - a - ((a - c) ** 2 + b ** 2) ** 0.5))

        num = -1 * (
                2 * (a * e ** 2 + c * d ** 2 - b * d * e + denum * f) *
                ((a + c) + ((a - c) ** 2 + b ** 2) ** 0.5)
        ) ** 0.5
        a = num / denum
        num = -1 * (
                2 * (a * e ** 2 + c * d ** 2 - b * d * e + denum * f) *
                ((a + c) - ((a - c) ** 2 + b ** 2) ** 0.5)
        ) ** 0.5
        b = num / denum
        return a, b, x, y, teta
