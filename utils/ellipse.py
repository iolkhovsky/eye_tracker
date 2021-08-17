import numpy as np


ELLIPSE_EQUATION_COEFF_CNT = 6


def solve_ellipse_equation(points):
    """Finds  coefficients of the general 2d ellipse equation using the set of 6 input points:
    Ax2 + Bxy + Cy2 + Dx + Ey + F = 0
    returns: (A, B, C, D, E, F)"""
    assert len(points) == ELLIPSE_EQUATION_COEFF_CNT
    mat_a = np.zeros(shape=(ELLIPSE_EQUATION_COEFF_CNT, ELLIPSE_EQUATION_COEFF_CNT), dtype=np.float64)
    for idx, p in enumerate(points):
        assert len(p) == 2
        x, y = p
        mat_a[idx] = np.asarray([x ** 2, x * y, y ** 2, x, y, 1], dtype=np.float64)
    try:
        u, s, v_transpose = np.linalg.svd(mat_a)
        vector_id = np.argmin(s)
        sorted_s = np.sort(s)
        assert sorted_s[0] * 100 < sorted_s[1] and abs(sorted_s[0] - sorted_s[1]) > 1e-9
        return v_transpose[vector_id]
    except np.linalg.LinAlgError:
        print("solve_ellipse_equation: Error during SVD computation")
        return None
    except AssertionError:
        print("solve_ellipse_equation: sigma matrix has more than one zero diagonal element")
        return None


def ellipse_equation_to_canonical(equation):
    if equation is None:
        return None
    assert len(equation) == ELLIPSE_EQUATION_COEFF_CNT
    a, b, c, d, e, f = equation
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
    a_axis = num / denum
    num = -1 * (
            2 * (a * e ** 2 + c * d ** 2 - b * d * e + denum * f) *
            ((a + c) - ((a - c) ** 2 + b ** 2) ** 0.5)
    ) ** 0.5
    b_axis = num / denum
    return a_axis, b_axis, x, y, teta
