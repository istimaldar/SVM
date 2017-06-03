import unittest
from itertools import filterfalse

import simplex


def genarate_A(n):
    length = 2 * n
    result = []
    for i in range(length - 1):
        equation = []
        for j in range(length):
            if i == 0:
                if j < n:
                    equation.append(1)
                else:
                    equation.append(0)
            else:
                if j == i - 1:
                    equation.append(-1)
                elif j == i + n - 1:
                    equation.append(1)
                else:
                    equation.append(0)
        result.append(equation)
    return result


def generate_b(n, C):
    result = []
    for i in range(2 * n - 1):
        if i == 0:
            result.append([0])
        else:
            result.append([C])
    return result


def generate_p(n):
    return [1 for element in range(n)]


def generate_w_minimize_system(A, b, C, p):
    n = len(A)
    n1 = len(C)
    result = []
    for i, equation in enumerate(A):
        w = [0 for element in range(n)]
        z1 = [0 for element in range(n1)]
        z2 = [0 for element in range(n1)]
        w[i] = 1
        equation = equation + w + z1 + z2 + b[i]
        result.append(equation)
    for i, equation in enumerate(C):
        equation = [2 * element for element in equation]
        w = [0 for element in range(n)]
        z1 = [0 for element in range(n1)]
        z2 = [0 for element in range(n1)]
        z1[i] = 1
        z2[i] = -1
        equation = equation + [0 for element in range(n - 1)] + w + z1 + z2 + [(-p[i])]
        result.append(equation)
    return result


def find_basis(conditions, n, n1):
    basis = []
    for i in range(2 * n - 1):
        basis.append(len(conditions[i]) - (2 * n1 + 2 * n - i))
    for i in range(n1):
        value = len(conditions[2 * n - 1 + i]) - (2 * n1 - i + 1) if conditions[2 * n - 1 + i][-(2 * n1 - i)] > 0 else \
            len(conditions[2 * n - 1 + i]) - (n1 - i + 1)
        basis.append(value)
    return basis


def generate_z_minimization_function(C, n, C_matrix, basis):
    A = genarate_A(n)
    n1 = len(C_matrix)
    conditions = generate_w_minimize_system(A, generate_b(n, C), C_matrix, generate_p(2))
    for i, condition in enumerate(conditions):
        conditions[i] = condition[:2 * n] + condition[4 * n - 1:]
    basis2 = list(filterfalse(lambda x: x < 2 * n, basis))
    basis2.append(len(conditions[0]) - 1)
    basis2 = [i for i in range(2 * n, len(conditions[0])) if i not in basis2]
    conditions = [list(column) for i, column in enumerate(zip(*conditions)) if i not in basis2]
    conditions = [list(column) for column in zip(*conditions)]


def minimize_z(C, n, C_matrix, basis):
    generate_z_minimization_function(C, n, C_matrix, basis)


def minimize_w(C, n, C_matrix):
    A = genarate_A(n)
    n1 = len(C_matrix)
    conditions = generate_w_minimize_system(A, generate_b(n, C), C_matrix, generate_p(2))
    function = [1 if n + len(A) - 1 <= i < n + 2 * len(A) - 1 else 0 for i in range(len(conditions[0]) - 1)]
    return simplex.simplex_method(function, conditions, find_basis(conditions, n, n1))


def wolf_method(C, n, C_matrix):
    first_phase, result, basis = minimize_w(C, n, C_matrix)
    first_phase = first_phase[:2 * n] + first_phase[4 * n - 1:]
    for i, element in enumerate(basis):
        if element > 2 * n:
            basis[i] = element - 2 * n + 1
    first_phase = first_phase[2 * n:] + list(filterfalse(lambda x: x == 0, first_phase[:2 * n]))
    minimize_z(C, n, C_matrix, basis)


class WolfTest(unittest.TestCase):
    def test_A(self):
        self.assertEqual(genarate_A(2), [[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1]])

    def test_b(self):
        self.assertEqual(generate_b(2, 3), [[0], [3], [3]])

    def test_equations(self):
        self.assertEqual(generate_w_minimize_system(genarate_A(2), generate_b(2, 2), [[9, -1], [-1, 9]], generate_p(2)),
                         [[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
                          [0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2], [18, -2, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1],
                          [-2, 18, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1]])

    def test_minimize_w(self):
        self.assertEqual(minimize_w(2, 2, [[9, -1], [-1, 9]])[0], [0.0, 0, 2.0, 2.0, 0, 0, 0, 0, 0, 1.0, 1.0])


if __name__ == "__main__":
    # unittest.main()
    wolf_method(2, 2, [[-4.5, 0.5], [0.5, -4.5]])
