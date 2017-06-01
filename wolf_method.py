import unittest
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


def minimize_w(C, n, C_matrix):
    A = genarate_A(n)
    conditions = generate_w_minimize_system(A, generate_b(n, C), C_matrix, generate_p(2))
    function = [1 if n + len(A) - 1 <= i < n +  2 * len(A) - 1 else 0 for i in range(len(conditions[0]) - 1)]
    print(function)
    print(conditions)
    return simplex.solve_simplex_custom_basis(function, conditions)


class WolfTest(unittest.TestCase):
    def test_A(self):
        self.assertEqual(genarate_A(2), [[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1]])

    def test_b(self):
        self.assertEqual(generate_b(2, 3), [[0], [3], [3]])

    def test_equations(self):
        self.assertEqual(generate_w_minimize_system(genarate_A(2), generate_b(2, 2), [[9, -1], [-1, 9]], generate_p(2)),
                         [[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 2],
                          [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 2],
                          [0, 0, 0, 1, 0, -1, 0, 18, -2, 0, 0, -1],
                          [0, 0, 0, 0, 1, 0, -1, -2, 18, 0, 0, -1]])


if __name__ == "__main__":
    # unittest.main()
    print(minimize_w(2, 2, [[9, -1], [-1, 9]]))