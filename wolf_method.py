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


def generate_w_minimize_system(A, b, C):
    n = len(A)
    for i, equation in enumerate(A):
        w = [0 in range(n)]
        z1 = [0 in range(n)]
        z2 = [0 in range(n)]
        w[i] = 1
        equation = equation + w + z1 + z2


class WolfTest(unittest.TestCase):
    def test_A(self):
        self.assertEqual(genarate_A(2), [[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1]])

    def test_b(self):
        self.assertEqual(generate_b(2, 3), [[0], [3], [3]])


if __name__ == "__main__":
    unittest.main()