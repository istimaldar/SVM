import unittest
import simplex
import wolf_method
import svm


class TestClass(unittest.TestCase):
    def test_maximize(self):
        self.assertEqual(simplex.simplex_method([9, 5, 4, 3, 2, 0], [[1, -2, 2, 0, 0, 1, 6], [1, 2, 1, 1, 0, 0, 24],
                                                            [2, 1, -4, 0, 1, 0, 30]], [], False)[0],
                         [0, 7.0, 10.0, 0, 63.0, 0])

    def test_minimize(self):
        self.assertEqual(simplex.simplex_method([1, 9, 5, 3, 4, 14], [[1, 0, 0, 1, 0, 0, 20], [0, 1, 0, 0, 1, 0, 50],
                                                   [0, 0, 1, 0, 0, 1, 30], [0, 0, 0, 1, 1, 1, 60]],
                             [1, 3, 4, 5])[0], [10, 0, 30, 10, 50, 0])

    def test_A_svm(self):
        machine = svm.SVM(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 1)
        self.assertEqual(machine.classify([-1, -1]), -1)
        self.assertEqual(machine.classify([-1, 1]), 1)
        self.assertEqual(machine.classify([1, -1]), 1)
        self.assertEqual(machine.classify([1, 1]), -1)

    def test_b_svm(self):
        machine = svm.SVM(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 1)
        self.assertEqual(machine.get_C(), [[9, -1, -1, 1], [-1, 9, 1, -1], [-1, 1, 9, -1], [1, -1, -1, 9]])

    def test_simplest_b(self):
        machine = svm.SVM(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [1, 1]], [-1, 1], 1)
        self.assertEqual(machine.get_C(), [[9, -1], [-1, 9]])

    def test_A(self):
        self.assertEqual(wolf_method.genarate_A(2), [[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1]])

    def test_b(self):
        self.assertEqual(wolf_method.generate_b(2, 3), [[0], [3], [3]])

    def test_equations(self):
        self.assertEqual(wolf_method.generate_w_minimize_system(wolf_method.genarate_A(2), wolf_method.generate_b(2, 2),
                                                                [[9, -1], [-1, 9]], wolf_method.generate_p(2)),
                         [[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
                          [0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2], [18, -2, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1],
                          [-2, 18, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1]])


if __name__ == "__main__":
    unittest.main()