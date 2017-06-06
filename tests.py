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
        machine.build()
        self.assertEqual(machine.classify([-1, -1]), -1)
        self.assertEqual(machine.classify([-1, 1]), 1)
        self.assertEqual(machine.classify([1, -1]), 1)
        self.assertEqual(machine.classify([1, 1]), -1)

    def test_b_svm(self):
        machine = svm.SVM(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 1)
        machine.build()
        self.assertEqual(machine.get_C(), [[9, -1, -1, 1], [-1, 9, 1, -1], [-1, 1, 9, -1], [1, -1, -1, 9]])

    def test_simplest_b(self):
        machine = svm.SVM(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [1, 1]], [-1, 1], 1)
        machine.build()
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

    def test_generate_equations(self):
        self.assertEqual(wolf_method.generate_equations([[2, 3, 1, 0], [1, 4, 0, 1]], [[6], [5]],
                                                        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                        [[-1], [-2], [0], [0]]),
                         [{'x': [2, 3, 1, 0], 'w': [1, 1], 'v': [0, 0, 0, 0],
                           'u': [0, 0], 'z1': [0, 0, 0, 0], 'z2': [0, 0, 0, 0], 'mup': [0], 'result': [6]},
                          {'x': [1, 4, 0, 1], 'w': [1, 1], 'v': [0, 0, 0, 0],
                           'u': [0, 0], 'z1': [0, 0, 0, 0], 'z2': [0, 0, 0, 0], 'mup': [0],
                           'result': [5]},
                          {'x': [0.5, 0, 0, 0], 'w': [0, 0], 'v': [-1, -1, -1, -1],
                           'u': [2, 1], 'z1': [1, 1, 1, 1], 'z2': [-1, -1, -1, -1], 'mup': [-1],
                           'result': [0]},
                          {'x': [0, 0.5, 0, 0], 'w': [0, 0], 'v': [-1, -1, -1, -1],
                           'u': [3, 4], 'z1': [1, 1, 1, 1], 'z2': [-1, -1, -1, -1], 'mup': [-2],
                           'result': [0]},
                          {'x': [0, 0, 0, 0], 'w': [0, 0], 'v': [-1, -1, -1, -1],
                           'u': [1, 0], 'z1': [1, 1, 1, 1], 'z2': [-1, -1, -1, -1], 'mup': [0],
                           'result': [0]},
                          {'x': [0, 0, 0, 0], 'w': [0, 0], 'v': [-1, -1, -1, -1],
                           'u': [0, 1], 'z1': [1, 1, 1, 1], 'z2': [-1, -1, -1, -1], 'mup': [0],
                           'result': [0]}])

    def test_equation_to_array(self):
        self.assertEqual(wolf_method.equation_to_array([{'x': [1, 2, 3], 'w': [1, 5, 2], 'result': [1]}], ['x']),
                         [1, 2, 3, 1])

    def test_basis_to_array(self):
        self.assertEqual(wolf_method.basis_to_array({'x': [True, False, False], 'w': [False, True, True],
                                                     'result': [1]}, ['x', 'w']),
                         [0, 4, 5])


if __name__ == "__main__":
    unittest.main()
