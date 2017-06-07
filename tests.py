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
        self.assertEqual(wolf_method.generate_a(2), [[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1]])

    def test_b(self):
        self.assertEqual(wolf_method.generate_b(2, 3), [[0], [3], [3]])

    def test_generate_equations(self):
        self.assertEqual(wolf_method.generate_equations([[2, 3, 1, 0], [1, 4, 0, 1]], [[6], [5]],
                                                        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                        [[-1], [-2], [0], [0]]),
                         [{'x': [2, 3, 1, 0], 'w': [1, 0], 'v': [0, 0, 0, 0],
                           'u': [0, 0], 'z1': [0, 0, 0, 0], 'z2': [0, 0, 0, 0], 'mup': [0], 'result': [6]},
                          {'x': [1, 4, 0, 1], 'w': [0, 1], 'v': [0, 0, 0, 0],
                           'u': [0, 0], 'z1': [0, 0, 0, 0], 'z2': [0, 0, 0, 0], 'mup': [0],
                           'result': [5]},
                          {'x': [0.5, 0, 0, 0], 'w': [0, 0], 'v': [-1, 0, 0, 0],
                           'u': [2, 1], 'z1': [1, 0, 0, 0], 'z2': [-1, 0, 0, 0], 'mup': [-1],
                           'result': [0]},
                          {'x': [0, 0.5, 0, 0], 'w': [0, 0], 'v': [0, -1, 0, 0],
                           'u': [3, 4], 'z1': [0, 1, 0, 0], 'z2': [0, -1, 0, 0], 'mup': [-2],
                           'result': [0]},
                          {'x': [0, 0, 0, 0], 'w': [0, 0], 'v': [0, 0, -1, 0],
                           'u': [1, 0], 'z1': [0, 0, 1, 0], 'z2': [0, 0, -1, 0], 'mup': [0],
                           'result': [0]},
                          {'x': [0, 0, 0, 0], 'w': [0, 0], 'v': [0, 0, 0, -1],
                           'u': [0, 1], 'z1': [0, 0, 0, 1], 'z2': [0, 0, 0, -1], 'mup': [0],
                           'result': [0]}])

    def test_equation_to_array(self):
        self.assertEqual(wolf_method.equation_to_array([{'x': [1, 2, 3], 'w': [1, 5, 2], 'result': [1]}], ['x']),
                         [[1, 2, 3, 1]])

    def test_basis_to_array(self):
        self.assertEqual(wolf_method.basis_to_array({'x': [True, False, False], 'w': [False, True, True],
                                                     'result': [1]}, ['x', 'w']),
                         [0, 4, 5])

    def test_generate_function(self):
        self.assertEqual(wolf_method.generate_function({'x': [1, 2, 3], 'w': [1, 5, 2], 'result': [1]}, ['x', 'w'], 'w')
                         , [0, 0, 0, 1, 1, 1])

    def test_array_to_basis(self):
        self.assertEqual(wolf_method.array_to_basis([1, 3, 5], {'x': [True, True, True], 'w': [False, False],
                                                                'z': [True, False], 'd': [True, True, True]},
                                                    ['x', 'w', 'z']), {'x': [False, True, False], 'w': [True, False],
                                                                       'z': [True, False], 'd': [False, False, False]})

    def test_remove_z(self):
        self.assertEqual(wolf_method.update_basis_after_first_minimization({'z1': [True, True, False],
                                                                            'z2': [False, False, True],
                                                                            'w': [False]},
                                                                           [{'z1': [1, 0, 0], 'z2': [-1, 0, 0],
                                                                             'w': [False]},
                                                                            {'z1': [0, 1, 0], 'z2': [0, -1, 0],
                                                                             'w': [False]},
                                                                            {'z1': [0, 0, -1], 'z2': [0, 0, -1],
                                                                             'w': [False]}]),
                         ({'z': [True, True, True]}, [{'z': [1, 0, 0]}, {'z': [0, 1, 0]}, {'z': [0, 0, -1]}]))

    def test_build_excluded_array(self):
        self.assertEqual(wolf_method.build_excluded_array({'x': [1, 2, 3, 4], 'w': [1, 4, 6, 7], 'q': [8, 8, 8]},
                                                          ['x', 'w', 'q'], ['q']), [8, 9, 10])


if __name__ == "__main__":
    unittest.main()
