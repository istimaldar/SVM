import unittest
import svm


class TestClass(unittest.TestCase):
    def test_A_svm(self):
        machine = svm.SVM().train(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 1)
        self.assertEqual(machine.classify([-1, -1]), -1)
        self.assertEqual(machine.classify([-1, 1]), 1)
        self.assertEqual(machine.classify([1, -1]), 1)
        self.assertEqual(machine.classify([1, 1]), -1)

    def test_b_svm(self):
        machine = svm.SVM().train(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 1)
        self.assertEqual(machine.C, [[9, -1, -1, 1], [-1, 9, 1, -1], [-1, 1, 9, -1], [1, -1, -1, 9]])

    def test_simplest_b(self):
        machine = svm.SVM().train(svm.POLYNOMIAL, [1, 1, 2], [[-1, -1], [1, 1]], [-1, 1], 1)
        self.assertEqual(machine.C, [[9, -1], [-1, 9]])

if __name__ == "__main__":
    unittest.main()
