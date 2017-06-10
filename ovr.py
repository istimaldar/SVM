import svm
from classifier import Classifier
from kernels import *


class OVR(Classifier):
    def __init__(self):
        super().__init__()

    def train(self, number_of_classes: int, kernel_type: str, params: list, X: list, Y: list, c: float):
        """

        :param number_of_classes:
        :param kernel_type:
        :param params:
        :param X:
        :param Y:
        :param c:
        :return:
        """
        self.machine = [svm.SVM() for machine in range(number_of_classes)]
        for n, machine in enumerate(self.machine):
            y_for_svm = [1 if element == n else -1 for element in Y]
            machine.train(kernel_type, params, X, y_for_svm, c)

    def classify(self, vector: list) -> int:
        results = [machine.get_distance(vector) for machine in self.machine]
        maximum = max(results)
        if maximum > 0:
            return results.index(max(results))
        else:
            return -1

if __name__ == '__main__':
    machine = OVR()
    machine.train(2, POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [1, 0, 0, 1], 5)
    print(machine.classify([-1, 1]))
