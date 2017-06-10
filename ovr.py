import svm
from classifier import Classifier
from kernels import *


class OVR(Classifier):
    """
    This class implements a multi-class classifier that operates on the principle 'one versus the rest'
    """
    def __init__(self):
        """
        Constructor. Invokes the parent class constructor.
        """
        super().__init__()

    def train(self, number_of_classes: int, kernel_type: str, params: list, X: list, Y: list, c: float):
        """
        This method builds svm and must be called before using the class
        :param number_of_classes: the number of classes to which the input object
        :param kernel_type: name of kernel type for internal svm. The constants defined in svm namespace must be used.
        :param params: params for internal svm kernel
        :param X: input vectors set
        :param Y: output values set. Shows the number of the class to which the learning vector should be assigned.
        Must be in range 0..number_of_classes
        :param c: param for all internal svm
        """
        self.machine = [svm.SVM() for element in range(number_of_classes)]
        for n, machine in enumerate(self.machine):
            y_for_svm = [1 if element == n else -1 for element in Y]
            machine.train(kernel_type, params, X, y_for_svm, c)

    def classify(self, vector: list) -> int:
        """
        The method classifies the input sample.
        :param vector: input sample
        :return: class number to which the object belongs
        """
        results = [element.get_distance(vector) for element in self.machine]
        maximum = max(results)
        return results.index(maximum)

if __name__ == '__main__':
    machine = OVR()
    machine.train(2, POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [1, 0, 0, 1], 5)
    print(machine.classify([-1, 1]))
