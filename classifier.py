import svm
from abc import abstractclassmethod


class Classifier:
    def __init__(self):
        self.machine = None

    @abstractclassmethod
    def train(self, number_of_classes: int, kernel_type: str, params: list, X: list, Y: list, c: float):
        """

        :return:
        """
        return
