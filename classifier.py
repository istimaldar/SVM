"""
The file contains a class that is an abstract classifier
"""
from abc import abstractclassmethod


class Classifier:
    """
    An abstract classifier. Contains abstract method train.
    """
    def __init__(self):
        """
        Constructor. Assigns the initial values to the fields of the class.
        """
        self.machine = None
        self.X = None
        self.Y = None
        self.c = 0.
        self.number_of_classes = 0
        self.kernel_type = ""

    @abstractclassmethod
    def train(self, number_of_classes: int, kernel_type: str, params: list, X: list, Y: list, c: float):
        """
        This method builds svm and must be called before using the class
        :param number_of_classes: the number of classes to which the input object
        :param kernel_type: name of kernel type for internal svm. The constants defined in svm namespace must be used.
        :param params: params for internal svm kernel
        :param X: input vectors set
        :param Y: output values set
        :param c: param for all internal svm
        :except AssertionError: raises if params has wrong type
        :except ValueError: raises if params has wrong values
        """
        return
