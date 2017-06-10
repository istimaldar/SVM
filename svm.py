"""
This file implements the basic logic of the operation of the support vector machine
"""
from math import exp

import matplotlib.pyplot as plt

import utility

import minimization

from kernels import *


class SVM:
    """
    This class implements the basic logic of SVM
    """

    kernel_types = {"linear": linear_kernel, "polynomial": polynomial_kernel, "gaussian": gaussian_kernel,
                    "exponential": exponential_kernel, "laplacian": laplacian_kernel}

    def __init__(self):
        """
        SVM constructor.
        WARNING! The constructor call will not prepare svm for usage. You should call build()
        """
        self.X = None
        self.Y = None
        self.b = 0
        self.params = None
        self.kernel = lambda *p: exec('raise ValueError("No kernel is defined")')
        self.c = None
        self.matrix = []
        self.C = []
        self.results = []

    def train(self, kernel_type: str, params: list, X: list, Y: list, c: float) -> None:
        """
        This method builds svm and must be called before using the class
        :param kernel_type: name of kernel type. The constants defined in svm namespace must be used.
        :param params: params for kernel
        :param X: input vectors set
        :param Y: output values set. Should contains only 1 for positive class and -1 for negative class
        :param c: param for svm
        :except AssertionError: raises if params has wrong type
        :except ValueError: raises if params has wrong values
        :return: None
        """
        assert X, list
        for x in X:
            assert x, list
        assert Y, list
        for y in Y:
            assert y, list
        if len(X) != len(Y):
            raise ValueError("X and Y should be the same size")
        for res in Y:
            if res != 1 and res != -1:
                raise ValueError("Y should contains only 1 and -1")
        self.X = X
        self.Y = Y
        self.params = params
        self.kernel = SVM.kernel_types[kernel_type]
        for i in range(len(self.X)):
            alpha = {}
            equation = []
            for j in range(len(self.X)):
                data = self.Y[i] * self.Y[j] * self.kernel(self.X[i], self.X[j], *self.params)
                equation.append(data)
                alpha[j] = data
            h = {0: self.Y[i]}
            for j in range(1, 2 * len(self.X) + 1):
                if j == i * 2 + 1:
                    h[j] = -1
                elif j == i * 2 + 2:
                    h[j] = 1
                else:
                    h[j] = 0
            result = 1
            self.matrix.append({"alpha": alpha, "h": h, "result": result, "answer": {"alpha": {}, "h": {}}})
            self.C.append(equation)
        self.c = c
        # self.__minimize_lagrange()
        self.results = minimization.minimize(self.get_C(), len(self.X), self.Y, self.c)
        self.__find_b()

    def __minimize_lagrange(self) -> None:
        """
        Private method minimizing the width of the space between the positive and negative classes,
        using the method of Lagrange multipliers. Need to solve 3 ** number_of_input_vectors equations.
        Too slow on large data sets.
        :return: None
        :except ValueError: data can't be separated using current kernel. Try to use other kernels.
        """
        crammer_matrix = []
        for equasion in self.matrix:
            crammer_line = []
            for value in sorted(equasion["alpha"]):
                crammer_line.append(equasion["alpha"][value])
            crammer_line.append(equasion["result"])
            crammer_matrix.append(crammer_line)
        try:
            self.results = utility.solve_crammer(crammer_matrix)
        except ZeroDivisionError:
            raise ValueError("Error this data set is not lineraly separable in z-space. Try to use other kernel")

    def __find_b(self) -> None:
        """
        Private method finding the bias coefficient
        :return: None
        """
        for m, alpha in enumerate(self.results):
            if alpha != 0:
                break
        self.b = 1 / self.Y[m]
        for i in range(len(self.results)):
            self.b -= self.results[i] * self.Y[i] * self.kernel(self.X[i], self.X[m], *self.params)

    def classify(self, vector: list) -> int:
        """
        The method classifies the input sample.
        :param vector: input sample
        :return: 1 if sample belongs to positive class, 0 if sample belongs to negative class
        """
        result = 0
        for i in range(len(self.results)):
            result += self.results[i] * self.Y[i] * self.kernel(self.X[i], vector, *self.params) + self.b
        if result < 0:
            return -1
        return 1

    def get_distance(self, vector: list) -> float:
        """
        This method calculates the distance from the point to the reference hyperplane
        :param vector: input sample
        :return: distance from the point to the reference hyperplane
        """
        result = 0
        for i in range(len(self.results)):
            result += self.results[i] * self.Y[i] * self.kernel(self.X[i], vector, *self.params) + self.b
        return result

    def draw_plots(self) -> None:
        """
        Draws a graph on which the points are located. Works only for two-dimensional space.
        Red color draws the points belonging to the positive class. Blue - to the negative.
        :return: None
        """
        if len(self.X[0]) > 2:
            raise ValueError("Only 2d plots is currently supported.")
        plot_data = {"positive": {"x": [], "y": []}, "negative": {"x": [], "y": []}}
        for x in self.X:
            if self.classify(x) == 1:
                plot_data["positive"]["x"].append(x[0])
                plot_data["positive"]["y"].append(x[1])
            else:
                plot_data["negative"]["x"].append(x[0])
                plot_data["negative"]["y"].append(x[1])
        plt.plot(plot_data["positive"]["x"], plot_data["positive"]["y"], 'ro', plot_data["negative"]["x"],
                 plot_data["negative"]["y"], 'bo')
        shift = [
            0.3 * max((max(plot_data["positive"]["x"]), max(plot_data["negative"]["x"]))) -
            min((min(plot_data["positive"]["x"]), min(plot_data["negative"]["x"]))),
            0.3 * max((max(plot_data["positive"]["y"]), max(plot_data["negative"]["y"]))) -
            min((min(plot_data["positive"]["y"]), min(plot_data["negative"]["y"])))
        ]
        borders = [
            min((min(plot_data["positive"]["x"]), min(plot_data["negative"]["x"]))) - shift[0],
            max((max(plot_data["positive"]["x"]), max(plot_data["negative"]["x"]))) + shift[0],
            min((min(plot_data["positive"]["y"]), min(plot_data["negative"]["y"]))) - shift[1],
            max((max(plot_data["positive"]["y"]), max(plot_data["negative"]["y"]))) + shift[1]
        ]
        plt.axis(borders)
        plt.show()

    def get_C(self):
        return self.C


if __name__ == "__main__":
    machine = SVM()
    machine.train(POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 5)
    machine.draw_plots()
