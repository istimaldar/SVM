"""
This file implements the basic logic of the operation of the support vector machine
"""
import copy
from math import exp
import matplotlib.pyplot as plt
import utility

LINEAR = "linear"
POLYNOMIAL = "polynomial"
GAUSSIAN = "gaussian"
EXPONENTIAL = "exponential"
LAPLICAN = "laplacian"


class SVM:
    """
    This class implements the basic logic of SVM
    """
    @staticmethod
    def linear_kernel(X: list, Y: list, c: float=0, *p) -> float:
        """
        This method implements linear kernel K(X, Y) = transpose(X) * Y + c
        :param X: first vector
        :param Y: second vector
        :param c: coefficient c
        :param p: other parameters, in case the method receives more
        :return: result of kernel of current vectors
        """
        assert X, list
        assert Y, list
        assert c, float
        return utility.multiply_vector(X, Y) + c

    @staticmethod
    def polynomial_kernel(X: list, Y: list, c: float=0, alpha: float=0, d: float=2, *p) -> float:
        """
        This method implements polynomial kernel K(X, Y) =  (alpha * transpose(X) * Y + c) ** d
        :param X: first vector
        :param Y: second vector
        :param c: coefficient c
        :param alpha: coefficient alpha
        :param d: coefficient d
        :param p: other parameters, in case the method receives more
        :return: result of kernel of current vectors
        """
        assert X, list
        assert Y, list
        assert c, float
        assert alpha, float
        assert d, float
        return (utility.multiply_vector([alpha * x for x in X], Y) + c) ** d

    @staticmethod
    def gaussian_kernel(X: list, Y: list, sigma: float=1, *p) -> float:
        """
        This method implements gaussian kernel K(X, Y) =  exp(-(||X - Y|| ** 2) / (2 * sigma ** 2))
        :param X: first vector
        :param Y: second vector
        :param sigma: coefficient sigma
        :param p: other parameters, in case the method receives more
        :return: result of kernel of current vectors
        """
        assert X, list
        assert Y, list
        assert sigma, float
        return exp(-(utility.euclidean_distance(X, Y) ** 2) / (2 * (sigma ** 2)))

    @staticmethod
    def exponential_kernel(X: list, Y: list, sigma: float=1, *p) -> float:
        """
        This method implements exponential kernel K(X, Y) =  exp(-(||X - Y||) / (2 * sigma ** 2))
        :param X: first vector
        :param Y: second vector
        :param sigma: coefficient sigma
        :param p: other parameters, in case the method receives more
        :return: result of kernel of current vectors
        """
        assert X, list
        assert Y, list
        assert sigma, float
        return exp(-(utility.euclidean_distance(X, Y)) / (2 * (sigma ** 2)))

    @staticmethod
    def laplacian_kernel(X: list, Y: list, sigma: float=1, *p) -> float:
        """
        This method implements laplacian kernel K(X, Y) =  exp(-(||X - Y||) / sigma)
        :param X: first vector
        :param Y: second vector
        :param sigma: coefficient sigma
        :param p: other parameters, in case the method receives more
        :return: result of kernel of current vectors
        """
        assert X, list
        assert Y, list
        assert sigma, float
        return exp(-(utility.euclidean_distance(X, Y)) / sigma)

    kernel_types = {"linear": linear_kernel, "polynomial": polynomial_kernel, "gaussian": gaussian_kernel,
                    "exponential": exponential_kernel, "laplacian": laplacian_kernel}

    def __init__(self, kernel_type: str, params: list, X: list, Y: list, c: float):
        """
        SVM constructor.
        WARNING! The constructor call will not prepare svm for usage. You should call build()
        :param kernel_type: name of kernel type. The constants defined in svm namespace must be used.
        :param params: params for kernel
        :param X: input vectors set
        :param Y: output values set. Should contains only 1 for positive class and -1 for negative class
        :param c: param for svm
        :except AssertionError: raises if params has wrong type
        :except ValueError: raises if params has wrong values
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
        self.b = 0
        self.params = params
        self.kernel = SVM.kernel_types[kernel_type].__func__
        self.c = c
        self.matrix = []
        self.C = []
        self.results = []

    def build(self) -> None:
        """
        This method builds svm and must be called before using the class
        :return: None
        """
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
        self.__minimize_lagrange()
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
        Private method finding the bias coefficient/
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

    def draw_plots(self) -> None:
        """
        Draws a graph on which the points are located. Works only for two-dimensional space.
        Red color draws the points belonging to the positive class. Blue - to the negative.
        :return: None
        """
        if len(self.X[0]) > 2:
            raise ValueError("Only 2d plots is currently supported.")
        plot_data = {"positive": {"x": [], "y": []}, "negative": {"x" :[], "y": []}}
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

    def check_defines(self):
        """
        Will be implemented in future versions
        :return: None
        """
        result = 0
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                result += self.Y[i] * self.Y[j] * self.kernel(self.X[i], self.X[j], *self.params)
        if result <= 0:
            raise ValueError("The Wolff method for these values is too complicated to compute and therefore will be"
                             " implemented later")

    def get_C(self):
        return self.C