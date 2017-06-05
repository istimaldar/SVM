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
    @staticmethod
    def solve_crammer(X):
        main = utility.crammer_main_matrix(X)
        result = []
        determinant = utility.determinant(main)
        if determinant == 0:
            return utility.solve_gauss_jordan(X)
        for i in range(len(X)):
            result.append(utility.determinant(utility.crammer_matrix(X, i)) / determinant)
        return result

    @staticmethod
    def linear_kernel(X, Y, c=0, *p):
        return utility.multiply_vector(X, Y) + c

    @staticmethod
    def polynomial_kernel(X, Y, c=0, alpha=0, d=2, *p):
        return (utility.multiply_vector([alpha * x for x in X], Y) + c) ** d

    @staticmethod
    def gaussian_kernel(X, Y, sigma=1, *p):
        return exp(-(utility.euclidean_distance(X, Y) ** 2) / (2 * (sigma ** 2)))

    @staticmethod
    def exponential_kernel(X, Y, sigma=1, *p):
        return exp(-(utility.euclidean_distance(X, Y)) / (2 * (sigma ** 2)))

    @staticmethod
    def laplacian_kernel(X, Y, sigma, *p):
        return exp(-(utility.euclidean_distance(X, Y)) / sigma)

    kernel_types = {"linear": linear_kernel, "polynomial": polynomial_kernel, "gaussian": gaussian_kernel,
                    "exponential": exponential_kernel, "laplacian": laplacian_kernel}

    def __init__(self, kernel_type, params, X, Y, c, crutch=True):
        assert X, list
        for x in X:
            assert x, list
        assert Y, list
        for y in Y:
            assert y, list
        if len(X) != len(Y):
            raise ValueError("X and Y should be the same size")
        self.X = X
        self.Y = Y
        self.b = 0
        self.params = params
        self.kernel = SVM.kernel_types[kernel_type].__func__
        self.c = c
        self.matrix = []
        self.C = []
        for i in range(len(X)):
            alpha = {}
            equation = []
            for j in range(len(X)):
                data = Y[i] * Y[j] * self.kernel(X[i], X[j], *params)
                equation.append(data)
                alpha[j] = data
            h = {0: Y[i]}
            for j in range(1, 2 * len(X) + 1):
                if j == i * 2 + 1:
                    h[j] = -1
                elif j == i * 2 + 2:
                    h[j] = 1
                else:
                    h[j] = 0
            result = 1
            self.matrix.append({"alpha": alpha, "h": h, "result": result, "answer": {"alpha": {}, "h": {}}})
            self.C.append(equation)
        if not crutch:
            alpha = {n: d for n, d in enumerate(Y)}
            h = {n: 0 for n in range(len(X) * 2 + 1)}
            result = 0
            self.matrix.append({"alpha": alpha, "h": h, "result": result, "answer": {"alpha": {}, "h": {}}})
        self.results = []
        self.is_maximization_quadratic_programming_problem()
        if crutch:
            self.minimize_lagrange()
        else:
            self.find_extremums(self.matrix, 0, self.results)
            self.format_results()
            self.calculate()
            self.find_max()
        self.find_b()

    def minimize_lagrange(self):
        crammer_matrix = []
        for equasion in self.matrix:
            crammer_line = []
            for value in sorted(equasion["alpha"]):
                crammer_line.append(equasion["alpha"][value])
            crammer_line.append(equasion["result"])
            crammer_matrix.append(crammer_line)
        try:
            self.results = SVM.solve_crammer(crammer_matrix)
        except ZeroDivisionError:
            print("Error this data set is not lineraly separable in z-space. Try to use other kernel")

    def find_extremums(self, matrix, order, results):
        if len(matrix[0]["alpha"]) + len(matrix[0]["h"]) <= len(matrix):
            crammer_matrix = []
            for equation in matrix:
                matrix_line = []
                for coefficient in sorted(equation["alpha"]):
                    matrix_line.append(equation["alpha"][coefficient])
                for coefficient in sorted(equation["h"]):
                    matrix_line.append(equation["h"][coefficient])
                matrix_line.append(equation["result"])
                crammer_matrix.append(matrix_line)
            try:
                answers = SVM.solve_crammer(crammer_matrix)
            except ZeroDivisionError:
                return
            result = {}
            for key in matrix[0]["answer"]:
                result[key] = matrix[0]["answer"][key]
            for key in sorted(matrix[0]["alpha"]):
                result["alpha"][key] = answers.pop(0)
            for key in sorted(matrix[0]["h"]):
                result["h"][key] = answers.pop(0)
            for h in result["h"]:
                if result["h"][h] < 0:
                    return
            for alpha in result["alpha"]:
                if result["alpha"][alpha] < 0 or result["alpha"][alpha] > self.c:
                    return
            results.append(result)
            return
        modified = copy.deepcopy(matrix[:])
        for equation in modified:
            equation["alpha"].pop(order, None)
            equation["answer"]["alpha"][order] = 0
            equation["h"].pop(2 * order + 2, None)
            equation["answer"]["h"][2 * order + 2] = 0
        self.find_extremums(modified, order + 1, results)
        modified = copy.deepcopy(matrix[:])
        for equation in modified:
            equation["alpha"].pop(order, None)
            equation["answer"]["alpha"][order] = self.c
            equation["result"] -= self.c
            equation["h"].pop(2 * order + 1, None)
            equation["answer"]["h"][2 * order + 1] = 0
        self.find_extremums(modified, order + 1, results)
        modified = copy.deepcopy(matrix[:])
        for equation in modified:
            equation["h"].pop(2 * order + 1, None)
            equation["answer"]["h"][2 * order + 1] = 0
            equation["h"].pop(2 * order + 2, None)
            equation["answer"]["h"][2 * order + 2] = 0
        self.find_extremums(modified, order + 1, results)

    def format_results(self):
        new_results = []
        for result in self.results:
            new_result = []
            for value in sorted(result["alpha"]):
                new_result.append(result["alpha"][value])
            new_results.append(new_result)
        self.results = new_results

    def calculate(self):
        new_results = []
        for result in self.results:
            value = 0
            for alpha in result:
                value += alpha
            for i in range(len(result)):
                for j in range(len(result)):
                    value += result[i] * result[j] * self.Y[i] * self.Y[j] * self.kernel(self.X[i], self.X[j], *self.params)
            new_results.append((result, value))
        self.results = new_results

    def find_max(self):
        if len(self.results) == 0:
            return []
        max_val = (self.results[0][0], self.results[0][1])
        for result in self.results:
            if result[1] > max_val[1]:
                max_val = (result[0], result[1])
        self.results = max_val[0]

    def find_b(self):
        for m, alpha in enumerate(self.results):
            if alpha != 0:
                break
        self.b = 1 / self.Y[m]
        for i in range(len(self.results)):
            self.b -= self.results[i] * self.Y[i] * self.kernel(self.X[i], self.X[m], *self.params)

    def classify(self, vector):
        result = 0
        for i in range(len(self.results)):
            result += self.results[i] * self.Y[i] * self.kernel(self.X[i], vector, *self.params) + self.b
        if result < 0:
            return -1
        return 1

    def hyperplane(self):
        pass

    def is_maximization_quadratic_programming_problem(self):
        matrix = []
        for equasion in self.matrix:
            crammer_line = []
            for value in sorted(equasion["alpha"]):
                crammer_line.append(equasion["alpha"][value])
            crammer_line.append(equasion["result"])
            matrix.append(crammer_line)
        if utility.determinant(matrix) <= 0:
            print("Hyperplane width cannot be minimized using quadratic programming. Possibly, the data is not linearly"
                  " separable in z-space. Try using a different kernel.")

    def draw_plots(self):
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
        result = 0
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                result += self.Y[i] * self.Y[j] * self.kernel(self.X[i], self.X[j], *self.params)
        if result <= 0:
            raise ValueError("The Wolff method for these values is too complicated to compute and therefore will be"
                             " implemented later")

    def get_C(self):
        return self.C


if __name__ == "__main__":
    svm = SVM(POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 1)
    print(svm.matrix)
    print(svm.classify([1, 1]))
    svm.draw_plots()