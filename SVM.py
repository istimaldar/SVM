from math import exp
LINEAR = "linear"
POLYNOMIAL = "polynomial"
GAUSSIAN = "gaussian"


class SVM:
    @staticmethod
    def multiply_vector(X, Y):
        if len(X) != len(Y):
            raise ValueError("X and Y should be the same size")
        result = 0
        for i in range(len(X)):
            result += X[i] * Y[i]
        return result

    @staticmethod
    def euclidean_distance(X, Y):
        vector = [(x - y) ** 2 for x, y in (X, Y)]
        return sum(vector) ** 0.5

    @staticmethod
    def minor(X, i, j):
        result = []
        for i1, x in enumerate(X):
            if i1 == i:
                continue
            temp = []
            for j1, y in enumerate(x):
                if j1 == j:
                    continue
                temp.append(y)
            result.append(temp)
        return result

    @staticmethod
    def determinant(X):
        assert X, list
        size = len(X[0])
        for x in X:
            assert x, list
            if len(x) != size:
                raise ValueError("Wrong matrix")
        if len(X) == 1:
            return X[0][0]
        result = 0
        for i in range(len(X)):
            result += ((-1) ** i) * X[0][i] * SVM.determinant(SVM.minor(X, 0, i))
        return result

    @staticmethod
    def crammer_matrix(X, j):
        assert X, list
        for x in X:
            assert x, list
            if len(x) != len(X) + 1:
                raise ValueError("Wrong matrix size")
        result = []
        for i1 in range(len(X)):
            temp = []
            for j1 in range(len(X)):
                temp.append(X[i1][j1] if j1 != j else X[i1][len(X)])
            result.append(temp)
        return result

    @staticmethod
    def crammer_main_matrix(X):
        assert X, list
        for x in X:
            assert x, list
            if len(x) != len(X) + 1:
                raise ValueError("Wrong matrix size")
        return [x[:-1] for x in X]

    @staticmethod
    def solve_crammer(X):
        main = SVM.crammer_main_matrix(X)
        result = []
        determinant = SVM.determinant(main)
        if determinant == 0:
            raise ZeroDivisionError
        for i in range(len(X)):
            result.append(SVM.determinant(SVM.crammer_matrix(X, i)) / determinant)
        return result

    @classmethod
    def linear_kernel(X, Y, c=0):
        return SVM.multiply_vector(X, Y) + c

    @classmethod
    def polynomial_kernel(X, Y, c=0, alpha=0, d=2):
        return (SVM.multiply_vector([alpha * x for x in X], Y) + c) ** d

    @classmethod
    def gaussian_kernel(X, Y, sigma=1):
        return exp(-(SVM.euclidean_distance(X, Y) ** 2) / (2 * (sigma ** 2)))

    kernel_types = {"linear": linear_kernel, "polynomial": polynomial_kernel, "gaussian": gaussian_kernel}

    def __init__(self, kernel_type, params, X, Y, c):
        assert X, list
        for x in X:
            assert x, list
        assert Y, list
        for y in Y:
            assert y, list
        if len(X) != len(Y):
            raise ValueError("X and Y should be the same size")
        self.kernel = SVM.kernel_types[kernel_type].__func__
        self.c = c
        self.matrix = []
        for i in range(len(X)):
            alpha = {}
            for j in range(len(X)):
                data = Y[i] * Y[j] * self.kernel(X[i], X[j], *params)
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
        alpha = {n: d for n, d in enumerate(Y)}
        h = {n: 0 for n in range(len(X) * 2 + 1)}
        result = 0
        self.matrix.append({"alpha": alpha, "h": h, "result": result, "answer": {"alpha": {}, "h": {}}})
        self.results = []
        self.form_crammer_matrix(self.matrix, 0, self.results)

    def form_crammer_matrix(self, matrix, order, results):
        # assert results, list
        if len(matrix[0]["alpha"]) + len(matrix[0]["h"]) == len(matrix):
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
            results.append(result)
            return
        modified = matrix[:]
        for equation in modified:
            equation["alpha"].pop(order, None)
            equation["answer"]["alpha"][order] = 0
            equation["h"].pop(2 * order + 2, None)
            equation["answer"]["h"][2 * order + 2] = 0
        self.form_crammer_matrix(modified, order + 1, results)
        modified = matrix[:]
        for equation in modified:
            equation["alpha"].pop(order, None)
            equation["answer"]["alpha"][order] = self.c
            equation["result"] -= self.c
            equation["h"].pop(2 * order + 1, None)
            equation["answer"]["h"][2 * order + 1] = 0
        self.form_crammer_matrix(modified, order + 1, results)
        modified = matrix[:]
        for equation in modified:
            equation["h"].pop(2 * order + 1, None)
            equation["answer"]["h"][2 * order + 1] = 0
            equation["h"].pop(2 * order + 2, None)
            equation["answer"]["h"][2 * order + 2] = 0
        self.form_crammer_matrix(modified, order + 1, results)


if __name__ == "__main__":
    svm = SVM(POLYNOMIAL, [1, 1, 2], [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, -1], 4)
    for line in svm.matrix:
        print(len(line["h"]))