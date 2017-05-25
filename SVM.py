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
        for i in range(len(X)):
            result.append(SVM.determinant(SVM.crammer_matrix(X, i)) / SVM.determinant(main))
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

    def __init__(self, kernel_type, params, X, Y):
        assert X, list
        for x in X:
            assert x, list
        assert Y, list
        for y in Y:
            assert y, list
        if len(X) != len(Y):
            raise ValueError("X and Y should be the same size")
        self.kernel = SVM.kernel_types[kernel_type].__func__
        print(self.kernel)
        self.matrix = []
        for i in range(len(X)):
            temp = []
            for j in range(len(X)):
                print(str(Y[i]) + "*" + str(Y[j]) + "*" + str(self.kernel(X[i], X[j], *params)))
                temp.append(Y[i] * Y[j] * self.kernel(X[i], X[j], *params))
            self.matrix.append(temp)


if __name__ == "__main__":
    svm = SVM(POLYNOMIAL, [1, 1, 2], [[2, 3], [1, 5]], [4, 2])
    print(svm.matrix)