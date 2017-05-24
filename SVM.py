from math import exp


class SVM:
    @staticmethod
    def multiply_vector(X, Y):
        if len(X) != len(Y):
            raise ValueError("X and Y should be the same size")
        result = 0
        for x, y in X, Y:
            result += x * y
        return result

    @staticmethod
    def euclidean_distance(X, Y):
        vector = [(x - y) ** 2 for x, y in (X, Y)]
        return sum(vector) ** 0.5

    @staticmethod
    def linear_kernel(X, Y, c=0):
        return SVM.multiply_vector(X, Y) + c

    @staticmethod
    def polynomial_kernel(X, Y, c=0, alpha=0, d=2):
        X = [alpha * x for x in X]
        return (SVM.multiply_vector(X, Y) + c) ** d

    @staticmethod
    def gaussian_kernel(X, Y, sigma=1):
        return exp(-(SVM.euclidean_distance(X, Y) ** 2) / (2 * (sigma ** 2)))

    kernel_types = {"linear": linear_kernel, "polynomial": polynomial_kernel, "gaussian": gaussian_kernel}

    def __init__(self, kernel_type, X, Y):
        pass




if __name__ == "__main__":
    print(SVM.multiply_vector([-1, -1], [-1, -1]))