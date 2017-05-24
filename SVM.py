class SVM:
    @staticmethod
    def linear_kernel(X, Y, c=0):
        pass

    @staticmethod
    def polynomial_kernel(X, Y, c=0, alpha=0, d=2):
        pass

    @staticmethod
    def gaussian_kernel(X, Y, sigma=1):
        pass

    kernel_types = {"linear": linear_kernel, "polynomial": polynomial_kernel, "gaussian": gaussian_kernel}

    def __init__(self):
        pass

    @staticmethod
    def multiply_vector(X, Y):
        if len(X) == len(Y):
            raise ValueError("X and Y should be the same size")
