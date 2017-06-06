"""
This file contains general mathematics operations which are used in other files
"""
import copy


def multiply_vector(X: list, Y: list) -> float:
    """
    Multiplies two vectors of the same size and returns a number
    :param X: first vector
    :param Y: second vector
    :return: result of multiplying
    :except AssertionError: Params are not vectors
    :except ValueError: Vectors are not same size
    """
    assert X, list
    assert Y, list
    if len(X) != len(Y):
        raise ValueError("X and Y should be the same size")
    result = 0
    for i in range(len(X)):
        result += X[i] * Y[i]
    return result


def euclidean_distance(X: list, Y: list) -> float:
    """
    The function calculates the Euclidean distance between two vectors
    :param X: first vector
    :param Y: second vector
    :return: Euclidean distance between two vectors
    :except AssertionError: Params are not vectors
    :except ValueError: Vectors are not same size
    """
    assert X, list
    assert Y, list
    if len(X) != len(Y):
        raise ValueError("X and Y should be the same size")
    vector = [(x - y) ** 2 for x, y in (X, Y)]
    return sum(vector) ** 0.5


def solve_gauss_jordan(X: list) -> list:
    """
    This function solves the system of linear equations by the Gauss-Jordan method and returns the vector of solutions
    :param X: matrix n x n + 1 which is a Gauss matrix
    :return: vector of solutions
    :except AssertionError: Param is not matrix
    :except ValueError: Param size is not n x n + 1
    """
    assert X, list
    if len(X) != len(X[0]) - 1:
        raise ValueError("Wrong matrix size")
    matrix = copy.deepcopy(X)
    #  Leads the matrix to a Gaussian form
    for i in range(len(matrix)):
        mul = matrix[i][i]
        #  Set main diagonal element to 1
        for j in range(i, len(matrix[i])):
            matrix[i][j] /= mul
        #  Set elements to the left of the main diagonal to 0
        for k in range(i + 1, len(matrix)):
            mul = matrix[k][i]
            result = 0
            for j in range(i, len(matrix[i])):
                matrix[k][j] -= matrix[i][j] * mul
                result += matrix[k][j] ** 2
    matrix = [list(reversed(equation)) for equation in reversed(matrix)]
    result = []
    for i in range(len(matrix)):
        result.append(matrix[i][0] / matrix[i][i + 1])
        for j in range(i, len(matrix)):
            matrix[j][0] -= result[i] * matrix[j][i + 1]
            matrix[j][i + 1] = 0
    return list(reversed(result))


def minor(X: list, i: int, j: int) -> list:
    """
    The function finds the matrix minor by element i, j
    :param X: matrix
    :param i: line number 0..n
    :param j: column number 0..m
    :return: matrix whose determinant is a minor
    :except AssertionError: Wrong param types
    :except ValueError: i or j beyond the boundaries of the matrix X
    """
    result = []
    assert X, list
    try:
        X[i][j]
    except IndexError:
        raise ValueError("i and j must be lesser than matrix size")
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


def determinant(X: list) -> float:
    """
    Finds the determinant of the matrix
    :param X: matrix
    :return: determinant of the matrix
    :except AssertionError: X is not matrix
    :except ValueError: Wrong matrix size
    """
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
        result += ((-1) ** i) * X[0][i] * determinant(minor(X, 0, i))
    return result


def crammer_matrix(X: list, j: int) -> list:
    """
    The function constructs a crammer matrix with a number that is transmitted by the second parameter.
    Matrix should be n x n + 1
    :param X: initial matrix
    :param j: number of crammer matrix
    :return: crammer matrix j
    :except AssertionError: X is not matrix or j is not number
    :except ValueError: Wrong matrix size
    """
    assert X, list
    for x in X:
        assert x, list
        if len(x) != len(X) + 1:
            raise ValueError("Wrong matrix size")
    result = []
    #  replaces column j by last column
    for i1 in range(len(X)):
        temp = []
        for j1 in range(len(X)):
            temp.append(X[i1][j1] if j1 != j else X[i1][len(X)])
        result.append(temp)
    return result


def crammer_main_matrix(X: list) -> list:
    """
    Function returns the main matrix of the cramer
    :param X: initial matrix
    :return: main crammer matrix
    :except AssertionError: X is not matrix
    :except ValueError: Wrong matrix size
    """
    assert X, list
    for x in X:
        assert x, list
        if len(x) != len(X) + 1:
            raise ValueError("Wrong matrix size")
    #  Cuts the last column of the matrix
    return [x[:-1] for x in X]


def solve_crammer(X: list) -> list:
    """
    Function solves a system of linear equations, represented with a matrix form by the method of the crammer
    :param X: matrix
    :return: vector of solutions
    :except AssertionError: X is not matrix
    :except ValueError: Wrong matrix size or system of equations is incompatible
    """
    assert X, list
    for x in X:
        assert x, list
        if len(x) - 1 != len(X):
            raise ValueError("Matrix has wrong size")
    main = crammer_main_matrix(X)
    result = []
    try:
        det = determinant(main)
    except ZeroDivisionError:
        raise ValueError("The system of equations is incompatible")
    for i in range(len(X)):
        result.append(determinant(crammer_matrix(X, i)) / det)
    return result
