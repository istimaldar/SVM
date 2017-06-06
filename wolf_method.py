"""
This file implements Wolf method. Main function is wolf_method(..)
"""
from itertools import filterfalse
import simplex


def generate_a(n: int) -> list:
    """
    This function generates a matrix A for the Wolf method
    :param n: number of x variables
    :return: matrix A for Wolf method
    """
    length = 2 * n
    result = []
    for i in range(length - 1):
        equation = []
        for j in range(length):
            if i == 0:
                if j < n:
                    equation.append(1)
                else:
                    equation.append(0)
            else:
                if j == i - 1:
                    equation.append(-1)
                elif j == i + n - 1:
                    equation.append(1)
                else:
                    equation.append(0)
        result.append(equation)
    return result


def generate_b(n: int, C: float) -> list:
    """
    This function generates a vector B for the Wolf method
    :param n: number of x variables
    :param C: svm coefficient C
    :return: vector B for Wolf method
    """
    result = []
    for i in range(2 * n - 1):
        if i == 0:
            result.append([0])
        else:
            result.append([C])
    return result


def generate_p(n: int) -> list:
    """
    This function generates a vector P for the Wolf method
    :param n: number of x variables
    :return: vector P for Wolf method
    """
    return [1 for element in range(n)]


def generate_equations(A: list, B: list, C: list, P: list) -> list:
    """
    Generates a system of linear equations for the Wolff method
    :param A: matrix A for Wolf method
    :param B: vector B for Wolf method
    :param C: matrix C for Wolf method
    :param P: vector P for Wolf method
    :return: list of maps contains coefficients for variables and results
    """
    result = []
    for i in range(len(A)):
        equation = {'x': A[i], 'w': [1 for j in range(len(A))], 'v': [0 for j in range(len(A[i]))],
                    'u': [0 for j in range(len(A))], 'z1': [0 for j in range(len(A[i]))],
                    'z2': [0 for j in range(len(A[i]))], 'mup': [0], 'result': B[i]}
        result.append(equation)
    transpone_A = [list(i) for i in zip(*A)]
    for i in range(len(C)):
        equation = {'x': C[i], 'w': [0 for j in range(len(A))], 'v': [-1 for j in range(len(C[i]))],
                    'u': [j for j in transpone_A[i]], 'z1': [1 for j in range(len(C[i]))],
                    'z2': [-1 for j in range(len(C[i]))], 'mup': P[i], 'result': [0]}
        result.append(equation)
    return result


def equation_to_array(equations: list, variables: list) -> list:
    """
    This function forms a matrix of conditions for the simplex method
    :param equations: list of maps contains coefficients for variables and results
    :param variables: list of the names of variables that must be included in the matrix
    :return: the matrix of conditions for the simplex method
    :except AssertionError: wrong parameters type
    """
    assert equations, list
    assert variables, list
    result = []
    for equation in equations:
        for variable in variables:
            result += equation[variable]
        result += equation.get('result', [])
    return result


def basis_to_array(basis: map, variables: list) -> list:
    """
    This function forms a basis vector for the simplex method
    :param basis: map, contains True if variable is in basis and False otherwise
    :param variables: list of the names of variables that must be included in the matrix
    :return: the basis vector for the simplex method
    :except AssertionError: wrong parameters type
    """
    assert basis, map
    assert variables, list
    start = 0
    result = []
    for variable in variables:
        for i, value in enumerate(basis[variable]):
            if value:
               result.append(start + i)
        start += len(basis[variable])
    return result


def wolf_method(A, B, C, P):
    n = len(A[0])
    result = {'x': [0 for i in A[0]], 'w': [0 for j in range(len(A))], 'v': [0 for j in range(len(A[0]))],
              'u': [0 for j in range(len(A))], 'z1': [0 for j in range(len(A[0]))],
              'z2': [0 for j in range(len(A[0]))], 'mu': [0]}
    basis = {'x': [True for i in A[0]], 'w': [True for j in range(len(A))], 'v': [False for j in range(len(A[0]))],
             'u': [False for j in range(len(A))], 'z1': [False for j in range(len(A[0]))],
             'z2': [False for j in range(len(A[0]))], 'mu': [False]}
    equations = generate_equations(A, B, C, P)


if __name__ == "__main__":
    pass