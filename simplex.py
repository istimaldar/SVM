import time
from itertools import filterfalse
import utility


def calculate_basis(equations: list, basis: list) -> list:
    """
    Solves the system of equations relative to the basis
    :param equations: The system of equations to solve
    :param basis: The basis for which the systems will be solved
    :return: Solution vector
    :except AssertionError: wrong parameters type
    """
    assert equations, list
    assert basis, list
    new_equations = [[] for equation in equations]
    for i in range(len(equations)):
        for j in basis:
            new_equations[i].append(equations[i][j])
        new_equations[i].append(equations[i][-1])
    return utility.solve_crammer(new_equations)


def decompose_basis(equations: list, basis: list, k: int) -> list:
    """
    Decomposes the specified variable over the indicated basis in the system of linear equations
    :param equations: system of linear equations in matrix form
    :param basis: the basis on which the variable is decomposed
    :param k: number of the variable for the expansion in the system of equations
    :return: result of decomposition
    :except AssertionError: wrong parameters type
    """
    assert equations, list
    assert basis, list
    new_equations = [[] for equation in equations]
    for i in range(len(equations)):
        for j in basis:
            new_equations[i].append(equations[i][j])
        new_equations[i].append(equations[i][k])
    return utility.solve_crammer(new_equations)


def calculate(function: list, values: list) -> list:
    """
    This function calculates a linear function for given values of the variables
    :param function: vector of coefficients for a linear function
    :param values: vector of variables values
    :return: vector af results
    :except AssertionError: wrong parameters type
    :except ValueError: The number of variables and values are not the same
    """
    assert function, list
    assert values, list
    if len(function) != len(values):
        raise ValueError("The number of variables and values must be the same")
    result = 0
    for i in range(len(function)):
        result += function[i] * values[i]
    return result


def simplex_method(function: list, conditions: list, basis: list=None, minimization: bool=True, excluded: list=None,
                   conditional: bool=False, n: int=0) -> (list, float, list):
    """
    Minimizes or maximizes the selected linear function of the simplex method. An initial basis can be given,
    variables that can not be included in the basis and / or pairs of variables
    that can not be in the basis simultaneously.
    :param function: coefficient vector for the original function in standard form
    :param conditions: linear system of conditions in the standard matrix form
    :param basis: vector of numbers of basic variables
    :param minimization: flag. True if the function is minimizing, false if maximizing.
    :param excluded: vector of variable numbers that can not be included in the basis
    :param conditional: true, if one can not simultaneously include in the basis i and i + n, i=0..n, else otherwise
    :param n: the point of symmetry with respect to which the variables can not be included simultaneously
    :return: return tuple of 3 values. First value result vector, containing variables after minimization/maximization
    :return: Second value is function value. Third value is final basis.
    """
    #  If basis is empty add last n variables to basis
    if basis is None:
        basis = []
    if excluded is None:
        excluded = []
    if len(basis) > len(conditions):
        basis = basis[:-(len(conditions) - len(basis))]
    offset = len(conditions[0]) - 2
    while len(basis) < len(conditions):
        if offset not in basis:
            basis.append(offset)
        offset -= 1
    #  While function not minimal / maximal
    while True:
        time_s = time.time()
        #  Find the expansion of the variable in terms of the basis
        decompositions = [decompose_basis(conditions, basis, i) for i in range(len(conditions[0]) - 1)]
        #  Find the value of the basis variables
        basis_value = calculate_basis(conditions, basis)
        result = calculate(function,
                           [0 if i not in basis else basis_value[basis.index(i)] for i in range(len(function))])
        coefficients = []
        for j, decomposition in enumerate(decompositions):
            coefficient = 0
            for i, value in enumerate(decomposition):
                coefficient += function[basis[i]] * value
            coefficients.append(coefficient - function[j])
        #  Determine which variable to include in the basis
        new_coefficients = [n for i, n in enumerate(coefficients) if i not in excluded]
        to_replace = max(new_coefficients) if minimization else min(new_coefficients)
        if (to_replace <= 0 and minimization) or (to_replace >= 0 and not minimization):
            break
        column = coefficients.index(to_replace)
        if conditional and column < 2 * n:
            if (column < n and column + n in basis) or (column >= n and column - n in basis):
                filterfalse(lambda x: (x <= 0 and minimization) or (x >= 0 and not minimization),
                            new_coefficients)
                filterfalse(lambda x: (x < n and x + n in basis) or (x >= n and x - n in basis),
                            new_coefficients)
                to_replace = max(coefficients) if minimization else min(coefficients)
                if (to_replace <= 0 and minimization) or (to_replace >= 0 and not minimization):
                    break
        to_replace = float('inf')
        string = 0
        for i, coefficient in enumerate(decompositions[column]):
                if coefficient > 0 and (basis_value[i] / coefficient) < to_replace:
                    to_replace = basis_value[i] / coefficient
                    string = i
        basis[string] = column

    basis_value = calculate_basis(conditions, basis)
    return [0 if i not in basis else basis_value[basis.index(i)] for i in range(len(function))], result, basis