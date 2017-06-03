import unittest

import copy
from itertools import filterfalse


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
        result += ((-1) ** i) * X[0][i] * determinant(minor(X, 0, i))
    return result


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


def crammer_main_matrix(X, *p):
    assert X, list
    for x in X:
        assert x, list
        if len(x) != len(X) + 1:
            raise ValueError("Wrong matrix size")
    return [x[:-1] for x in X]


def solve_crammer(X):
    main = crammer_main_matrix(X)
    result = []
    det = determinant(main)
    for i in range(len(X)):
        result.append(determinant(crammer_matrix(X, i)) / det)
    return result


def multiply_vectors(first, second):
    assert first, list
    assert second, list
    if len(first) != len(second):
        raise ValueError("Vectors must be the same size")
    result = 0
    for i, j in zip(first, second):
        result += i * j
    return result


def rearrange_equations(conditions):
    for i, condition in enumerate(conditions):
        if condition[len(condition) + i - len(conditions) - 1] == 0:
            for j in range(i + 1, len(conditions)):
                if conditions[j][len(condition) + i - len(conditions) - 1] != 0:
                    conditions[j], conditions[i] = conditions[i], conditions[j]


def check_for_maximum(vector):
    for value in vector:
        if value < 0:
            return False
    return True


def find_not_negative_min(tau):
    assert tau, list
    min_value = max([max(i) for i in tau])
    coordinates = (0, 0)
    for i, vector in enumerate(tau):
        for j, element in enumerate(vector):
            if 0 < element:
                min_value = element
                coordinates = (i, j)
                return min_value, coordinates


def solve_simplex(function, conditions):
    assert function, list
    assert conditions, list
    if len(conditions) == 0:
        raise ValueError("It is impossible to use the simplex method for unconditional minimization. "
                         "Set the conditions or try another method")
    length = len(conditions[0])
    for condition in conditions:
        if len(condition) != length:
            raise ValueError("Conditions must be in canonic form")
    if len(function) + 1 != length:
        raise ValueError("Function must be in canonic form")
    rearrange_equations(conditions)
    equations = [condition[-(len(conditions) + 1):] for condition in conditions]
    basis = solve_crammer(equations)
    basis_vectors = {i: len(condition) + i - len(conditions) - 1 for i, condition in enumerate(conditions)}
    C = function[-len(conditions):]
    decomposition_of_vectors = []
    for i, k in enumerate(function):
        Y = [condition[i] for condition in conditions]
        decomposition_of_vectors.append(multiply_vectors(C, Y) - k)
    while not check_for_maximum(decomposition_of_vectors):
        negative_variables = {}
        for i, vector in enumerate(decomposition_of_vectors):
            if vector < 0:
                negative_variables[len(negative_variables)] = i
        tau = []
        for i, res in enumerate(basis):
            line = []
            for key in negative_variables:
                if conditions[i][negative_variables[key]] == 0:
                    line.append(-1)
                else:
                    line.append(res / conditions[i][negative_variables[key]])
            tau.append(line)
        minimum = find_not_negative_min(tau)
        z = []
        for i in range(len(tau[minimum[1][0]])):
            z.append(-1 * tau[minimum[1][0]][i] * decomposition_of_vectors[negative_variables[i]])
        basis_vectors[minimum[1][0]] = negative_variables[z.index(max(z))]
        equations = [[condition[basis_vectors[key]] for key in sorted(basis_vectors)] for condition in conditions]
        equations = [equation + [condition[-1]] for equation, condition in zip(equations, conditions)]
        basis = solve_crammer(equations)
        for i in range(len(conditions)):
            if i != minimum[1][0]:
                mult = conditions[i][basis_vectors[minimum[1][0]]] / conditions[minimum[1][0]][basis_vectors[minimum[1][0]]];
                for j in range(len(conditions[i])):
                    conditions[i][j] -= conditions[minimum[1][0]][j] * mult
        C = [function[basis_vectors[key]] for key in sorted(basis_vectors)]# function[-len(conditions):]
        for i, k in enumerate(function):
            Y = [condition[i] / condition[basis_vectors[n]] for n, condition in enumerate(conditions)]
            decomposition_of_vectors[i] = multiply_vectors(C, Y) - k
        print(basis)
        basis.sort()
    result = [0 for i in range(len(conditions[0]) - 1)]
    for key in basis_vectors:
        result[basis_vectors[key]] = basis[key]
    return result


def calculate_basis(equations, basis):
    new_equations = [[] for equation in equations]
    for i in range(len(equations)):
        for j in basis:
            new_equations[i].append(equations[i][j])
        new_equations[i].append(equations[i][-1])
    return solve_crammer(new_equations)


def build_basis(equations, basis, k):
    new_equations = [[] for equation in equations]
    for i in range(len(equations)):
        for j in basis:
            new_equations[i].append(equations[i][j])
        new_equations[i].append(equations[i][k])

    return solve_crammer(new_equations)


def calculate(function, values):
    result = 0
    for i in range(len(function)):
        result += function[i] * values[i]
    return result


def simplex_method(function, conditions, basis=None, minimization=True, conditional=False, n=0):
    if basis is None:
        basis = []
    if len(basis) > len(conditions):
        basis = basis[:-(len(conditions) - len(basis))]
    offset = len(conditions[0]) - 2
    while len(basis) < len(conditions):
        if offset not in basis:
            basis.append(offset)
        offset -= 1
    while True:
        decompositions = [build_basis(conditions, basis, i) for i in range(len(conditions[0]) - 1)]
        basis_value = calculate_basis(conditions, basis)
        result = calculate(function,
                           [0 if i not in basis else basis_value[basis.index(i)] for i in range(len(function))])
        coefficients = []
        for j, decomposition in enumerate(decompositions):
            coefficient = 0
            for i, value in enumerate(decomposition):
                coefficient += function[basis[i]] * value
            coefficients.append(coefficient - function[j])
        to_replace = max(coefficients) if minimization else min(coefficients)
        if (to_replace <= 0 and minimization) or (to_replace >= 0 and not minimization):
            break
        column = coefficients.index(to_replace)
        if conditional and column < 2 * n:
            if (column < n and column + n in basis) or (column >= n and column - n in basis):
                new_coefficients = coefficients[:]
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


class SimplexTest(unittest.TestCase):
    def test_maximize(self):
        self.assertEqual(simplex_method([9, 5, 4, 3, 2, 0], [[1, -2, 2, 0, 0, 1, 6], [1, 2, 1, 1, 0, 0, 24],
                                                            [2, 1, -4, 0, 1, 0, 30]], [], False)[0],
                         [0, 7.0, 10.0, 0, 63.0, 0])

    def test_minimize(self):
        self.assertEqual(simplex_method([1, 9, 5, 3, 4, 14], [[1, 0, 0, 1, 0, 0, 20], [0, 1, 0, 0, 1, 0, 50],
                                                   [0, 0, 1, 0, 0, 1, 30], [0, 0, 0, 1, 1, 1, 60]],
                             [1, 3, 4, 5])[0], [10, 0, 30, 10, 50, 0])

    def test_one(self):
        self.assertEqual(simplex_method([3, 2, 0, 0, 0],
                                        [[2, 1, 1, 0, 0, 18], [2, 3, 0, 1, 0, 42], [3, 1, 0, 0, 1, 24]], [], False)[0][:2],
                         [3, 12])

if __name__ == "__main__":
    unittest.main()