import unittest


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
            if 0 < element < min_value:
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
    result = [0 for i in range(len(conditions[0]) - 1)]
    for key in basis_vectors:
        result[basis_vectors[key]] = basis[key]
    return result


class SimplexTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(solve_simplex([9, 5, 4, 3, 2, 0], [[1, -2, 2, 0, 0, 1, 6], [1, 2, 1, 1, 0, 0, 24],
                                                            [2, 1, -4, 0, 1, 0, 30]]), [0, 7.0, 10.0, 0, 63.0, 0])

if __name__ == "__main__":
    unittest.main()