import copy


def multiply_vector(X, Y):
        if len(X) != len(Y):
            raise ValueError("X and Y should be the same size")
        result = 0
        for i in range(len(X)):
            result += X[i] * Y[i]
        return result


def euclidean_distance(X, Y):
    vector = [(x - y) ** 2 for x, y in (X, Y)]
    return sum(vector) ** 0.5


def solve_gauss_jordan(X):
    matrix = copy.deepcopy(X)
    for i in range(len(matrix)):
        mul = matrix[i][i]
        for j in range(i, len(matrix[i])):
            matrix[i][j] /= mul
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