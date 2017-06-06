from itertools import filterfalse

import simplex


def genarate_A(n):
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


def generate_b(n, C):
    result = []
    for i in range(2 * n - 1):
        if i == 0:
            result.append([0])
        else:
            result.append([C])
    return result


def generate_p(n):
    return [1 for element in range(n)]


def generate_w_minimize_system(A, b, C, p):
    n = len(A)
    n1 = len(C)
    result = []
    for i, equation in enumerate(A):
        w = [0 for element in range(n)]
        z1 = [0 for element in range(n1)]
        z2 = [0 for element in range(n1)]
        w[i] = 1
        equation = equation + w + z1 + z2 + b[i]
        result.append(equation)
    for i, equation in enumerate(C):
        equation = [2 * element for element in equation]
        w = [0 for element in range(n)]
        z1 = [0 for element in range(n1)]
        z2 = [0 for element in range(n1)]
        z1[i] = 1
        z2[i] = -1
        equation = equation + [0 for element in range(n - 1)] + w + z1 + z2 + [(-p[i])]
        result.append(equation)
    return result


def generate_equations(A, B, C, P):
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


def find_basis(conditions, n, n1):
    basis = []
    for i in range(2 * n - 1):
        basis.append(len(conditions[i]) - (2 * n1 + 2 * n - i))
    for i in range(n1):
        value = len(conditions[2 * n - 1 + i]) - (2 * n1 - i + 1) if conditions[2 * n - 1 + i][-(2 * n1 - i)] > 0 else \
            len(conditions[2 * n - 1 + i]) - (n1 - i + 1)
        basis.append(value)
    return basis


def generate_z_minimization_function(C, n, C_matrix, basis):
    A = genarate_A(n)
    n1 = len(C_matrix)
    conditions = generate_w_minimize_system(A, generate_b(n, C), C_matrix, generate_p(2))
    for i, condition in enumerate(conditions):
        conditions[i] = condition[:2 * n] + condition[4 * n - 1:]
    basis2 = list(filterfalse(lambda x: x < 2 * n, basis))
    basis2.append(len(conditions[0]) - 1)
    basis2 = [i for i in range(2 * n, len(conditions[0])) if i not in basis2]
    conditions = [list(column) for i, column in enumerate(zip(*conditions)) if i not in basis2]
    conditions = [list(column) for column in zip(*conditions)]
    empty_line = [0 for i in conditions[0]]
    while len(conditions) < len(A) + len(A[0]):
        conditions.append(empty_line[:])
    v = [[-1 if i == j else 0 for j in range(len(A[0]))] for i in range(len(A[0]))]
    empty_line = [0 for i in v]
    while len(v) < len(conditions):
        v = [empty_line[:]] + v
    for i in range(len(conditions)):
        conditions[i] = conditions[i][:2 * n] + v[i] + conditions[i][2 * n:]
    transpose_A = [list(column) for column in zip(*A)]
    empty_line = [0 for i in A]
    while len(transpose_A) < len(conditions):
        transpose_A = [empty_line[:]] + transpose_A
    for i in range(len(conditions)):
        conditions[i] = conditions[i][:2 * n] + transpose_A[i] + conditions[i][2 * n:]
    return conditions


def build_new_basis(basis, n):
    return [element + 3 * n - 1 if element >= 2 * n else element for element in basis]


def minimize_z(A, B, C, P, basis):
    n = len(A[0])
    conditions = generate_z_minimization_function(A, B, C, P)
    function = [1 if len(conditions[0]) - n - 1 <= i < len(conditions[0]) - 1 else 0 for i in range(len(conditions[0]))]
    return simplex.simplex_method(function, conditions, build_new_basis(basis, n), True, n)


def minimize_w(A, B, C, P):
    n = len(A[0])
    conditions = generate_w_minimize_system(A, B, C, P)
    function = [1 if n + len(A) - 1 <= i < n + 2 * len(A) - 1 else 0 for i in range(len(conditions[0]) - 1)]
    return simplex.simplex_method(function, conditions, find_basis(conditions, n, n))


def wolf_method(A, B, C, P):
    n = len(A[0])
    first_phase, result, basis = minimize_w(A, B, C, P)
    first_phase = first_phase[:2 * n] + first_phase[4 * n - 1:]
    for i, element in enumerate(basis):
        if element > 2 * n:
            basis[i] = element - 2 * n + 1
    first_phase = first_phase[2 * n:] + list(filterfalse(lambda x: x == 0, first_phase[:2 * n]))
    return minimize_z(A, B, C, P, basis)


def equation_to_array(equations, variables):
    result = []
    for equation in equations:
        for variable in variables:
            result += equation[variable]
        result += equation['result']
    return result


def basis_to_array(basis, variables):
    start = 0
    result = []
    for variable in variables:
        for i, value in enumerate(basis[variable]):
            if value:
               result.append(start + i)
        start += len(basis[variable])
    return result


def go_for_it(A, B, C, P):
    n = len(A[0])
    result = {'x': [0 for i in A[0]], 'w': [0 for j in range(len(A))], 'v': [0 for j in range(len(A[0]))],
              'u': [0 for j in range(len(A))], 'z1': [0 for j in range(len(A[0]))],
              'z2': [0 for j in range(len(A[0]))], 'mu': [0]}
    basis = {'x': [True for i in A[0]], 'w': [True for j in range(len(A))], 'v': [False for j in range(len(A[0]))],
             'u': [False for j in range(len(A))], 'z1': [False for j in range(len(A[0]))],
             'z2': [False for j in range(len(A[0]))], 'mu': [False]}
    equations = generate_equations(A, B, C, P)


if __name__ == "__main__":
    minimize_w([[2, 3, 1, 0], [1, 4, 0, 1]], [[5], [6]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
               [[-1], [-2], [0], [0]])
