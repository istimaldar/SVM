"""
This file implements Wolf method. Main function is wolf_method(..)
"""
import simplex
import typing
Equations = typing.List[typing.Dict[str, typing.List[float]]]
Equation = typing.Dict[str, typing.List[float]]
Matrix = typing.List[typing.List[float]]
Vector = typing.List[float]
Variables = typing.List[str]
Basis = typing.Dict[str, typing.List[float]]

def generate_a(n: int) -> Matrix:
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


def generate_b(n: int, C: float) -> Vector:
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


def generate_p(n: int) -> Vector:
    """
    This function generates a vector P for the Wolf method
    :param n: number of x variables
    :return: vector P for Wolf method
    """
    return [1 for element in range(n)]


def generate_equations(A: Matrix, B: Matrix, C: Matrix, P: Matrix) -> Equations:
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
        equation = {'x': A[i], 'w': [1 if i == j else 0 for j in range(len(A))], 'v': [0 for j in range(len(A[i]))],
                    'u': [0 for j in range(len(A))], 'z1': [0 for j in range(len(A[i]))],
                    'z2': [0 for j in range(len(A[i]))], 'mup': [0], 'result': B[i]}
        result.append(equation)
    transpone_A = [list(i) for i in zip(*A)]
    for i in range(len(C)):
        equation = {'x': C[i], 'w': [0 for j in range(len(A))], 'v': [-1 if i == j else 0 for j in range(len(C[i]))],
                    'u': [j for j in transpone_A[i]], 'z1': [1 if i == j else 0 for j in range(len(C[i]))],
                    'z2': [-1 if i == j else 0 for j in range(len(C[i]))], 'mup': P[i], 'result': [0]}
        result.append(equation)
    return result


def equation_to_array(equations: Equations, variables: Variables) -> Vector:
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
        temp = []
        for variable in variables:
            temp += equation[variable]
        temp += equation.get('result', [])
        result.append(temp)
    return result


def basis_to_array(basis: Basis, variables: Variables) -> Vector:
    """
    This function forms a basis vector for the simplex method
    :param basis: map, contains True if variable is in basis and False otherwise
    :param variables: list of the names of variables that must be included in the matrix
    :return: the basis vector for the simplex method
    :except AssertionError: wrong parameters type
    """
    assert basis, dict
    assert variables, list
    start = 0
    result = []
    for variable in variables:
        for i, value in enumerate(basis[variable]):
            if value:
                result.append(start + i)
        start += len(basis[variable])
    return result


def generate_function(equation: Equation, variables: Variables, function_variable: str) -> Vector:
    """
    Generates coefficients for a minimized function
    :param equation: list of maps contains coefficients for variables and results
    :param variables: list of the names of variables that must be included in the function
    :param function_variable: the variable on which the minimization takes place
    :return: coefficients for a minimized function
    """
    result = []
    for var in variables:
        if var == function_variable:
            result += [1 for element in equation[var]]
        else:
            result += [0 for element in equation[var]]
    return result


#def build_excluded_array(equation: list, )


def array_to_basis(array: Vector, basis: Basis, variables: Variables) -> Basis:
    """
    This function translates the basis returned by the simplex method to a convenient form
    :param array: basis vector return simplex method
    :param basis: old basis
    :param variables: list of variables used to minimize
    :return: new basis

    """
    offset = 0
    for var in variables:
        for i in range(len(basis[var])):
            if offset + i in array:
                basis[var][i] = True
            else:
                basis[var][i] = False
        offset += len(basis[var])
    for key in basis:
        if key not in variables:
            for i in range(len(basis[key])):
                basis[key][i] = False
    return basis


def update_basis_after_first_minimization(basis: Basis, equations: Equations) -> (Basis, Equations):
    """
    This function removes unused z and all w
    :param basis: old basis
    :param equations: old equations
    :return: tuple of two params. First param is new basis. Second param is new equations.
    """
    for i, equation in enumerate(equations):
        z = []
        for condition, z1, z2 in zip(basis['z1'], equation['z1'], equation['z2']):
            if condition:
                z.append(z1)
            else:
                z.append(z2)
        del equations[i]['z1']
        del equations[i]['z2']
        del equations[i]['w']
        equations[i]['z'] = z
    del basis['z1']
    basis['z'] = [True for element in basis['z2']]
    del basis['z2']
    del basis['w']
    return basis, equations


def update_basis_after_second_minimization(basis: Basis, equations: Equations) -> (Basis, Equations):
    """
    This function removes all z
    :param basis: old basis
    :param equations: old equations
    :return: tuple of two params. First param is new basis. Second param is new equations.
    """
    for equation in equations:
        assert equation, dict
    for i in range(len(equations)):
        del equations[i]['z']
    del basis['z']
    return basis, equations


def wolf_method(A: Matrix, B: Matrix, C: Matrix, P: Matrix):
    """

    :param A:
    :param B:
    :param C:
    :param P:
    :return:
    """
    n = len(A[0])
    result = {'x': [0 for i in A[0]], 'w': [0 for j in range(len(A))], 'v': [0 for j in range(len(A[0]))],
              'u': [0 for j in range(len(A))], 'z1': [0 for j in range(len(A[0]))],
              'z2': [0 for j in range(len(A[0]))], 'mu': [0]}
    basis = {'x': [False for i in A[0]], 'w': [True for j in range(len(A))], 'v': [False for j in range(len(A[0]))],
             'u': [False for j in range(len(A))], 'z1': [True for j in range(len(A[0]))],
             'z2': [False for j in range(len(A[0]))], 'mu': [False]}
    equations = generate_equations(A, B, C, P)
    conditions = equation_to_array(equations, ['x', 'w', 'z1', 'z2'])
    function = generate_function(equations[0], ['x', 'w', 'z1', 'z2'], 'w')
    basis_vector = basis_to_array(basis, ['x', 'w', 'z1', 'z2'])
    variables, value, array_basis = simplex.simplex_method(function, conditions, basis_vector)
    basis, equations = update_basis_after_first_minimization(array_to_basis(array_basis, basis, ['x', 'w', 'z1', 'z2']),
                                                             equations)
    conditions = equation_to_array(equations, ['x', 'u', 'v', 'z'])
    function = generate_function(equations[0], ['x', 'u', 'v', 'z'], 'z')
    variables, value, array_basis = simplex.simplex_method(function, conditions, basis_vector, True, None, True,
                                                           len(equations[0]['x']))
    basis, equations = update_basis_after_second_minimization(array_to_basis(array_basis, basis, ['x', 'u', 'v']),
                                                              equations)
    conditions = equation_to_array(equations, ['x', 'u', 'v', 'mup'])
    function = generate_function(equations[0], ['x', 'u', 'v', 'mup'], 'mup')
    variables, value, array_basis = simplex.simplex_method(function, conditions, basis_vector, True, None, True,
                                                           len(equations[0]['x']))
    print(variables)


if __name__ == "__main__":
    wolf_method([[2, 3, 1, 0], [1, 4, 0, 1]], [[6], [5]],
                [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[-1], [-2], [0], [0]])
    print(0.5 * (1.8 ** 2) + 0.5 * (0.8 ** 2) - 1.8 - 1.6)
    print(0.5 * (0.55555 ** 2) + 0.5 * (1.1111 ** 2) - 0.555555 - 2.222)
