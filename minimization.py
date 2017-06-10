# TODO: Documentation for file
import quadprog
import typing
import numpy
import sys

Equations = typing.List[typing.Dict[str, typing.List[float]]]
Equation = typing.Dict[str, typing.List[float]]
Matrix = typing.List[typing.List[float]]
Vector = typing.List[float]
Variables = typing.List[str]
Basis = typing.Dict[str, typing.List[float]]


def generate_a(n: int) -> numpy.ndarray:
    return numpy.array([1.0 for i in range(n)])


def generate_g(g: Matrix) -> numpy.ndarray:
    return numpy.array([[float(element) for element in line] for line in g])


def generate_c(n: int, y: list) -> numpy.ndarray:
    return numpy.array([[float(y[i]) for i in range(n)]] +
                       [[1.0 if i == j else .0 for j in range(n)] for i in range(n)] +
                       [[-1.0 if i == j else .0 for j in range(n)] for i in range(n)]).transpose()


def generate_b(n: int, c: float) -> numpy.ndarray:
    return numpy.array([0. for i in range(n + 1)] + [float(-c) for i in range(n)])


def minimize(g: Matrix, n: int, y: list, c: float):
    return quadprog.solve_qp(G=generate_g(g), a=generate_a(n), C=generate_c(n, y), b=generate_b(n, c), meq=1)[0]