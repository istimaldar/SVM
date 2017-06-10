"""
This file contains functions that are kernels
"""
from math import exp

import utility


LINEAR = "linear"
POLYNOMIAL = "polynomial"
GAUSSIAN = "gaussian"
EXPONENTIAL = "exponential"
LAPLICAN = "laplacian"


def linear_kernel(X: list, Y: list, c: float = 0, *p) -> float:
    """
    This method implements linear kernel K(X, Y) = transpose(X) * Y + c
    :param X: first vector
    :param Y: second vector
    :param c: coefficient c
    :param p: other parameters, in case the method receives more
    :return: result of kernel of current vectors
    """
    assert X, list
    assert Y, list
    assert c, float
    return utility.multiply_vector(X, Y) + c


def polynomial_kernel(X: list, Y: list, c: float = 0, alpha: float = 0, d: float = 2, *p) -> float:
    """
    This method implements polynomial kernel K(X, Y) =  (alpha * transpose(X) * Y + c) ** d
    :param X: first vector
    :param Y: second vector
    :param c: coefficient c
    :param alpha: coefficient alpha
    :param d: coefficient d
    :param p: other parameters, in case the method receives more
    :return: result of kernel of current vectors
    """
    assert X, list
    assert Y, list
    assert c, float
    assert alpha, float
    assert d, float
    return (utility.multiply_vector([alpha * x for x in X], Y) + c) ** d


def gaussian_kernel(X: list, Y: list, sigma: float = 1, *p) -> float:
    """
    This method implements gaussian kernel K(X, Y) =  exp(-(||X - Y|| ** 2) / (2 * sigma ** 2))
    :param X: first vector
    :param Y: second vector
    :param sigma: coefficient sigma
    :param p: other parameters, in case the method receives more
    :return: result of kernel of current vectors
    """
    assert X, list
    assert Y, list
    assert sigma, float
    return exp(-(utility.euclidean_distance(X, Y) ** 2) / (2 * (sigma ** 2)))


def exponential_kernel(X: list, Y: list, sigma: float = 1, *p) -> float:
    """
    This method implements exponential kernel K(X, Y) =  exp(-(||X - Y||) / (2 * sigma ** 2))
    :param X: first vector
    :param Y: second vector
    :param sigma: coefficient sigma
    :param p: other parameters, in case the method receives more
    :return: result of kernel of current vectors
    """
    assert X, list
    assert Y, list
    assert sigma, float
    return exp(-(utility.euclidean_distance(X, Y)) / (2 * (sigma ** 2)))


def laplacian_kernel(X: list, Y: list, sigma: float = 1, *p) -> float:
    """
    This method implements laplacian kernel K(X, Y) =  exp(-(||X - Y||) / sigma)
    :param X: first vector
    :param Y: second vector
    :param sigma: coefficient sigma
    :param p: other parameters, in case the method receives more
    :return: result of kernel of current vectors
    """
    assert X, list
    assert Y, list
    assert sigma, float
    return exp(-(utility.euclidean_distance(X, Y)) / sigma)
