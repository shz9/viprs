import numpy as np


def dict_concat(d):
    """
    Concatenate the values of a dictionary into a single vector
    :param d: A dictionary where values are numeric scalars or vectors
    """
    return np.concatenate([d[c] for c in sorted(d.keys())])


def dict_mean(d):
    """
    Estimate the mean of the values of a dictionary
    :param d: A dictionary where values are numeric scalars or vectors
    """
    return np.mean([np.mean(v) for v in d.values()])


def dict_sum(d, transform=None):
    """
    Estimate the sum of the values of a dictionary
    :param d: A dictionary where values are numeric scalars or vectors
    :param transform: Transformation to apply before summing.
    """
    if transform is None:
        return np.sum([np.sum(v) for v in d.values()])
    else:
        return np.sum([np.sum(transform(v)) for v in d.values()])


def dict_elementwise_transform(d, transform):
    """
    Apply a transformation to values of a dictionary
    :param d: A dictionary where values are numeric scalars or vectors
    :param transform: A function to apply to
    """
    return {c: np.vectorize(transform)(v) for c, v in d.items()}


def dict_elementwise_dot(d1, d2):
    """
    Apply element-wise product between the values of two dictionaries

    :param d1: A dictionary where values are numeric scalars or vectors
    :param d2: A dictionary where values are numeric scalars or vectors
    """
    return {c: d1[c]*d2[c] for c, v in d1.items()}


def dict_dot(d1, d2):
    """
    Perform dot product on the elements of d1 and d2
    :param d1: A dictionary where values are numeric scalars or vectors
    :param d2: A dictionary where values are numeric scalars or vectors
    """
    return np.sum([np.dot(d1[c], d2[c]) for c in d1.keys()])


def dict_set(d, value):
    """
    :param d: A dictionary where values are numeric vectors
    :param value: A value to set for all vectors
    """
    for c in d:
        d[c][:] = value

    return d


def dict_repeat(value, shapes):
    """
    Given a value, create a dictionary where the value is repeated
    according to the shapes parameter
    :param shapes: A dictionary of shapes. Key is arbitrary, value is integer input to np.repeat
    :param value:  The value to repeat
    """
    return {c: np.repeat(value, c_size) for c, c_size in shapes.items()}
