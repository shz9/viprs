import numpy as np
import psutil


def fits_in_memory(alloc_size, max_prop=.9):
    """
    Check whether there's enough memory resources to load an object
    with the given allocation size (in MB).
    :param alloc_size: The allocation size
    :param max_prop: The maximum proportion of available memory allowed for the object
    """

    avail_mem = psutil.virtual_memory().available / (1024.0 ** 2)

    if alloc_size / avail_mem > max_prop:
        return False
    else:
        return True


def dict_concat(d, axis=0):
    """
    Concatenate the values of a dictionary into a single vector
    :param d: A dictionary where values are numeric scalars or vectors
    :param axis: Concatenate along given axis.
    """
    return np.concatenate([d[c] for c in sorted(d.keys())], axis=axis)


def dict_mean(d, axis=None):
    """
    Estimate the mean of the values of a dictionary
    :param d: A dictionary where values are numeric scalars or vectors
    :param axis: Perform aggregation along given axis.
    """
    return np.mean(np.array([np.mean(v, axis=axis) for v in d.values()]), axis=axis)


def dict_sum(d, axis=None, transform=None):
    """
    Estimate the sum of the values of a dictionary
    :param d: A dictionary where values are numeric scalars or vectors
    :param axis: Perform aggregation along given axis.
    :param transform: Transformation to apply before summing.
    """
    if transform is None:
        return np.sum(np.array([np.sum(v, axis=axis) for v in d.values()]), axis=axis)
    else:
        return np.sum(np.array([np.sum(transform(v), axis=axis) for v in d.values()]), axis=axis)


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
    return {c: value*np.ones(shp) for c, shp in shapes.items()}


def expand_column_names(c_name, shape, sep='_'):
    """
    Given a desired column name `c_name` and a matrix `shape`
    that we'd like to apply the column name to, return a list of
    column names for every column in the matrix. The column names will be
    in the form of `c_name` followed by an index, separated by `sep`.

    For example, if the column name is `BETA`, the
    shape is (100, 3) and the separator is `_`, we return a list with:
    [`BETA_0`, `BETA_1`, `BETA_2`]

    If the matrix in question is a vector, we just return the column name
    without any indices appended to it.

    :param c_name: A string object
    :param shape: The shape of a numpy matrix or vector
    :param sep: The separator
    """

    if len(shape) < 2:
        return c_name
    elif shape[1] == 1:
        return c_name
    else:
        return [c_name + f'{sep}{i}' for i in range(shape[1])]
