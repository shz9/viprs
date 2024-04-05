

class OptimizeResult(object):
    """
    A class to store the results/progress of an optimization algorithm.
    Similar to the `OptimizeResult` class from `scipy.optimize`,
    but with a few additional fields.
    """

    def __init__(self):

        self.message = None
        self.stop_iteration = None
        self.success = None
        self.fun = None
        self.nit = None

    @property
    def iterations(self):
        """
        Return the number of iterations at its current value.
        """
        return self.nit

    @property
    def objective(self):
        """
        Return the objective function value at its current value.
        """
        return self.fun

    def reset(self):
        """
        Reset the stored values to their initial state.
        """

        self.message = None
        self.stop_iteration = False
        self.success = False
        self.fun = None
        self.nit = 0

    def update(self, fun, stop_iteration=False, success=False, message=None):
        """
        Update the stored values with new values.
        :param fun: The new objective function value
        :param stop_iteration: A flag to indicate whether the optimization algorithm has stopped iterating
        :param success: A flag to indicate whether the optimization algorithm has succeeded
        :param message: A detailed message about the optimization result.
        """

        self.fun = fun
        self.stop_iteration = stop_iteration
        self.success = success
        self.message = message

        self.nit += 1

    def __str__(self):
        return str(self.__dict__)
