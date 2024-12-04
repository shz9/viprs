

class OptimizeResult(object):
    """
    A class to store the results/progress of an optimization algorithm.
    Similar to the `OptimizeResult` class from `scipy.optimize`,
    but with a few additional fields and parameters.

    :ivar message: A message about the optimization result
    :ivar stop_iteration: A flag to indicate whether the optimization algorithm has stopped iterating
    :ivar success: A flag to indicate whether the optimization algorithm has succeeded
    :ivar fun: The current objective function value
    :ivar nit: The current number of iterations
    :ivar error_on_termination: A flag to indicate whether the optimization algorithm stopped due to an error.
    """

    def __init__(self):

        self.message = None
        self.stop_iteration = None
        self.success = None
        self.fun = None
        self.nit = 0
        self.error_on_termination = False

        self._last_drop_iter = None
        self._oscillation_counter = 0

    @property
    def iterations(self):
        """
        :return: The current number of iterations.
        """
        return self.nit

    @property
    def objective(self):
        """
        :return: The current value for the objective function.
        """
        return self.fun

    @property
    def converged(self):
        """
        :return: The flag indicating whether the optimization algorithm has converged.
        """
        return self.success

    @property
    def valid_optim_result(self):
        """
        :return: Boolean flag indicating whether the optimization result is valid in
        the sense tht it either successfully converged OR it stopped iterating without
        an error (due to e.g. reaching maximum number of iterations).
        """
        return self.success or (self.stop_iteration and not self.error_on_termination)

    @property
    def oscillation_counter(self):
        """
        :return: The number of oscillations in the objective function value.
        """
        return self._oscillation_counter

    def reset(self):
        """
        Reset the stored values to their initial state.
        """

        self.message = None
        self.stop_iteration = False
        self.success = False
        self.fun = None
        self.nit = 0
        self.error_on_termination = False
        self._last_drop_iter = None
        self._oscillation_counter = 0

    def _reset_oscillation_counter(self):
        """
        Reset the oscillation counter.
        """
        self._oscillation_counter = 0

    def update(self, fun, stop_iteration=False, success=False, message=None, increment=True):
        """
        Update the stored values with new values.
        :param fun: The new objective function value
        :param stop_iteration: A flag to indicate whether the optimization algorithm has stopped iterating
        :param success: A flag to indicate whether the optimization algorithm has succeeded
        :param message: A detailed message about the optimization result.
        :param increment: A flag to indicate whether to increment the number of iterations.
        """

        # If there's a drop in the objective, start tracking potential oscillations:
        if self.fun is not None and fun < self.fun:
            if self._last_drop_iter is not None and self.nit - self._last_drop_iter == 1:
                self._oscillation_counter += 1

            self._last_drop_iter = self.nit + 1
        elif self._last_drop_iter is not None and self.nit > self._last_drop_iter:
            # If there's no drop and the last drop is more than 2 iteration ago,
            # then reset the oscillation counter
            self._reset_oscillation_counter()

        self.fun = fun
        self.stop_iteration = stop_iteration
        self.success = success
        self.message = message

        self.nit += int(increment)

        if stop_iteration and not success and "Maximum iterations" not in message:
            self.error_on_termination = True

    def __str__(self):
        return str(self.__dict__)
