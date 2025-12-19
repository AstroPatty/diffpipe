class BadInputError(Exception):
    """
    This is a catch-all for cases where the input provided is invalid. It's just
    used to signal to the pool manager that we need to exit.
    """

    pass
