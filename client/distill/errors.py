class LabError(Exception):
    """Everything this package refuses to do, in one line a learner can act on.

    A notebook shows an exception's message and its traceback; nothing here
    raises with a cause attached, so the message is all there is to read.
    """
