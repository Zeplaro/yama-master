# encoding: utf8

from functools import wraps
from maya import cmds
from . import utils


def mayaundo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            cmds.undoInfo(openChunk=True)
            result = func(*args, **kwargs)
            return result
        finally:
            cmds.undoInfo(closeChunk=True)

    return wrapper


def keepsel(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sel = cmds.ls(sl=True, fl=True)
        result = func(*args, **kwargs)
        cmds.select(sel)
        return result

    return wrapper


def verbose(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"---- Calling {func.__name__}; -args: {args}; --kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"---- Result of {func.__name__} is : {result}; of type : {type(result).__name__}")
        return result

    return wrapper


def condition_debugger(condition):
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            print(
                f"#$@&%*!    Function: '{func.__name__}'; args={args}, kwargs={kwargs}    #$@&%*!"
            )

            if eval(condition):
                raise RuntimeError(
                    f"Condition met before: '{func.__name__}'; Condition: '{condition}'"
                )

            result = func(*args, **kwargs)

            if eval(condition):
                raise RuntimeError(
                    f"Condition met after: '{func.__name__}'; Condition: '{condition}'"
                )

            return result

        return wrap

    return decorator


def string_args(func):
    """
    Converts all args to string, and YamList to list (by using recursive_map), before passing them to the given func.
    Doing this allows to convert the args to str and still have the same behavior as the equivalent cmds function would
    with the same arguments.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(utils.recursive_map(str, args, forcerecursiontypes=True))
        return func(*args, **kwargs)

    return wrapper
