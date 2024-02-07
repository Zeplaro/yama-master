# encoding: utf8

"""
Contains all the custom exceptions and warning classes and methods
"""

from maya import cmds


class ObjExistsError(Exception):
    pass


def objExists(obj, raiseError=False, verbose=False):
    obj = str(obj)
    if cmds.objExists(obj):
        return True
    elif raiseError:
        if "." in obj:
            raise AttributeExistsError(f"Attribute '{obj}' does not exist in the current scene")
        raise ObjExistsError(f"Object '{obj}' does not exist in the current scene")
    elif verbose == "warning":
        cmds.warning(f"'{obj}' does not exist in the current scene")
    else:
        return False


class AttributeExistsError(AttributeError, ObjExistsError):
    pass
