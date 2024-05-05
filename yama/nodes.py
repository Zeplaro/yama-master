# encoding: utf8

"""
Contains all the class for maya node types, and all nodes related functions.
The main functions used would be createNode to create a new node and get it initialized as its proper class type; and
yam or yams to initialize existing objects into their proper class type.
"""

import abc
import math

from maya import cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import maya.OpenMaya as om1

from . import weightlist, config, checks, utils, decorators


def getMObject(node: str) -> om.MObject:
    """
    Gets the associated OpenMaya.MObject for the given node.
    :param node: str, node unique name
    :return: OpenMaya.MObject
    """
    mSelectionList = om.MSelectionList()
    try:
        mSelectionList.add(node)
    except RuntimeError:
        if checks.objExists(node):
            raise Exception(f"More than one object called '{node}'")
        checks.objExists(node, raiseError=True)
    try:
        MObject = mSelectionList.getDependNode(0)
    except Exception as e:
        raise Exception(f"Failed to getDependNode on : '{node}'; {e}")
    return MObject


gmo = getMObject


def yam(node):
    # type: (Yam | str | om.MObject | om.MDagPath | om.MPlug) -> DependNode | 'attributes.Attribute'
    """
    Handles all node class assignment to assign the proper class depending on the node type.
    Also works with passing a 'node.attribute'.

    examples :
    >> yam('skincluster42')
    ---> SkinCluster('skincluster42')

    >> yam('pCube1.tz')
    ---> Attribute('pCube1.tz')
    """

    attribute = None

    if hasattr(node, "isAYamObject"):
        return node

    elif isinstance(node, str):
        if "." in node:  # checking if an attribute or component was given with the node
            node, attribute = node.split(".", 1)
        MObject = getMObject(node)

    elif node.__class__ == om.MObject:  # Not using isinstance() for efficiency
        MObject = node

    elif node.__class__ == om.MDagPath:  # Not using isinstance() for efficiency
        MObject = node.node()

    elif node.__class__ == om.MPlug:  # Not using isinstance() for efficiency
        from . import attributes

        return attributes.Attribute(MPlug=node)

    else:
        raise TypeError(
            "yam(): str, OpenMaya.MObject, OpenMaya.MDagPath or OpenMaya.MPlug expected; "
            f"got {node.__class__.__name__}."
        )

    yam_node = Singleton(MObject)
    if attribute:
        return yam_node.attr(attribute)
    return yam_node


def yams(*nodes):
    # type: (*(Yam | str | om.MObject | om.MDagPath | om.MPlug | list | tuple, ...)) -> YamList
    """
    Returns each node or attributes initialized as their appropriate DependNode or Attribute. See 'yam' function.
    :param nodes: Yam | str | om.MObject | om.MDagPath | om.MPlug of existing nodes.
    :return: YamList of DependNode
    """
    try:
        return YamList(yam(node) for node in nodes)
    # in case a list or tuple was passed as single argument instead of an unpacked list.
    except TypeError as e:
        try:  # trying again using the first element of the nodes.
            return YamList(yam(node) for node in nodes[0])
        # to raise the proper TypeError in case an element of the nodes[0] is the wrong type.
        except TypeError as f:
            raise f
        except Exception:  # in case there was actually an issue with the given nodes.
            raise e


def createNode(*args, **kwargs):
    """
    Creates a node and returns it as its appropriate class depending on its type.
    :param args: cmds args for cmds.createNode
    :param kwargs: cmds kwargs for cmds.createNode
    :return: a DependNode object
    """
    if "ss" not in kwargs and "skipSelect" not in kwargs:
        kwargs["ss"] = True
    return yam(cmds.createNode(*args, **kwargs))


def spaceLocator(name="locator", pos=(0, 0, 0), rot=(0, 0, 0), parent=None, ws=False):
    """
    Creates a locator.
    :param name: str, the locator name
    :param pos: list(float, float, float), the locator position after being parented
    :param rot: list(float, float, float), the locator rotation after being parented
    :param parent: the locator parent
    :param ws: if true sets the pos and rot values in world space
    :return: the locator Transform node object
    """
    loc = yam(cmds.spaceLocator(name=name)[0])
    loc.parent = parent
    loc.setXform(t=pos, ro=rot, ws=ws)
    return loc


@decorators.string_args
def duplicate(*objs, **kwargs):
    """Wrapper for cmds.duplicate"""
    kwargs["fullPath"] = True
    return yams(cmds.duplicate(objs, **kwargs))


@decorators.string_args
def ls(*args, **kwargs):
    """Wrapper for 'maya.cmds.ls' but returning yam objects."""
    if "fl" not in kwargs and "flatten" not in kwargs:
        kwargs["flatten"] = True
    return yams(cmds.ls(*args, **kwargs))


def selected(**kwargs):
    """Returns current scene selection as yam objects; kwargs are passed on to 'ls'."""
    if (
        "os" not in kwargs
        and "sl" not in kwargs
        and "orderedSelection" not in kwargs
        and "selection" not in kwargs
    ):
        kwargs["orderedSelection"] = True
    return ls(**kwargs)


@decorators.string_args
def select(*args, **kwargs):
    """Set current scene selection with given args and kwargs. Allows to pass yam object into the select function."""
    cmds.select(*args, **kwargs)


@decorators.string_args
def listAttr(*args, **kwargs):
    """
    Wrapper for cmds.listAttr returning Component objects.

    Notes:
        There is no way to get the nodes from which the attributes were queried in the results of cmds.listAttr, which
        is why this is so convoluted.
        The node name is extracted and stored before querying the attributes, then plugged back with the attribute
        before being converted to a Component object with cpn.encode.

    Args:
        args (DependNode | Attribute | str): The object, node or attribute to query the attributes from.
        kwargs: kwargs passed on to cmds.listAttr
    Returns:
        list[Attribute, ...]
    """
    kwargs["leaf"] = False

    if not args:  # Working with selection if no args to match cmds.listAttr behaviour.
        args = cmds.ls(orderedSelection=True)

    node_args = []
    for arg in args:
        node, *attrs = arg.split(".")

        # Listing all corresponding nodes in case a wildcard '*' or '?' symbol was used in the node name.
        if "*" in node or "?" in node:
            nodes = cmds.ls(node)
            node_args += [[node, ".".join([node, *attrs])] for node in nodes]

        else:
            node_args.append([node, arg])

    results = []
    for node, arg in node_args:
        attrs = cmds.listAttr(arg, **kwargs) or []
        result = [f"{node}.{attr}" for attr in attrs]

        # If querying sub-attributes of an attribute the original attribute is also returned and needs to be removed
        # from the results.
        if "." in arg and result:
            if result[0] == arg:
                result.pop(0)

        results += result

    return yams(results)


@decorators.string_args
def listHistory(*args, type: "str | [str, ...]" = None, **kwargs) -> ["DependNode", ...]:
    """
    Wrapper for cmds.listHistory returning Yam objects.

    Notes:
        Unlike cmds.listHistory(), this function can take a 'type' kwarg to filter nodes per types.

    Args:
        args (DependNode | str): The node to query the history from.
        type (str): Filters only the nodes of this type.
        kwargs: kwargs passed on to cmds.listHistory .
    Returns:
        list[DependNode, ...]
    """
    history = yams(cmds.listHistory(*args, **kwargs) or [])

    if type:
        history = [x for x in history if x.isa(type)]

    return history


@decorators.string_args
def constraint(*args, type, **kwargs):
    func = getattr(cmds, type)

    return yam(func(*args, **kwargs)[0])


def parentConstraint(*args, **kwargs):
    return constraint(*args, type="parentConstraint", **kwargs)


def pointConstraint(*args, **kwargs):
    return constraint(*args, type="pointConstraint", **kwargs)


def orientConstraint(*args, **kwargs):
    return constraint(*args, type="orientConstraint", **kwargs)


def scaleConstraint(*args, **kwargs):
    return constraint(*args, type="scaleConstraint", **kwargs)


def aimConstraint(*args, **kwargs):
    return constraint(*args, type="aimConstraint", **kwargs)


class Singleton:
    """
    Handles the instances of Yam nodes to return already instantiated nodes instead of creating a new object if the node
    has already been instantiated.
    """

    _instances = {}

    def __new__(cls, MObject):
        # type: (Singleton, om.MObject) -> DependNode
        """
        Returns the node instance corresponding to the given MObject if it has already been instantiated, otherwise it
        returns a new instance of the given nodeClass corresponding to the given MObject.
        :param MObject: A valid OpenMaya MObject corresponding to a node.
        """
        try:  # Faster than checking for type using isinstance
            handle = om.MObjectHandle(MObject)
        except ValueError:
            raise TypeError(
                f"Expected OpenMaya.MObject type, instead got : '{type(MObject).__name__}'"
            )
        if not handle.isValid():
            raise ValueError("Given MObject does not contain a valid node")
        hash_code = handle.hashCode()

        if config.use_singleton and hash_code in cls._instances:
            return cls._instances[hash_code]

        else:  # finding if node type has a supported class
            type_id = MObject.apiType()
            # skips the SupportedTypes.getclass if exact type is in supported_class
            if type_id in SupportedTypes.classes_MFn:
                assigned_class = SupportedTypes.classes_MFn[type_id]
            else:
                assigned_class = cls.getclass_cmds(MObject)
            yam_node = assigned_class(MObject)
            cls._instances[hash_code] = yam_node
            return yam_node

    @classmethod
    def getclass(cls, MObject):
        """
        Gets the class that most closely corresponds to the given MObject.

        Raises an error if no corresponding class was found, not even DependNode, meaning that
        MObject.hasFn(om.MFn.KDependencyNode) returned : False.

        Has been found to be faster than getting a reversed om.MGlobal.getFunctionSetList(MObject) and returning the
        first matched MFn from SupportedTypes.classes_MFn.

        :param MObject: A valid OpenMaya.MObject
        :return: assigned class
        """

        def getit(data, assigned=None):
            for (child, fn), sub_data in data.items():
                if MObject.hasFn(fn):
                    return getit(sub_data, child)
            return assigned

        node_class = getit(SupportedTypes.inheritance_tree)
        if node_class:
            return node_class
        raise ValueError("Given MObject does not contain a valid dependencyNode.")

    @classmethod
    def getclass_cmds(cls, MObject):
        """
        Old way to get the class that most closely corresponds to the given MObject.

        Slower than getclass but could be useful for more granular class assignment;
        i.e., there is no way to check for a node that would inherit from ControlPoint only, because there is no
        kControlPoint in om.MFn, but this is possible using cmds.nodeType(node_name, inherited=True) and match the
        returned type to the closest supported class.

        Raises an error if no corresponding class was found, not even DependNode, meaning that
        OpenMaya.MFnDependencyNode(MObject).name() failed.
        :param MObject: A valid OpenMaya.MObject
        :return: assigned class
        """
        # checking if node type has a supported class, if not defaults to DependNode
        if MObject.hasFn(om.MFn.kDagNode):
            # Getting the shortest unique name
            node_name = om.MDagPath.getAPathTo(MObject).partialPathName()
        else:
            try:
                node_name = om.MFnDependencyNode(MObject).name()
            except RuntimeError as e:
                raise ValueError(f"Given MObject does not contain a valid dependencyNode; {e}")
        # checks each inherited types for the node
        for node_type in reversed(cmds.nodeType(node_name, i=True)):
            if node_type in SupportedTypes.classes_str:
                return SupportedTypes.classes_str[node_type]
        return DependNode

    @classmethod
    def clear(cls):
        cls._instances = {}

    @classmethod
    def exists(cls, MObject):
        try:  # Faster than checking for type using isinstance
            handle = om.MObjectHandle(MObject)
        except ValueError:
            raise TypeError(
                f"Expected OpenMaya.MObject type, instead got : '{type(MObject).__name__}'"
            )

        hash_code = handle.hashCode()
        if hash_code in cls._instances:
            return cls._instances[hash_code]
        return False


class Yam(abc.ABC):
    """
    Abstract class for all objects related to maya nodes, attributes and components.
    Should not be instantiated by itself.
    """

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @property
    def isAYamObject(self):
        """Used to check if an object is an instance of Yam with the faster hasattr instead of slower isinstance."""
        return True

    def __str__(self):
        return self.name

    def __repr__(self):
        """
        Not a valid Python expression that could be used to recreate an object with the same value since the MObject and
        MFnDependencyNode don't have clean str representation.
        """
        return f"{self.__class__.__name__}('{self.name}')"

    def __bool__(self):
        """
        Needed for Truth testing since __len__ is defined but does not work on non ControlPoint nodes.
        :return: True
        """
        return True

    @abc.abstractmethod
    def types(self):
        pass

    def isa(self, types) -> bool:
        """Returns True is self is of the given type."""
        if isinstance(types, str):
            return types in self.types()

        elif isinstance(types, (list, tuple)):
            for t in types:
                if self.isa(t):
                    return True
            return False

        elif isinstance(types, type):
            return isinstance(self, types)

        raise TypeError(
            f"{type(types).__name__} is not a valid parameter type for .isa() , "
            "valid  types are: str, list, tuple, type."
        )


class DependNode(Yam):
    """
    Main maya node type that all nodes inherits from.
    Links the node to its maya api 2.0 MObject to access many of the faster api functions and classes (MFn), and also in
    case the node name changes.

    All implemented maya api functions and variables are from api 2.0; api 1.0 functions and variables that are
    implemented, are defined with a '1' suffix.
    e.g.:
    >>> node = yam('pCube1')
    >>> node.MObject   # <-- returns a api 2.0 object
    >>> node.MObject1  # <-- returns a api 1.0 object
    """

    _MFN_FUNC = om.MFnDependencyNode
    _MFN_OBJECT = "MObject"

    def __init__(self, MObject):
        """
        Needs a maya api 2.0 MObject associated to the node and a MFnDependencyNode initialized with the MObject.
        :param MObject: maya api MObject
        """
        super().__init__()
        self.MObject = MObject
        self._MObject1 = None
        self._MFn = None
        self._attributes = {}
        self._hashCode = None
        self._types = None

    @property
    def isAYamNode(self):
        """Used to check if an object is an instance of DependNode with the faster hasattr instead of slower
        isinstance."""
        return True

    @property
    def MFn(self):
        """
        Gets the associated api 2.0 MFn object
        :return: api 2.0 MFn Object
        """
        if self._MFn is None:
            self._MFn = self._MFN_FUNC(getattr(self, self._MFN_OBJECT))
        return self._MFn

    def __getattr__(self, attr):
        """
        Returns an Attribute linked to self.
        :param attr: str
        :return: Attribute object
        """
        return self.attr(attr)

    def __add__(self, other):
        """
        Adds the node's name and returns a str
        :param other: str or DependNode
        :return: str
        """
        return self.name + other

    def __iadd__(self, other):
        self.rename(self.name + str(other))

    def __radd__(self, other):
        """
        Adds the node's name and returns a str.
        :param other: str or DependNode
        :return: str
        """
        return other + self.name

    def __eq__(self, other):
        """
        Compare the api 2.0 MObject for the == operator.
        :param other:
        :return:
        """
        if hasattr(other, "MObject"):
            return self.MObject == other.MObject
        else:
            try:
                return self.MObject == yam(other).MObject
            except (RuntimeError, TypeError):
                return False

    def __hash__(self):
        return self.hashCode

    @property
    def MObject1(self):
        """
        Gets the associated api 1.0 MObject
        :return: api 1.0 MObject
        """
        if self._MObject1 is None:
            mSelectionList = om1.MSelectionList()
            mSelectionList.add(self.name)
            MObject = om1.MObject()
            mSelectionList.getDependNode(0, MObject)
            self._MObject1 = MObject
        return self._MObject1

    def rename(self, newName):
        """
        Renames the node.
        Needs to use cmds to be undoable.
        :param newName: str
        """
        if not config.undoable:
            self.MFn.setName(newName)
        else:
            cmds.rename(self.name, newName)

    @property
    def name(self):
        return self.MFn.name()

    @name.setter
    def name(self, value):
        self.rename(value)

    @property
    def shortName(self):
        """Returns the node name only, without any '|' and 'parent' in case other nodes have the same name"""
        return self.MFn.name()

    def attr(self, attr):
        """
        Gets an Attribute object for the given attr.
        This function should be used in case the attr conflicts with a function or attribute of the class.
        :param attr: str
        :return: Attribute object
        """
        from . import attributes

        if config.use_singleton:
            if attr in self._attributes and not self._attributes[attr].MPlug.isNull:
                return self._attributes[attr]

        attribute = attributes.getAttribute(self, attr)
        self._attributes[attr] = attribute
        return attribute

    def hasattr(self, attr):
        return checks.objExists(f"{self}.{attr}")

    def addAttr(self, longName, **kwargs):
        # Checks if 'attributeType' or 'at' is in kwargs and has a value
        if "at" not in kwargs and "attributeType" not in kwargs:
            raise RuntimeError(
                f"Failed to add attribute '{longName}' on '{self}'; No attribute type given"
            )

        # Enabling hasMinValue or hasMaxValue toggle if a min or max value is given
        if (kwargs.get("min") or kwargs.get("minValue")) and not (
            "hasMinValue" in kwargs or "hnv" in kwargs
        ):
            kwargs["hasMinValue"] = True
        if (kwargs.get("max") or kwargs.get("maxValue")) and not (
            "hasMaxValue" in kwargs or "hxv" in kwargs
        ):
            kwargs["hasMaxValue"] = True

        cmds.addAttr(self.name, longName=longName, **kwargs)
        return self.attr(longName)

    def listRelatives(self, **kwargs):
        """
        Returns the maya cmds.listRelatives as DependNode objects.
        :param kwargs: kwargs passed on to cmds.listRelatives
        :return: list[DependNode, ...]
        """
        kwargs["fullPath"] = True  # Needed in case of multiple obj with same name
        return yams(cmds.listRelatives(self.name, **kwargs) or [])

    def listConnections(self, **kwargs):
        """
        Returns the maya cmds.listConnections as DependNode, Attribute or Component objects.
        'skipConversionNodes' set to True by default if not in kwargs.
        :param kwargs: kwargs passed on to cmds.listConnections
        :return: list[Attribute, ...]
        """
        if "scn" not in kwargs and "skipConversionNodes" not in kwargs:
            kwargs["scn"] = True
        return yams(cmds.listConnections(self.name, **kwargs) or [])

    def inputs(self, **kwargs):
        """
        Returns listConnections with destination connections disabled.
        See listConnections for more info.
        """
        return self.listConnections(destination=False, **kwargs)

    def outputs(self, **kwargs):
        """
        Returns listConnections with source connections disabled.
        See listConnections for more info.
        """
        return self.listConnections(source=False, **kwargs)

    def listAttr(self, **kwargs):
        return listAttr(self, **kwargs)

    def listHistory(self, **kwargs):
        """
        Returns the maya cmds.listHistory as DependNode, Attribute or Component objects.
        Allows the kwarg 'type' (unlike cmds.listHistory) to only return objects of given type.
        :param kwargs: kwargs passed on to cmds.listConnections
        :return: list[Attribute, ...]
        """
        type_ = None
        if "type" in kwargs:
            type_ = kwargs.pop("type")
        nodes = yams(cmds.listHistory(self.name, **kwargs))
        if type_:
            nodes.keepType(type_)
        return nodes

    def type(self) -> str:
        """
        Returns the node type name
        :return: str
        """
        return self.types()[-1]

    def types(self) -> [str, ...]:
        """
        Lists the inherited maya types.
        e.g.: for a joint -> ['containerBase', 'entity', 'dagNode', 'transform', 'joint']
        :return: list of inherited maya types
        """
        if self._types is None:
            self._types = cmds.nodeType(self.name, inherited=True)
        return self._types

    @property
    def hashCode(self):
        if not self._hashCode:
            self._hashCode = om.MObjectHandle(self.MObject).hashCode()
        return self._hashCode

    def uuid(self, asString=True):
        if asString:
            return self.MFn.uuid().asString()
        return self.MFn.uuid()


class DagNode(DependNode):
    """
    Subclass for all maya dag nodes.
    """

    _MFN_FUNC = om.MFnDagNode

    def __init__(self, MObject):
        super().__init__(MObject)
        self._MDagPath = None
        self._MDagPath1 = None

    @property
    def MDagPath(self):
        """
        Gets the associated api 2.0 MDagPath
        :return: api 2.0 MDagPath
        """
        if self._MDagPath is None or not self._MDagPath.isValid():
            self._MDagPath = om.MDagPath.getAPathTo(self.MObject)
        return self._MDagPath

    @property
    def MDagPath1(self):
        """
        Gets the associated api 1.0 MDagPath
        :return: api 1.0 MDagPath
        """
        if self._MDagPath1 is None:
            self._MDagPath1 = om1.MDagPath.getAPathTo(self.MObject1)
        return self._MDagPath1

    @property
    def parent(self):
        """
        Gets the node parent node or None if in world.
        :return: DependNode or None
        """
        parent = self.MFn.parent(0)
        if parent.apiTypeStr == "kWorld":
            return None
        return yam(parent)

    @parent.setter
    def parent(self, parent):
        """
        Sets the node under the given parent or world if None.
        :return: DependNode or None
        """
        if self.parent == parent:
            return
        elif parent is None:
            cmds.parent(self.name, world=True)
        else:
            cmds.parent(self.name, parent)

    @property
    def name(self):
        """
        Returns the minimum string representation in case of other objects with same name.
        """
        return self.MDagPath.partialPathName()

    @name.setter
    def name(self, value):
        """
        Property setter needed again even if defined in parent class.
        """
        self.rename(value)

    @property
    def longName(self):
        """
        Gets the full path name of the object including the leading |.
        :return: str
        """
        return self.MDagPath.fullPathName()

    @property
    def fullPath(self):
        return self.longName

    def allParents(self, reverse: bool = False) -> ["Transform", ...]:
        """
        Returns a list of all the nodes parent up to world.
        Order if not reversed is from closest to the world to closest to the node.

        Args:
            reverse (bool): If True: returns the list in reverse order.

        Returns:
            (list): List of the node's parents.
        """

        def getParents(node):
            """Gets the parents of the node recursively."""
            parent = node.parent
            if parent:
                return getParents(parent) + [parent]
            return []

        parents = getParents(self)
        if reverse:
            parents.reverse()
        return parents


class Transform(DagNode):
    _MFN_FUNC = om.MFnTransform

    def __len__(self):
        shape = self.shape
        if shape:
            return len(shape)
        raise TypeError(f"object has no shape and type '{self.__class__.__name__}' has no len()")

    def __contains__(self, item):
        """
        Checks if the item is a children of self.
        """
        if not isinstance(item, DependNode):
            try:
                item = yam(item)
            except Exception as e:
                raise RuntimeError(f"in '{self}.__contains__('{item}')'; {e}")
        return item.longName.startswith(self.longName)

    def attr(self, attr):
        """
        Gets an Attribute or Component object for the given attr.
        This function should be use in case the attr conflicts with a function or attribute of the class.
        :param attr: str
        :return: Attribute object
        """
        from . import components

        try:
            return components.getComponent(self, attr)  # Trying to get component if one
        except (RuntimeError, TypeError):
            return super().attr(attr)

    def children(self, type=None, noIntermediate=True):
        """
        Gets all the node's children as a list of DependNode.
        :param type: Only returns nodes of the given type.
        :param noIntermediate: if True skips intermediate shapes.
        :return: list of DependNode
        """
        children = yams(self.MDagPath.child(x) for x in range(self.MDagPath.childCount()))
        if noIntermediate:
            children = YamList(x for x in children if not x.MFn.isIntermediateObject)
        if type:
            children.keepType(type)
        return children

    @property
    def child(self):
        if self.MDagPath.childCount():
            return yam(self.MDagPath.child(0))

    def shapes(self, type=None, noIntermediate=True):
        """
        Gets the shapes of the transform as DependNode.
        :param noIntermediate: if True skips intermediate shapes.
        :return: list of Shape object
        """
        children = yams(self.MDagPath.child(x) for x in range(self.MDagPath.childCount()))
        children = YamList(x for x in children if isinstance(x, Shape))
        if noIntermediate:
            children = YamList(x for x in children if not x.MFn.isIntermediateObject)
        if type:
            children.keepType(type)
        return children

    @property
    def shape(self):
        """
        Returns the first shape if it exists.
        :return: Shape object
        """
        shapes = self.shapes()
        return shapes[0] if shapes else None

    def allDescendents(self, type=None):
        if type:
            return self.listRelatives(allDescendents=True, type=type)
        return self.listRelatives(allDescendents=True)

    def getXform(self, **kwargs):
        """
        Wraps the query of cmds.xform, the kwarg query=True is not needed and will be set to True.
        :param kwargs: any kwargs queryable from cmds.xform
        :return: the queried value·s
        """
        kwargs["q"] = True
        return cmds.xform(self.name, **kwargs)

    def setXform(self, **kwargs):
        """
        Wraps the of cmds.xform, the kwarg query=True cannot be used.
        :param kwargs: any kwargs settable with cmds.xform
        :return: the queried value·s
        """
        if "q" in kwargs or "query" in kwargs:
            raise RuntimeError("setXform kwargs cannot contain 'q' or 'query'")
        cmds.xform(self.name, **kwargs)

    def getPosition(self, ws=False):
        """
        Gets the translate position of the node.
        :param ws: If True returns the world space position of the node
        :return: [float, float, float]
        """
        return self.getXform(t=True, ws=ws, os=not ws)

    def setPosition(self, value, ws=False):
        """
        Sets the translate position of the node.
        :param value: [float, float, float]
        :param ws: If True sets the world space position of the node
        """
        self.setXform(t=value, ws=ws, os=not ws)

    def distance(self, obj):
        """
        Gets the distance between self and given obj
        :param obj: Transform, Component or str
        :return: float
        """
        if isinstance(obj, str):
            obj = yam(obj)
        from . import components

        if not isinstance(obj, (Transform, components.Component)):
            raise AttributeError(
                "wrong type given, expected : 'Transform', 'Component' or 'str', "
                f"got : {obj.__class__.__name__}"
            )

        return utils.distance(self.getPosition(ws=True), obj.getPosition(ws=True))


class Joint(Transform):
    pass


class Constraint(Transform):
    """
    Base class for constraint nodes.
    Should not be instanced on its own and only used as inherited class for actual
    constraint type nodes, e.g.: 'parentConstraint', 'orientConstraint', etc...
    """

    _CMDS_FUNC = None

    def _raise_no_func(self):
        raise RuntimeError(
            f"No associated cmds constraint function found for : '{self.name}'. Constraint should"
            " not be instanced on its own, only used as inherited class for actual constraint type"
            " nodes, e.g.: 'parentConstraint', 'orientConstraint', etc..."
        )

    def weightAttrs(self):
        """
        Gets the node's constraint weight attributes.
        :return: YamList([Attribute, ])
        """
        if not self._CMDS_FUNC:
            self._raise_no_func()
        return YamList(
            self.attr(attr) for attr in self._CMDS_FUNC(self.name, q=True, weightAliasList=True)
        )

    def weightTargets(self):
        """
        Gets the node's constraint 'target' (is how maya calls them), the nodes that control the constrained node.
        :return: YamList([Transform, ])
        """
        if not self._CMDS_FUNC:
            self._raise_no_func()
        return yams(self._CMDS_FUNC(self.name, q=True, targetList=True))


class ParentConstraint(Constraint):
    _CMDS_FUNC = cmds.parentConstraint


class OrientConstraint(Constraint):
    _CMDS_FUNC = cmds.orientConstraint


class PointConstraint(Constraint):
    _CMDS_FUNC = cmds.pointConstraint


class ScaleConstraint(Constraint):
    _CMDS_FUNC = cmds.scaleConstraint


class AimConstraint(Constraint):
    _CMDS_FUNC = cmds.aimConstraint


class Shape(DagNode):
    @property
    def parent(self):
        return super().parent

    @parent.setter
    def parent(self, parent):
        if self.parent is parent:
            return
        cmds.parent(self.name, parent, r=True, s=True)


class ControlPoint(Shape):
    """
    Handles shape that have control point; e.g.: Mesh, NurbsCurve, NurbsSurface, Lattice
    """

    def __len__(self):
        return len(cmds.ls(self.name + ".cp[*]", fl=True))

    def getPositions(self, ws=False):
        return [cp.getPosition(ws=ws) for cp in self.cp]

    def setPositions(self, data, ws=False):
        for cp, pos in zip(self.cp, data):
            cp.setPosition(pos, ws=ws)

    def attr(self, attr):
        """
        Gets an Attribute or Component object for the given attr.
        This function should be use in case the attr conflicts with a function or attribute of the class.
        :param attr: str
        :return: Attribute object
        """
        try:
            from . import components

            return components.getComponent(self, attr)  # Trying to get component if one
        except (RuntimeError, TypeError):
            return super().attr(attr)


class SurfaceShape(ControlPoint):
    """
    Handles control point shapes that have a surface; e.g.: Mesh and NurbsSurface
    Mainly used to check object isinstance.
    """

    pass


class Mesh(SurfaceShape):
    _MFN_FUNC = om.MFnMesh
    _MFN_OBJECT = "MDagPath"

    def __len__(self):
        """
        Returns the mesh number of vertices.
        :return: int
        """
        return self.MFn.numVertices

    def shells(self, indexOnly=False):
        polygon_counts, polygon_connects = self.MFn.getVertices()
        faces_vtxs = []
        i = 0
        for poly_count in polygon_counts:
            faces_vtxs.append(polygon_connects[i : poly_count + i])
            i += poly_count

        shells = []
        finished = False
        while not finished:
            finished = True
            for face_vtxs in faces_vtxs:
                added = False
                for shell in shells:
                    for vtx in face_vtxs:
                        if vtx in shell:
                            shell.update(face_vtxs)
                            added = True
                            finished = False
                            break
                    if added:
                        break
                if not added:
                    shells.append(set(face_vtxs))
            if not finished:
                faces_vtxs = shells
                shells = []
        if indexOnly:
            return shells
        vtxs = []
        for shell in shells:
            vtxs.append(YamList(self.vtx[x] for x in shell))
        return vtxs


class NurbsCurve(ControlPoint):
    _MFN_FUNC = om.MFnNurbsCurve
    _MFN_OBJECT = "MDagPath"

    def __len__(self):
        """
        Returns the mesh number of cvs.
        :return: int
        """
        return self.MFn.numCVs

    @staticmethod
    def create(cvs, knots, degree, form, parent):
        if isinstance(parent, str):
            parent = yam(parent).MObject
        elif isinstance(parent, Transform):
            parent = parent.MObject
        else:
            raise TypeError(
                f"Expected parent of type str or Transform, got : {parent}, {type(parent).__name__}"
            )
        curve = om.MFnNurbsCurve().create(cvs, knots, degree, form, False, True, parent)
        return yam(curve)

    def arclen(self, ws=False):
        """
        Gets the world space or object space arc length of the curve.
        :param ws: if False return the objectSpace length.
        :return: float
        """
        if not ws:
            return self.MFn.length()
        return cmds.arclen(self.name)

    @property
    def data(self):
        data = {
            "cvs": self.MFn.cvPositions(),
            "knots": self.knots(),
            "degree": self.degree(),
            "form": self.form(),
        }
        return data

    def knots(self):
        return list(self.MFn.knots())

    def degree(self):
        return self.MFn.degree

    def form(self):
        return self.MFn.form


class NurbsSurface(SurfaceShape):
    _MFN_FUNC = om.MFnNurbsSurface
    _MFN_OBJECT = "MDagPath"

    def __len__(self):
        return self.lenU() * self.lenV()

    def lenUV(self):
        """
        Gets the number of cvs in U and V
        :return: [int, int]
        """
        return [self.lenU(), self.lenV()]

    def lenU(self):
        """
        Gets the number of cvs in U
        :return: int
        """
        return self.MFn.numCVsInU

    def lenV(self):
        """
        Gets the number of cvs in V
        :return: int
        """
        return self.MFn.numCVsInV

    def getOrientationAtParam(self, u, v, ws=True):
        # TODO: Add up, u, and v axis kwarg
        space = om.MSpace.kWorld if ws else om.MSpace.kObject

        normal = self.MFn.normal(u, v, space=space).normalize()
        tangentU, tangentV = self.MFn.tangents(u, v, space=space)
        tangentU, tangentV = tangentU.normalize(), tangentV.normalize()
        matrix = (
            list(tangentV)
            + [0.0]
            + list(normal)
            + [0.0]
            + list(tangentU)
            + [0.0, 0.0, 0.0, 0.0, 1.0]
        )
        mmatrix = om.MMatrix(matrix)
        trans_matrix = om.MTransformationMatrix(mmatrix)
        euler_radians = trans_matrix.rotation()
        euler_degrees = [math.degrees(angle) for angle in euler_radians]
        return euler_degrees

    def getPositionAtParam(self, u, v, ws=True):
        space = om.MSpace.kWorld if ws else om.MSpace.kObject
        return list(self.MFn.getPointAtParam(u, v, space=space))[:-1]


class Lattice(ControlPoint):
    def __len__(self):
        """
        Returns the number of points in the lattice
        :return:
        """
        x, y, z = self.lenXYZ()
        return x * y * z

    def lenXYZ(self):
        """
        Returns the number of points in X, Y and Z
        :return: [int, int, int]
        """
        return cmds.lattice(self.name, divisions=True, q=True)

    def lenX(self):
        """
        Returns the number of points in X
        :return: int
        """
        return self.lenXYZ()[0]

    def lenY(self):
        """
        Returns the number of points in Y
        :return: int
        """
        return self.lenXYZ()[1]

    def lenZ(self):
        """
        Returns the number of points in Z
        :return: int
        """
        return self.lenXYZ()[2]


class GeometryFilter(DependNode):
    _MFN_FUNC = oma.MFnGeometryFilter

    @property
    def geometry(self):
        geo = cmds.deformer(self.name, q=True, geometry=True)
        return yam(geo[0]) if geo else None

    def getComponentAtIndex(self, index=0):
        """
        Gets the OpenMaya components influenced by the deformer at the given index.
        :return: om.MObject
        """
        return self.MFn.getComponentAtIndex(index)


class WeightGeometryFilter(GeometryFilter):
    def getWeights(self, force_clamp=True, min_value=0.0, max_value=1.0, round_value=None):
        geometry = self.geometry
        if not geometry:
            raise RuntimeError(f"Deformer '{self}' is not connected to a geometry")
        weightsAttr = self.weightsAttr
        return weightlist.WeightList(
            (weightsAttr[x].value for x in range(len(geometry))),
            force_clamp=force_clamp,
            min_value=min_value,
            max_value=max_value,
            round_value=round_value,
        )

    def setWeights(self, weights):
        weightsAttr = self.weightsAttr
        for i, weight in enumerate(weights):
            weightsAttr[i].value = weight

    @property
    def weights(self):
        return self.getWeights()

    @weights.setter
    def weights(self, weights):
        self.setWeights(weights)

    @property
    def weightsAttr(self):
        """
        Gets easy access to the standard deformer weight attribute.
        :return: Attribute
        """
        return self.weightList[0].weights


class Cluster(WeightGeometryFilter):
    def localize(self):
        """
        Adds a 'root' node as parent of the cluster. The root can be moved around without the cluster having an
        influence on the deformed shape.
        :return: the new root transform of the cluster.
        """
        handle_shape = self.handleShape
        if not handle_shape:
            raise RuntimeError(f"No clusterHandle found connected to {self.name}")

        root_grp = createNode(
            "transform", name=self.shortName + "_clusterRoot", parent=handle_shape.parent.parent
        )
        cluster_grp = createNode("transform", name=self.shortName + "_cluster", parent=root_grp)

        cluster_grp.worldMatrix.connectTo(self.matrix, force=True)
        cluster_grp.matrix.connectTo(self.weightedMatrix, force=True)
        cluster_grp.parentInverseMatrix.connectTo(self.bindPreMatrix, force=True)
        cluster_grp.parentMatrix.connectTo(self.preMatrix, force=True)
        self.clusterXforms.breakConnection()
        cmds.delete(handle_shape.parent.name)
        return root_grp

    @property
    def handleShape(self):
        return self.clusterXforms.source.node


class SoftMod(WeightGeometryFilter):
    def localize(self):
        """
        Adds a 'root' node as parent of the softMod. The root can be moved around to change from where the softMod
        starts its influence.
        :return: the new root transform of the softMod.
        """
        handle_shape = self.handleShape
        if not handle_shape:
            raise RuntimeError(f"No softModHandle found connected to {self.name}")

        root_grp = createNode(
            "transform", name=self.shortName + "_softModRoot", parent=handle_shape.parent.parent
        )
        softMod_grp = createNode("transform", name=self.shortName + "_softMod", parent=root_grp)
        falloffRadius = softMod_grp.addAttr(
            "falloffRadius",
            attributeType="double",
            keyable=True,
            hasMinValue=True,
            minValue=0.0,
            defaultValue=self.falloffRadius.value,
        )
        falloffRadius.connectTo(self.falloffRadius, force=True)

        softMod_grp.matrix.connectTo(self.weightedMatrix, force=True)
        softMod_grp.parentInverseMatrix.connectTo(self.bindPreMatrix, force=True)
        softMod_grp.parentMatrix.connectTo(self.preMatrix, force=True)
        softMod_grp.worldMatrix.connectTo(self.matrix, force=True)
        dmx = createNode("decomposeMatrix", name=self.shortName + "_DMX")
        softMod_grp.parentMatrix.connectTo(dmx.inputMatrix)
        dmx.outputTranslate.connectTo(self.falloffCenter, force=True)

        self.softModXforms.breakConnection()
        cmds.delete(handle_shape.parent.name)
        return root_grp

    @property
    def handleShape(self):
        return self.softModXforms.source.node


class SkinCluster(GeometryFilter):
    _MFN_FUNC = oma.MFnSkinCluster

    @classmethod
    def create(
        cls,
        geometry: "str | CpnNode",
        influences: "[str | Joint, ...]",
        name: str = None,
        createBindPose: bool = True,
        weights: "[[float, ...], int] | None" = None,
        setDefaultWeights: bool = False,
        lockGeometryTRS: bool = True,
    ) -> "SkinCluster":
        """
        Creates a skinCluster node on the given mesh with the given influences.

        Warnings:
            If no weights are given and setDefaultWeights is set to False: maya will fatal error if you try to paint
            skin weights using the paint skin weights tool.

        Notes:
            When creating a bindPose manually and in order for it to work properly, like it would when created via
            cmds.skinCluster, the bindPose attribute "xformMatrix" must be set with the local matrix value of each
            binPose members. This attribute is hidden, does not show when using the node editor "Show All Attributes" or
            in the connection editor, and was a pain in the a** to find.

        Args:
            geometry (str|CpnNode): The geometry to deform with the created skinCluster.
            influences (list): A list of skinCluster influences.
            name (str): The name of the created skinCluster node.
            createBindPose (bool): If True: create a dagPose node and connects it to the skinCluster and influences.
            weights: (list|None): The weights for the skinCluster in the same format as returned by the .getWeights
                                  method.
            setDefaultWeights (bool): If True sets the weights to 1.0 for all vertices on the first influences. This
                                      prevents maya from getting a fatal error when trying to paint the weights when no
                                      weights are set on the skinCluster.
            lockGeometryTRS (bool): If True: locks the translate, rotate, and scale attributes of the given geometry
                                    transform.

        Returns:
            (SkinCluster): The initialized skinCluster object for the skinCluster node.
        """
        if isinstance(influences, (str, Yam)):
            influences = [influences]
        influences = yams(influences)

        # Creating the skinCluster with its connections to the geometry
        skinCluster = yam(cmds.deformer(str(geometry), type="skinCluster")[0])

        if lockGeometryTRS:
            geometry = yam(geometry)
            if geometry.isa("shape"):
                geometry = geometry.parent
            for trs in "trs":
                for xyz in "xyz":
                    geometry.attr(trs + xyz).lock()

        if name:
            skinCluster.name = name

        # Connecting the influences to the skinCluster
        for index, inf in enumerate(influences):
            inf.worldMatrix.connectTo(skinCluster.matrix[index])
            inf.objectColorRGB.connectTo(skinCluster.influenceColor[index])
            if not inf.hasattr("lockInfluenceWeights"):
                inf.addAttr(
                    "lockInfluenceWeights",
                    shortName="liw",
                    cachedInternally=True,
                    minValue=0,
                    maxValue=1,
                    attributeType="bool",
                )
            inf.lockInfluenceWeights.connectTo(skinCluster.lockWeights[index])

        # Creating the skinCluster bindPose
        if createBindPose:
            bindPose = createNode("dagPose", name="bindPose")
            bindPose.bindPose.value = True
            bindPose.message.connectTo(skinCluster.bindPose)

            # Connecting the joints and all their parents to the bindPose node in the same way the cmds.skinCluster
            # function would have connected them.
            members = {}
            for inf in influences:
                for node in inf.allParents() + [inf]:
                    if node in members:
                        continue

                    # Connecting and saving the bindPose members attribute.
                    member_attr = bindPose.members.nextAvailableElement()
                    member_index = member_attr.index
                    node.message.connectTo(member_attr)
                    members[node] = member_attr

                    # Connecting the corresponding bindPose parents attribute.
                    parent = node.parent
                    if parent:
                        members[parent].connectTo(bindPose.parents[member_index])

                    # Setting or connecting the node's world matrix to the bindPose worldMatrix attribute.
                    if node not in influences:
                        bindPose.worldMatrix[member_index].value = node.worldMatrix.value
                    else:
                        node.worldMatrix.connectTo(bindPose.worldMatrix[member_index])

                    # Setting the node's matrix to bindPose xformMatrix attribute, which is necessary for the bindPose
                    # reset maya tool to function properly.
                    bindPose.xformMatrix[member_index].value = node.matrix.value

        # Setting the given weights
        if weights:
            skinCluster.setWeights(weights)

        # Sets the weights to 1.0 for all vertices on the first influences. This prevents a fatal error if we try to
        # paint the weights when no weights are set on the skinCluster.
        elif setDefaultWeights:
            cmds.skinPercent(
                skinCluster.name,
                skinCluster.geometry.name,
                transformValue=[skinCluster.influences()[0].name, 1.0],
                zri=True,
            )

        # Resetting the skinCluster bindPreMatrix to the current joints positions
        skinCluster.reskin()

        return skinCluster

    def influences(self):
        return yams(self.MFn.influenceObjects())

    def getVertexWeight(self, index, influence_indexes=None):
        if influence_indexes is None:
            influence_indexes = self.getInfluenceIndexes()
        weights = weightlist.WeightList()
        weightsAttr = self.weightList[index].weights
        for i, jnt_index in enumerate(influence_indexes):
            weights.append(weightsAttr[jnt_index].value)
        return weights

    def setVertexWeight(self, index, values, influence_indexes=None):
        if influence_indexes is None:
            influence_indexes = self.getInfluenceIndexes()
        weightsAttr = self.weightList[index].weights
        for index, value in enumerate(values):
            weightsAttr[influence_indexes[index]].value = value

    def getInfluenceWeights(self, influence, geo=None, components=None):
        """
        Gets the weights of the given influence.
        :param influence: int or Joint object
        :param geo: om.MDagPath of the deformed geometry
        :param components: om.MObject of the deformed components
        :return: WeightList
        """
        if isinstance(influence, str):
            influence = yam(influence)
        if hasattr(influence, "isAYamNode"):
            influence = self.influences().index(influence)
        if not geo or not components:
            geo = self.geometry.MDagPath
            components = self.getComponentAtIndex()
        return weightlist.WeightList(self.MFn.getWeights(geo, components, influence))

    def setInfluenceWeights(self, influence, weights, geo=None, components=None):
        if not config.undoable:
            return self.setInfluenceWeightsOM(
                influence=influence, weights=weights, geo=geo, components=components
            )

        if not hasattr(influence, "isAYamNode"):
            influence = self.influences()[influence]

        attr_index = self.indexForInfluenceObject(influence)

        for i, weight in enumerate(weights):
            self.weightList[i].weights[attr_index].value = weight

    def setInfluenceWeightsOM(self, influence, weights, geo=None, components=None):
        if hasattr(influence, "isAYamNode"):
            inf_index = self.influences().index(influence)
        else:
            inf_index = influence
        if not geo or not components:
            geo = self.geometry.MDagPath
            components = self.getComponentAtIndex()
        self.MFn.setWeights(geo, components, om.MIntArray([inf_index]), om.MDoubleArray(weights))

    def getWeights(self, force_clamp=True, min_value=0.0, max_value=1.0, round_value=None):
        weights = []
        # Getting the weights
        weights_array, num_influences = self.MFn.getWeights(
            self.geometry.MDagPath, self.getComponentAtIndex()
        )

        for i in range(0, len(weights_array), num_influences):
            weights.append(
                weightlist.WeightList(
                    weights_array[i : i + num_influences],
                    force_clamp=force_clamp,
                    min_value=min_value,
                    max_value=max_value,
                    round_value=round_value,
                )
            )
        return weights

    def setWeights(self, weights):
        if config.undoable:
            # Hacking the undo queue ! Setting all weigths to 1.0 on first joint to register a change in the undo queue
            cmds.skinPercent(
                self.name,
                self.geometry.name,
                transformValue=[self.influences()[0].name, 1.0],
                zri=True,
            )
            # for vtx, weight in enumerate(weights):
            #     self.setVertexWeight(index=vtx, values=weight)
        # else:
        self.setWeightsOM(weights)

    def setWeightsOM(self, weights):
        # Getting an array of the influences indexes
        num_influences = len(self.influences())
        influence_array = om.MIntArray(range(num_influences))
        # Flattening the list of weights into a single list
        flat_weights = om.MDoubleArray([i[j] for i in weights for j in range(num_influences)])
        # Setting the weights
        self.MFn.setWeights(
            self.geometry.MDagPath, self.getComponentAtIndex(), influence_array, flat_weights
        )

    @property
    def weights(self):
        return self.getWeights()

    @weights.setter
    def weights(self, weights):
        self.setWeights(weights)

    def getDQWeigts(self, force_clamp=True, min_value=0.0, max_value=1.0, round_value=None):
        return weightlist.WeightList(
            self.MFn.getBlendWeights(self.geometry.MDagPath, self.getComponentAtIndex()),
            force_clamp=force_clamp,
            min_value=min_value,
            max_value=max_value,
            round_value=round_value,
        )

    def setDQWeights(self, weights):
        if config.undoable:
            weightsAttr = self.blendWeights
            for vtx, value in enumerate(weights):
                weightsAttr[vtx].value = value
        else:
            self.setDQWeightsOM(weights)

    def setDQWeightsOM(self, weights):
        # Converting the data into maya array
        weights = om.MDoubleArray(weights)
        # Setting the weights
        self.MFn.setBlendWeights(self.geometry.MDagPath, self.getComponentAtIndex(), weights)

    @property
    def dQWeights(self):
        return self.getDQWeigts()

    @dQWeights.setter
    def dQWeights(self, weights):
        self.setDQWeights(weights)

    def getPointsAffectedByInfluence(self, influence):
        """
        Returns the points affected by the given influence
        :param influence: DependNode
        :return: YamList of components
        """
        influence = yam(influence)
        if influence in self.influences():
            influence = influence.MDagPath
        else:
            if not isinstance(influence, DagNode):
                raise TypeError(
                    f"Influence should be of type 'DependNode' not '{type(influence).__name__}'"
                )
            raise RuntimeError("Influence not found in skin cluster")

        vtx_list = yams(self.MFn.getPointsAffectedByInfluence(influence)[0].getSelectionStrings())
        vtxs = YamList()
        for vtx in vtx_list:
            if hasattr(vtx, "isAYamComponent"):
                vtxs.append(vtx)
            else:
                for vtx_ in vtx:
                    vtxs.append(vtx_)
        return vtxs

    def reskin(self):
        """
        Resets the skinCluster and mesh to the current influences position.
        """
        for matrix_indexed in self.matrix:
            bpm_attr = self.bindPreMatrix[matrix_indexed.index]
            if not bpm_attr.isSettable():
                if config.verbose:
                    cmds.warning(f"{bpm_attr} is connected and can't be reset")
                continue
            bpm_attr.value = list(om.MMatrix(matrix_indexed.value).inverse())
        cmds.skinCluster(self.name, e=True, rbm=True)

    def indexForInfluenceObject(self, influence):
        """
        Returns the influence connection index on the skinCluster.
        :param influence: DagNode
        :return: int
        """
        return self.MFn.indexForInfluenceObject(influence.MDagPath)

    def getInfluenceIndexes(self):
        return [self.indexForInfluenceObject(inf) for inf in self.influences()]

    @staticmethod
    def convertWeightsDataToPerVertex(data):
        num_vtx = len(data[0])
        range_influence = range(len(data))
        new_data = []
        for vtx in range(num_vtx):
            weights = weightlist.WeightList()
            for inf in range_influence:
                weights.append(data[inf][vtx])
            new_data.append(weights)
        return new_data

    @staticmethod
    def convertWeightsDataToPerInfluence(data):
        range_vtx = range(len(data))
        num_influence = len(data[0])
        new_data = []
        for inf in range(num_influence):
            weights = weightlist.WeightList()
            for vtx in range_vtx:
                weights.append(data[vtx][inf])
            new_data.append(weights)
        return new_data

    @property
    def data(self):
        return {
            "influences": self.influences().names,
            "weights": self.weights,
            "dQWeights": self.dQWeights,
            "skinningMethod": self.skinningMethod.value,
            "maxInfluences": self.maxInfluences.value,
            "normalizeWeights": self.normalizeWeights.value,
            "maintainMaxInfluences": self.maintainMaxInfluences.value,
            "weightDistribution": self.weightDistribution.value,
        }

    @data.setter
    def data(self, data):
        missing = []
        to_add = []
        influences = self.influences()
        for influence in data["influences"]:
            if not checks.objExists(influence):
                missing.append(influence)
            elif influence not in influences:
                to_add.append(influence)
        if missing:
            raise RuntimeError(f"Influences not found in scene : {missing}")
        self.addInfluences(to_add)

        self.skinningMethod.value = data["skinningMethod"]
        self.maxInfluences.value = data["maxInfluences"]
        self.normalizeWeights.value = data["normalizeWeights"]
        self.maintainMaxInfluences.value = data["maintainMaxInfluences"]
        self.weightDistribution.value = data["weightDistribution"]
        self.weights = data["weights"]
        self.dQWeights = data["dQWeights"]

    def addInfluence(self, influence):
        if hasattr(influence, "isAYamNode"):
            influence = influence.name
        cmds.skinCluster(self.name, edit=True, addInfluence=influence, weight=0.0)

    def addInfluences(self, influences):
        for inf in influences:
            self.addInfluence(inf)


class BlendShape(WeightGeometryFilter):
    """
    Class to wrap cmds and OpenMaya functions to easily interact with a blendShape node.

    Note for targets and in-betweens :
      Assumes for a target at value 1.0, that the target deltas are connected on the corresponding inputTargetItem plug
      at the index 6000.
      And in-between shapes are connected on index : int(inBetween_value * 1000 + 5000).
      e.g.: shape at value 1.0 is connected to index 6000, in-between shape at value 0.42 is connected at index 5420,
      in-between shape at value -1.2 is connected at index 3800, etc...

      In-betweens only works down to a value -5.0 because : -5 * 1000 + 5000 == 0 which is index 0. At a value lower
      than -5, the blendShape will connect the first attempt to index 2147483647 and then fail on any other attempt.
      This happens because 2147483647 is the largest value that a signed 32-bit integer field can hold, which is why
      going below index 0 loops back to this number.
      Warning: if negative values are used for in-betweens the blendShape node will snap back to its 0.0 value shape if
      the target attribute goes below -5.0. It is recommended to lock the target attribute lower range if negative
      in-between values are used.
    """

    def __getitem__(self, item):
        return self.target(item)

    def __contains__(self, item):
        """
        Checks if the blendShape node contains a given target.
        Works with string name of the target or BlendShapeTarget object.

        Args:
            item: string or BlendShapeTarget

        Returns: bool, True if the given item is a target of the blendShape node.
        """
        import attributes

        if isinstance(item, attributes.BlendShapeTarget):
            item = item.attribute
        return item in [target.attribute for target in self.targets()]

    def targets(self):
        from .attributes import BlendShapeTarget

        return YamList(BlendShapeTarget(self.weight[x].MPlug, self) for x in self.targetIndices())

    def target(self, index):
        targets = self.targets()

        # checking for alias names per targets
        if isinstance(index, str):
            try:
                index = targets.getattrs("alias").index(index)
            except ValueError:
                raise ValueError(f"'{index}' not in blendShape target list.")

        try:
            return targets[index]
        except IndexError:
            raise IndexError(f"BlendShape target index out of range : {index}.")

    @property
    def weightsAttr(self):
        return self.inputTarget[0].baseWeights

    def targetIndices(self):
        return list(self.weight.MPlug.getExistingArrayAttributeIndices())

    def addTarget(self, target, index=None, topologyCheck=False):
        """
        Value for a new target has to be 1.0 otherwise the target attribute is not added to the blendShape which is
        needed to return a corresponding BlendShapeTarget attribute.
        """
        existing_target_indices = self.targetIndices() or [-1]
        if index is None:
            index = existing_target_indices[-1] + 1

        if index < 0:
            raise RuntimeError(
                f"Target index cannot be negative; got : {index} for target '{target}'."
            )

        while index in existing_target_indices:
            index += 1

        cmds.blendShape(
            self.name,
            e=True,
            t=(self.geometry.name, index, str(target), 1.0),
            topologyCheck=topologyCheck,
        )
        return self.target(self.targetIndices().index(index))

    def addEmptyTarget(self, name):
        name_warning = False
        if cmds.ls(name) and config.verbose:
            name_warning = True
        temp = duplicate(self.geometry)[0]
        temp.name = name
        if name_warning and config.verbose:
            cmds.warning(
                f"An object with the same given target name : {name}, exists in the scene. Target"
                f" name used instead is : {temp.shortName}."
            )

        target = self.addTarget(temp)
        cmds.delete(temp.name)
        return target

    def addInBetween(self, target, index, value):
        index = self.target(index).index
        cmds.blendShape(
            self.name, e=True, inBetween=True, t=(self.geometry.name, index, str(target), value)
        )

    def getDeltas(self):
        return {target.alias: target.getDeltas() for target in self.targets()}

    def setDeltas(self, deltas, addMissingTargets=True):
        current_targets_names = [target.attribute for target in self.targets()]
        for target_name, delta in deltas.items():
            if target_name in current_targets_names:
                self[target_name].setDeltas(delta)
            elif addMissingTargets:
                target = self.addEmptyTarget(target_name)
                target.setDeltas(delta)
            elif config.verbose:
                cmds.warning(
                    f"Target {target_name} is missing from {self.name} and was not applied."
                )

    @property
    def data(self):
        targets = self.getTargets()
        return {
            "targets": self.targets().getattrs("alias"),
            "weights": self.weights,
            "targetWeights": [target.weights for target in targets],
        }

    @data.setter
    def data(self, data):
        self.weights = data["weights"]
        for i, target in enumerate(self.targets()):
            target.weights = data["targetWeights"][i]


class UVPin(DependNode):
    @staticmethod
    def createUVPin(mesh, name=None):
        mesh = yam(mesh)
        if isinstance(mesh, Transform) and mesh.shape:
            mesh = mesh.shape
        if not isinstance(mesh, SurfaceShape):
            raise TypeError("Target object must be a surface shape.")

        if name is None:
            name = f"{mesh}_UVP"
        pin = createNode("uvPin", name=name)
        pin.geometry = mesh
        return pin

    @property
    def geometry(self):
        connection = self.deformedGeometry.input()
        if connection:
            return connection.node

    @geometry.setter
    def geometry(self, geo):
        geo = yam(geo)
        if isinstance(geo, Transform):
            geo = geo.shape
        if not isinstance(geo, SurfaceShape):
            raise ValueError(f"Geometry must be a SurfaceShape; got {geo} -> {type(geo).__name__}.")

        if isinstance(geo, Mesh):
            out_attr = geo.worldMesh
        elif isinstance(geo, NurbsSurface):
            out_attr = geo.worldSpace
        else:
            raise NotImplementedError(
                f"Geometry '{geo}' of type '{type(geo).__name__}' not implemented."
            )

        out_attr.connectTo(self.deformedGeometry, force=True)

    def connectTransform(self, target, coordinates, index=None):
        target = yam(target)
        if not isinstance(target, Transform):
            raise TypeError(f"Target must be a transform; got : {target}, {type(target).__name__}.")
        if not len(coordinates) == 2:
            raise ValueError(f"Invalid UV coordinates given; {coordinates}.")

        if index is None:
            index = len(self.coordinate)
        self.coordinate[index].coordinateU.value = coordinates[0]
        self.coordinate[index].coordinateV.value = coordinates[1]
        mmx = createNode("multMatrix", name=f"{self}_MMX")
        dmx = createNode("decomposeMatrix", name=f"{self}_DMX")
        self.outputMatrix[index].connectTo(mmx.matrixIn[0])
        target.parentInverseMatrix.connectTo(mmx.matrixIn[1])
        mmx.matrixSum.connectTo(dmx.inputMatrix)
        dmx.outputTranslate.connectTo(target.translate, force=True)
        dmx.outputRotate.connectTo(target.rotate, force=True)
        dmx.outputScale.connectTo(target.scale, force=True)
        dmx.outputShear.connectTo(target.shear, force=True)

    def attachToClosestPoint(self, target):
        target = yam(target)
        if not isinstance(target, Transform):
            raise TypeError(f"Target must be a transform; got : {target}, {type(target).__name__}.")
        if not self.geometry:
            raise RuntimeError("No geometry connected to the uvPin.")

        geo = self.geometry
        point = om.MPoint(target.getXform(t=True, ws=True))
        if isinstance(geo, Mesh):
            u, v, _ = geo.MFn.getUVAtPoint(point, space=om.MSpace.kWorld)
        elif isinstance(geo, NurbsSurface):
            point, _, _ = geo.MFn.closestPoint(
                point, space=om.MSpace.kObject
            )  # TODO : fails to do it world space
            u, v = geo.MFn.getParamAtPoint(point, True)
        else:
            raise NotImplementedError(
                f"Geometry '{geo}' of type '{type(geo).__name__}' not implemeted."
            )
        self.normalizedIsoParms.value = False
        self.connectTransform(target, [u, v])


# These contain MFn and MFnData type names per {id#: name, ...}, to be able to get the type name from its id#.
# Warning : id# for nodes, attribute, etc... of the same type, are not consistent and change between maya versions.
MFN_TYPE_NAMES = {value: key for key, value in om.MFn.__dict__.items() if isinstance(value, int)}
MFNDATA_TYPE_NAMES = {
    value: key for key, value in om.MFnData.__dict__.items() if isinstance(value, int)
}


class SupportedTypes:
    """
    Data of supported types of maya nodes.

    Any new node class should be added to the inheritance_tree dict to be able to get it from the yam method and to
    classes_MFn dict for faster type lookup.
    New node class should also be added to the classes_str dict to be
    compatible with getclass_cmds.

    For classes_MFn : {om.MFn.kNodeMFnType: class,} where om.MFn.kNodeMFnType is a valid corresponding value from om.MFn
    For classes_str : {'mayaType': class,} where 'mayaType' is the type you get when using cmds.nodeType('node').
    """

    # Inheritance tree for all defined classes and their MFn types relative to each others.
    inheritance_tree = {
        (DependNode, om.MFn.kDependencyNode): {
            (DagNode, om.MFn.kDagNode): {
                (Transform, om.MFn.kTransform): {
                    (Joint, om.MFn.kJoint): {},
                    (Constraint, om.MFn.kConstraint): {
                        (ParentConstraint, om.MFn.kParentConstraint): {},
                        (OrientConstraint, om.MFn.kOrientConstraint): {},
                        (PointConstraint, om.MFn.kPointConstraint): {},
                        (ScaleConstraint, om.MFn.kScaleConstraint): {},
                        (AimConstraint, om.MFn.kAimConstraint): {},
                    },
                },
                (Shape, om.MFn.kShape): {
                    (SurfaceShape, om.MFn.kSurface): {
                        (Mesh, om.MFn.kMesh): {},
                        (NurbsSurface, om.MFn.kNurbsSurface): {},
                    },
                    (NurbsCurve, om.MFn.kNurbsCurve): {},
                    (Lattice, om.MFn.kLattice): {},
                },
            },
            (GeometryFilter, om.MFn.kGeometryFilt): {
                (WeightGeometryFilter, om.MFn.kWeightGeometryFilt): {
                    (Cluster, om.MFn.kClusterFilter): {},
                    (BlendShape, om.MFn.kBlendShape): {},
                    (SoftMod, om.MFn.kSoftModFilter): {},
                },
                (SkinCluster, om.MFn.kSkinClusterFilter): {},
            },
            (UVPin, om.MFn.kUVPin): {},
        },
    }

    # dict of MFn id to assigned yam class
    classes_MFn = {
        om.MFn.kDependencyNode: DependNode,  # 4
        om.MFn.kDagNode: DagNode,  # 107
        om.MFn.kTransform: Transform,  # 110
        om.MFn.kJoint: Joint,  # 121
        om.MFn.kConstraint: Constraint,  # 932; 928 in Maya 2020 ?! ¯\_(ツ)_/¯
        om.MFn.kParentConstraint: ParentConstraint,  # 242
        om.MFn.kPointConstraint: PointConstraint,  # 240
        om.MFn.kOrientConstraint: OrientConstraint,  # 239
        om.MFn.kScaleConstraint: ScaleConstraint,  # 244
        om.MFn.kAimConstraint: AimConstraint,  # 111
        om.MFn.kShape: Shape,  # 248
        om.MFn.kCluster: Shape,  # 251
        om.MFn.kSurface: SurfaceShape,  # 293
        om.MFn.kMesh: Mesh,  # 296
        om.MFn.kNurbsCurve: NurbsCurve,  # 267
        om.MFn.kNurbsSurface: NurbsSurface,  # 294
        om.MFn.kLattice: Lattice,  # 279
        om.MFn.kLocator: Shape,  # 281
        om.MFn.kGeometryFilt: GeometryFilter,  # 334
        om.MFn.kWeightGeometryFilt: WeightGeometryFilter,  # 346
        om.MFn.kClusterFilter: Cluster,  # 347
        om.MFn.kSkinClusterFilter: SkinCluster,  # 682
        om.MFn.kBlendShape: BlendShape,  # 336
        om.MFn.kCamera: Shape,  # 250
        om.MFn.kSoftModFilter: SoftMod,  # 348
        om.MFn.kUVPin: UVPin,  # 990; 986 in Maya 2020 ?! ¯\_(ツ)_/¯
    }
    # Some of the most commonly used DependNode classes for faster assigned class lookup
    supported_dependNodes = {
        om.MFn.kMatrixMult,  # 393
        om.MFn.kDecomposeMatrix,  # 1131
        om.MFn.kComposeMatrix,  # 1132
        om.MFn.kControllerTag,  # 1124
        om.MFn.kMultDoubleLinear,  # 770
        om.MFn.kUnitConversion,  # 526
        om.MFn.kPickMatrix,  # 1134
        om.MFn.kPlusMinusAverage,  # 458
        om.MFn.kBlendColors,  # 31
        om.MFn.kChoice,  # 36
        om.MFn.kCondition,  # 37
        om.MFn.kRemapValue,  # 933
        om.MFn.kReverse,  # 465
        om.MFn.kAddDoubleLinear,  # 5
        om.MFn.kMultiplyDivide,  # 445
        om.MFn.kDistanceBetween,  # 322
        om.MFn.kPluginDependNode,  # 456
    }
    for i in supported_dependNodes:
        classes_MFn[i] = DependNode

    # dict of maya nodeType to assigned yam class
    classes_str = {
        "dagNode": DagNode,
        "transform": Transform,
        "joint": Joint,
        "constraint": Constraint,
        "parentConstraint": ParentConstraint,
        "pointConstraint": PointConstraint,
        "orientConstraint": OrientConstraint,
        "scaleConstraint": ScaleConstraint,
        "aimConstraint": AimConstraint,
        "shape": Shape,
        "surfaceShape": SurfaceShape,
        "controlPoint": ControlPoint,
        "mesh": Mesh,
        "nurbsCurve": NurbsCurve,
        "nurbsSurface": NurbsSurface,
        "lattice": Lattice,
        "locator": Shape,
        "camera": Shape,
        "geometryFilter": GeometryFilter,
        "weightGeometryFilter": WeightGeometryFilter,
        "cluster": Cluster,
        "skinCluster": SkinCluster,
        "blendShape": BlendShape,
        "softMod": SoftMod,
        "uvPin": UVPin,
    }
    # Some of the most commonly used DependNode classes for faster assigned class lookup
    dependNodes_str = {
        "multMatrix",
        "decomposeMatrix",
        "composeMatrix",
        "controller",
        "multDoubleLinear",
        "unitConversion",
        "pickMatrix",
        "plusMinusAverage",
        "blendColors",
        "choice",
        "condition",
        "remapValue",
        "reverse",
        "addDoubleLinear",
        "multiplyDivide",
        "distanceBetween",
    }
    for i in dependNodes_str:
        classes_str[i] = DependNode

    # set of all assignable yam classes
    all_classes = {DependNode} | set(classes_str.values())

    # Warning in case a new class was added to the classes_MFn dict and not the classes_str dict
    diff = set(classes_MFn.values()) - all_classes
    if diff:
        cmds.warning("#" * 82)
        cmds.warning(
            "There is a node class in classes_MFn dict that is not listed in classes_str :"
            f" '{diff}'"
        )
        cmds.warning("#" * 82)


class YamList(list):
    """
    A list like class to contain any Yam objects and facilitate their handling.
    A YamList can only contain objects that inherits from the Yam class.

    Mainly used to call .names on the YamList to return a list of str object names that can be passed to cmds commands.
    Can be used to keep a type of node or remove a type of node from the list using .keepType and .popType
    """

    def __init__(self, items=(), no_init_check=False):
        super().__init__(items)
        self.no_check = False
        if not no_init_check:
            self._check_all()

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __str__(self):
        return f"{self.__class__.__name__}{self.names}"

    def __eq__(self, other):
        if isinstance(other, YamList):
            return str(list(self)) == str(list(other))
        return str(list(self)) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, item):
        if item.__class__ == slice:
            return YamList(super().__getitem__(item), no_init_check=True)
        return super().__getitem__(item)

    def _check(self, item):
        """
        Check that the given item is a Yam object and raises an error if it is not.
        :param item: object to check
        """
        if not self.no_check:
            if not hasattr(item, "isAYamObject"):
                raise TypeError(
                    f"YamList can only contain Yam objects. '{item}' is '{type(item).__name__}'."
                )

    def _check_all(self):
        """
        Check that all the items in the current object are Yam objects and raises an error if they are not.
        """
        if not self.no_check:
            for item in self:
                if not hasattr(item, "isAYamObject"):
                    raise TypeError(
                        f"YamList can only contain Yam objects. '{item}' is"
                        f" '{type(item).__name__}'."
                    )

    def append(self, item):
        if not self.no_check:
            self._check(item)
        super().append(item)

    def extend(self, items):
        if not self.no_check:
            if not isinstance(items, YamList):
                for item in items:
                    self._check(item)
        super().extend(items)

    def no_check_extend(self, items):
        """Extends the list with the given items but skips the type check of the items for efficiency."""
        no_check = self.no_check  # To keep the current status of no_check
        self.no_check = True
        self.extend(items)
        self.no_check = no_check

    def insert(self, index, item):
        self._check(item)
        super().insert(index, item)

    def sort(self, key=None, reverse=False):
        if key is None:

            def name(x):
                return x.name

            key = name
        super().sort(key=key, reverse=reverse)

    def attrs(self, attr):
        return YamList((x.attr(attr) for x in self), no_init_check=True)

    def values(self, attr=None):
        if attr:
            return [x.attr(attr).value for x in self]
        return [x.value for x in self]

    @property
    def names(self):
        return [x.name for x in self]

    def MObjects(self):
        return [x.MObject for x in self]

    def getattrs(self, attr, call=False):
        if call:
            attrs = [getattr(x, attr)() for x in self]
        else:
            attrs = [getattr(x, attr) for x in self]
        try:
            return YamList(attrs)
        except TypeError:
            return attrs

    def keepType(self, types):
        """
        Remove all nodes that are not of the given type.
        :param types: str or list, e.g.: 'joint' or ['blendShape', 'skinCluster']
        """
        for i, item in reversed(list(enumerate(self))):
            if not item.isa(types):
                self.pop(i)

    def popType(self, types):
        """
        Removes all nodes of given type from current object and returns them in a new YamList.
        :param types: str or list, e.g.: 'joint' or ['blendShape', 'skinCluster']
        :return: YamList of the removed nodes
        """
        popped = YamList()
        for i, item in reversed(list(enumerate(self))):
            if item.isa(types):
                popped.append(self.pop(i))
        return popped

    def copy(self):
        return YamList(self, no_init_check=True)


class Yum:
    """
    Lazy way to initialize an existing Yam node.

    Saves 3 characters by not having to type yam('nodeName') and the hassle of having to reach ( and ' which are very
    far on the keyboard and need the shift modifier.
    This expects that Yum was initialized at import and stored in variable yum

    Initialize it on a variable (usually already done at import of yama module)
    e.g. :
    >>> yum = Yum()
    And get node :
    >>> yum.nodeName
    """

    def __getattr__(self, item):
        return yam(item)
