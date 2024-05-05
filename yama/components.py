# encoding: utf8

"""
Contains all the class and functions for maya components.
"""
# TODO : Improve getComponent func
# TODO : fix setPositionsOM on cvs ?
from abc import ABCMeta, abstractmethod

from maya import cmds
import maya.api.OpenMaya as om

from . import config, nodes, checks


def getComponent(node, attr):
    """
    Returns the proper component object for the given node, component name and index.
    :param node: DependNode, The node to get the component from.
    :param attr: str, The component name to get the component from.
    :return: The component object.
    """
    if "." in attr:
        raise TypeError(f"Not a supported component : '{attr}'")

    split = []
    if "[" in attr:  # Checking if getting a specific index
        split = attr.split("[")
        attr = split.pop(0)

    if attr not in SupportedTypes.TYPES:
        raise TypeError(f"component '{attr}' not in supported types")

    # Changing node to its shape if given a transform
    if isinstance(node, nodes.Transform):
        node = node.shape
        if not node:
            raise RuntimeError(f"node '{node}' has no shape to get component on")

    if attr == "cp":
        if node.__class__ in SupportedTypes.SHAPE_COMPONENT:
            attr = SupportedTypes.SHAPE_COMPONENT[node.__class__]

    # Getting api_type to get the proper class for the component
    if (attr, node.__class__) in SupportedTypes.COMPONENT_SHAPE_MFNID:
        api_type = SupportedTypes.COMPONENT_SHAPE_MFNID[(attr, node.__class__)]
    else:
        om_list = om.MSelectionList()
        om_list.add(f"{node.name}.{attr}[0]")
        dag, comp = om_list.getComponent(0)
        api_type = comp.apiType()

    if api_type not in SupportedTypes.MFNID_COMPONENT_CLASS:
        raise TypeError(f"component '{attr}' of api type '{api_type}' not in supported types")

    comp_class = SupportedTypes.MFNID_COMPONENT_CLASS[api_type][1]
    component = comp_class(node, api_type)
    indices = []
    for index in split:
        index = index[:-1]  # Removing the closing ']'
        if index in ["*", ":"]:  # if using the maya wildcard symbols
            indices.append(slice(None))
        elif ":" in index:  # if using a slice to list multiple components
            # parsing the string into a proper slice
            slice_args = [int(x) if x else None for x in index.split(":")]
            indices.append(slice(*slice_args))
        else:
            indices.append(int(index))
    while indices:
        component = component[indices.pop(0)]
    return component


class Components(nodes.Yam):
    """
    Base class for components not indexed.
    """

    __metaclass__ = ABCMeta

    def __init__(self, node, apiType):
        super().__init__()
        if not isinstance(node, nodes.ControlPoint):
            raise TypeError(
                f"Expected component node of type ControlPoint, instead got : {node},"
                f" {type(node).__name__}"
            )
        self.node = node
        self.api_type = apiType
        self.component_name = SupportedTypes.MFNID_COMPONENT_CLASS[apiType][0]
        self.component_class = SupportedTypes.MFNID_COMPONENT_CLASS[apiType][2]
        self._types = None

    def __getitem__(self, item):
        if item == "*":
            item = slice(None)
        if isinstance(item, slice):
            return ComponentsSlice(self.node, self, item)
        return self.index(item)

    def __len__(self):
        return len(self.node)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.node.name}', '{self.component_name}')"

    @abstractmethod
    def __iter__(self):
        pass

    def __eq__(self, other):
        if isinstance(other, Component):
            return hash(self) == hash(other)
        else:
            try:
                return self == nodes.yam(other)
            except (TypeError, checks.ObjExistsError):
                return False

    def __hash__(self):
        return hash((self.node.hashCode, self.api_type))

    @property
    def name(self):
        return f"{self.node}.{self.component_name}[:]"

    def index(self, index, secondIndex=None, thirdIndex=None):
        return self.component_class(self.node, self, index, secondIndex, thirdIndex)

    def getPositions(self, ws=False):
        return [x.getPosition(ws=ws) for x in self]

    def setPositions(self, values, ws=False):
        for x, value in zip(self, values):
            x.setPosition(value, ws=ws)

    def types(self):
        if self._types is None:
            self._types = ["components", self.api_type]
        return self._types

    def type(self):
        return self.types()[-1]


class SingleIndexed(Components):
    """
    Base class for components indexed by a single index.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self.index(i)


class MeshVertices(SingleIndexed):
    """
    Class for mesh vertices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getPositions(self, ws=False):
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        return [[p.x, p.y, p.z] for p in self.node.MFn.getPoints(space)]

    def setPositions(self, values, ws=False):
        if not config.undoable:
            self.setPositionsOM(values, ws)
        else:
            super().setPositions(values, ws)

    def setPositionsOM(self, values, ws=False):
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        self.node.MFn.setPoints([om.MPoint(x) for x in values], space)


class CurveCVs(SingleIndexed):
    """
    Class for curve cvs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getPositions(self, ws=False):
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        return [[p.x, p.y, p.z] for p in self.node.MFn.cvPositions(space)]

    def setPositions(self, values, ws=False):
        if not config.undoable:
            self.setPositionsOM(values, ws)
        else:
            super().setPositions(values, ws)

    def setPositionsOM(self, values, ws=False):
        # TODO: doesn't work ? Does but has a refresh issue ?
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        mps = [om.MPoint(x) for x in values]
        self.node.MFn.setCVPositions(mps, space)


class DoubleIndexed(Components):
    """
    Base class for components indexed by two indices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for u in range(self.node.lenU()):
            for v in range(self.node.lenV()):
                yield self.index(u, v)


class TripleIndexed(Components):
    """
    Base class for components indexed by three indices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        lenx, leny, lenz = self.node.lenXYZ()
        for x in range(lenx):
            for y in range(leny):
                for z in range(lenz):
                    yield self.index(x, y, z)


class Component(nodes.Yam):
    """
    Base class for indexed component.
    """

    def __init__(self, node, components, index, secondIndex=None, thirdIndex=None):
        super().__init__()
        if isinstance(node, nodes.Transform):
            node = node.shape
        self.node = node
        self.components = components
        self.index = index
        self.second_index = secondIndex
        self.third_index = thirdIndex
        self._types = None

    @property
    def isAYamComponent(self):
        """Used to check if an object is an instance of Component using the faster hasattr method instead of the slower
        isinstance method."""
        return True

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.third_index is not None:
            return (
                f"{self.__class__.__name__}('{self.node}', '{self.components.component_name}',"
                f" {self.index}, {self.second_index}, {self.third_index})"
            )
        elif self.second_index is not None:
            return (
                f"{self.__class__.__name__}('{self.node}', '{self.components.component_name}',"
                f" {self.index}, {self.second_index})"
            )
        else:
            return (
                f"{self.__class__.__name__}('{self.node}', '{self.components.component_name}',"
                f" {self.index})"
            )

    def __getitem__(self, item):
        """
        todo : work with slices
        """
        if not isinstance(item, int):
            raise TypeError("Expected item of type int, got : {item}, {type(item).__name__}")
        if isinstance(self.components, SingleIndexed):
            raise IndexError(f"'{self}' is a single index component and cannot get a second index")

        if self.second_index is None:
            return self.__class__(self.node, self.components, self.index, item)

        elif self.third_index is None:
            if isinstance(self.components, TripleIndexed):
                return self.__class__(
                    self.node, self.components, self.index, self.second_index, item
                )
            else:
                raise IndexError(
                    f"'{self}' is a double index component and cannot get a third index"
                )
        raise IndexError(f"'{self}' cannot get four index")

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.node.hashCode, self.type(), self.indices()))

    def exists(self):
        return checks.objExists(self)

    @property
    def name(self):
        return f"{self.node}.{self.attribute}"

    @property
    def attribute(self):
        attribute = f"{self.components.component_name}[{self.index}]"
        if self.second_index is not None:
            attribute += f"[{self.second_index}]"
        if self.third_index is not None:
            attribute += f"[{self.third_index}]"
        return attribute

    def indices(self):
        return self.index, self.second_index, self.third_index

    def getPosition(self, ws=False):
        return cmds.xform(self.name, q=True, t=True, ws=ws, os=not ws)

    def setPosition(self, value, ws=False):
        cmds.xform(self.name, t=value, ws=ws, os=not ws)

    def types(self):
        if self._types is None:
            self._types = ["component", self.components.api_type]
        return self._types

    def type(self):
        return self.types()[-1]


class MeshVertex(Component):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getPosition(self, ws=False):
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        p = self.node.MFn.getPoint(self.index, space)
        return [p.x, p.y, p.z]

    def setPosition(self, values, ws=False):
        if not config.undoable:
            self.setPositionOM(values, ws)
        else:
            super().setPosition(values, ws)

    def setPositionOM(self, value, ws=False):
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        point = om.MPoint(value)
        self.node.MFn.setPoint(self.index, point, space)


class CurveCV(Component):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getPosition(self, ws=False):
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        p = self.node.MFn.cvPosition(self.index, space)
        return [p.x, p.y, p.z]

    def setPosition(self, value, ws=False):
        if not config.undoable:
            self.setPositionOM(value, ws)
        else:
            super().setPosition(value, ws)

    def setPositionOM(self, value, ws=False):
        if ws:
            space = om.MSpace.kWorld
        else:
            space = om.MSpace.kObject
        point = om.MPoint(*value)
        self.node.MFn.setCVPosition(self.index, point, space)


class ComponentsSlice(nodes.Yam):
    """
    A slice object to work with maya slices.
    Warning : ComponentsSlice does contain the last index of the slice unlike in a Python slice.
    """

    def __init__(self, node, components, components_slice):
        super().__init__()
        self.node = node
        self.components = components
        self._slice = components_slice
        self._types = None

    @property
    def start(self):
        return self._slice.start or 0

    @property
    def stop(self):
        if self._slice.stop is None:
            stop = len(self.components)
        else:
            if self._slice.stop < 0:
                # If using negative number for stop, then using python behavior of stopping at len(components)-stop
                stop = len(self.components) + self._slice.stop - 1
            else:
                # Unlike Python, Maya slices includes the stop index, e.g.: mesh.vtx[2:12] <-- includes vertex #12
                stop = self._slice.stop + 1
        return stop

    @property
    def step(self):
        return self._slice.step or 1

    @property
    def slice(self):
        return slice(self.start, self.stop, self.step)

    @property
    def indices(self):
        return tuple(range(self.start, self.stop, self.step))

    def __str__(self):
        if self.step != 1:
            return str(self.names)
        return self.name

    def __repr__(self):
        return (
            f"{self.__class__.__name__}('{self.node}', '{self.components.component_name}',"
            f" {self.slice})"
        )

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise RuntimeError("cannot slice a ComponentsSlice object")
        return self.components.index(self.indices[item])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for i in self.indices:
            yield self.components.index(i)

    def __eq__(self, other):
        if isinstance(other, ComponentsSlice):
            return self.indices == other.indices
        return False

    def __hash__(self):
        return hash((self.node.hashCode, self.components.api_type, self.indices))

    @property
    def name(self):
        if self.step != 1:
            raise RuntimeError(
                "ComponentSlice that including a step different than 1 can not be represented by a "
                "single Maya valid name. Use .names instead"
            )
        return f"{self.node}.{self.components.component_name}[{self.start}:{self.stop}]"

    @property
    def names(self):
        return [cp.name for cp in self]

    @property
    def index(self):
        return self.slice

    def types(self):
        if self._types is None:
            self._types = ["components", self.components.api_type]

    def type(self):
        return self.types()[-1]

    def getPositions(self, ws=False):
        return [x.getPosition(ws=ws) for x in self]

    def setPositions(self, values, ws=False):
        for x, value in zip(self, values):
            x.setPosition(value, ws=ws)

    def getPosition(self, ws=False):
        return self.getPositions(ws=ws)

    def setPosition(self, value, ws=False):
        self.setPositions(value, ws=ws)


class SupportedTypes:
    """
    Contains all the supported component types and their corresponding class, MFn id, yam class, etc...
    """

    # Removed 'v' because rarely used and similar to short name for 'visibility'
    TYPES = {
        "cv",
        "e",
        "ep",
        "f",
        "map",
        "pt",
        "sf",
        "u",
        "#v",
        "vtx",
        "vtxFace",
        "cp",
    }

    MFNID_COMPONENT_CLASS = {
        om.MFn.kCurveCVComponent: ("cv", CurveCVs, CurveCV),  # 533
        om.MFn.kCurveEPComponent: ("ep", SingleIndexed, Component),  # 534
        om.MFn.kCurveParamComponent: ("u", SingleIndexed, Component),  # 536
        om.MFn.kIsoparmComponent: ("v", DoubleIndexed, Component),  # 537
        om.MFn.kSurfaceCVComponent: ("cv", DoubleIndexed, Component),  # 539
        om.MFn.kLatticeComponent: ("pt", TripleIndexed, Component),  # 543
        om.MFn.kMeshEdgeComponent: ("e", SingleIndexed, Component),  # 548
        om.MFn.kMeshPolygonComponent: ("f", SingleIndexed, Component),  # 549
        om.MFn.kMeshVertComponent: ("vtx", MeshVertices, MeshVertex),  # 551
        om.MFn.kCharacterMappingData: ("vtxFace", SingleIndexed, Component),  # 741
        om.MFn.kSurfaceFaceComponent: ("sf", DoubleIndexed, Component),  # 774
        om.MFn.kMeshMapComponent: ("map", SingleIndexed, Component),  # 813
    }

    COMPONENT_SHAPE_MFNID = {
        ("cv", nodes.NurbsCurve): om.MFn.kCurveCVComponent,
        ("cv", nodes.NurbsSurface): om.MFn.kSurfaceCVComponent,
        ("e", nodes.Mesh): om.MFn.kMeshEdgeComponent,
        ("ep", nodes.NurbsCurve): om.MFn.kCurveEPComponent,
        ("f", nodes.Mesh): om.MFn.kMeshPolygonComponent,
        ("map", nodes.Mesh): om.MFn.kMeshMapComponent,
        ("pt", nodes.Lattice): om.MFn.kLatticeComponent,
        ("sf", nodes.NurbsSurface): om.MFn.kSurfaceFaceComponent,
        ("u", nodes.NurbsCurve): om.MFn.kCurveParamComponent,
        ("u", nodes.NurbsSurface): om.MFn.kIsoparmComponent,
        ("v", nodes.NurbsSurface): om.MFn.kIsoparmComponent,
        ("vtx", nodes.Mesh): om.MFn.kMeshVertComponent,
        ("vtxFace", nodes.Mesh): om.MFn.kCharacterMappingData,
    }

    SHAPE_COMPONENT = {
        nodes.Mesh: "vtx",
        nodes.NurbsCurve: "cv",
        nodes.NurbsSurface: "cv",
        nodes.Lattice: "pt",
    }
