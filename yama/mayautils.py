# encoding: utf8

from maya import cmds, mel
import maya.api.OpenMaya as om
from . import nodes, xformutils, decorators, config, utils


def createHook(node, parent=None, suffix="hook"):
    """
    Creates a transform that moves and behaves the same as the given node no mather its parenting.
    Usefull to separate parts of a rig that would otherwise be buried in the hierarchy.
    :param node: a transform node
    :param suffix: the suffix for the new hook name
    :param parent: the parent for the hook
    :return: the hook node
    """
    node = nodes.yam(node)
    if not isinstance(node, nodes.Transform):
        raise ValueError("Given node is not a transform")
    hook = nodes.createNode("transform", name=f"{node.shortName}_{suffix}")
    mmx = nodes.createNode("multMatrix", n=f"mmx_{node.shortName}_{suffix}")
    dmx = nodes.createNode("decomposeMatrix", n=f"dmx_{node.shortName}_{suffix}")
    node.worldMatrix[0].connectTo(mmx.matrixIn[1], f=True)
    hook.parentInverseMatrix[0].connectTo(mmx.matrixIn[2], f=True)
    mmx.matrixSum.connectTo(dmx.inputMatrix, f=True)
    dmx.outputShear.connectTo(hook.shear, f=True)
    dmx.outputTranslate.connectTo(hook.translate, f=True)
    dmx.outputScale.connectTo(hook.scale, f=True)
    dmx.outputRotate.connectTo(hook.rotate, f=True)
    hook.parent = parent
    return hook


def hierarchize(objs, reverse=False):
    """Returns a list of objects sorted by their hierarchy in the scene"""
    return nodes.YamList(sorted(objs, key=lambda x: x.longName, reverse=reverse))


def mxConstraint(source=None, target=None):
    if not source or not target:
        sel = nodes.selected(type="transform")
        if len(sel) != 2:
            raise RuntimeError(f"two 'transform' needed; {len(sel)} given")
        source, target = sel
    else:
        source, target = nodes.yams([source, target])

    mmx = nodes.createNode("multMatrix", n=f"{source.shortName}_mmx")
    dmx = nodes.createNode("decomposeMatrix", n=f"{source.shortName}_dmx")
    cmx = nodes.createNode("composeMatrix", n=f"{source.shortName}_cmx")
    cmx.outputMatrix.connectTo(mmx.matrixIn[0], f=True)
    source.worldMatrix.connectTo(mmx.matrixIn[1], f=True)
    target.parentInverseMatrix.connectTo(mmx.matrixIn[2], f=True)
    mmx.matrixSum.connectTo(dmx.inputMatrix, f=True)

    source_tmp = nodes.createNode("transform", n=f"{source.shortName}_sourceTMP")
    target_tmp = nodes.createNode("transform", n=f"{source.shortName}_targetTMP")
    xformutils.match([source, source_tmp])
    xformutils.match([target, target_tmp])
    target_tmp.parent = source_tmp

    cmx.inputTranslate.value = target_tmp.translate.value
    cmx.inputRotate.value = target_tmp.rotate.value
    cmx.inputScale.value = target_tmp.scale.value
    cmx.inputShear.value = target_tmp.shear.value

    dmx.outputTranslate.connectTo(target.translate, f=True)
    dmx.outputRotate.connectTo(target.rotate, f=True)
    dmx.outputScale.connectTo(target.scale, f=True)
    dmx.outputShear.connectTo(target.shear, f=True)

    cmds.delete(source_tmp.name, target_tmp.name)


def resetAttrs(objs=None, t=True, r=True, s=True, v=True, user=False, raiseErrors=True):
    """
    Resets the objs translate, rotate, scale, visibility and/or user defined attributes to their default values.
    :param objs: list of objects to reset the attributes on. If None then the selected objects are reset.
    :param t: if True resets the translate value to 0
    :param r: if True resets the rotate value to 0
    :param s: if True resets the scale value to 1
    :param v: if True resets the visibility value to 1
    :param user: if True resets the user attributes values to their respective default values.
    :param raiseErrors: If True, raises the encountered errors; skips them if False
    """
    if not objs:
        objs = nodes.selected(type="transform")
        if not objs:
            raise RuntimeError("No object given or transform selected")
    objs = nodes.yams(objs)

    tr = ""
    if t:
        tr += "t"
    if r:
        tr += "r"
    for obj in objs:
        for axe in "xyz":
            for tr_ in tr:
                attr = obj.attr(tr_ + axe)
                if attr.isSettable():
                    attr.value = 0
            if s:
                attr = obj.attr("s" + axe)
                if attr.isSettable():
                    attr.value = 1
        if v:
            attr = obj.v
            if attr.isSettable():
                attr.value = True
        if user:
            attrs = obj.listAttr(ud=True, scalar=True, visible=True)
            for attr in attrs:
                if not attr.isSettable():
                    continue
                try:
                    attr.value = attr.defaultValue
                except Exception as e:
                    if raiseErrors:
                        raise e
                    cmds.warning(f"Failed to set defaultValue on {attr} : {e}")


def insertGroup(obj, suffix="GRP"):
    if not obj:
        raise ValueError("No obj given; Use 'insertGroups' to work on selection")
    obj = nodes.yam(obj)
    grp = nodes.createNode("transform", name=f"{obj.shortName}_{suffix}")
    world_matrix = obj.getXform(m=True, ws=True)
    parent = obj.parent
    if parent:
        grp.parent = parent
    grp.setXform(m=world_matrix, ws=True)
    obj.parent = grp
    return grp


def insertGroups(objs=None, suffix="GRP"):
    if not objs:
        objs = nodes.selected(type="transform")
        if not objs:
            raise RuntimeError("No object given and no 'transform' selected")
    objs = nodes.yams(objs)
    grps = nodes.YamList()
    for obj in objs:
        grps.append(insertGroup(obj, suffix=suffix))
    return grps


def wrapMesh(objs=None, ws=True):
    """
    Wraps the meshes of all but the first object to the first object's mesh by finding the closest point.

    If no objects are passed, the function takes the selected objects. If no 'transform' or 'mesh' is selected, a
    'RuntimeError' will be raised.

    :param objs: (list of nodes) List of objects to be wrapped. Default is None.
    :param ws: (bool) If True, the matching is done in world space. Otherwise, the matching is done in object space.
    """
    if not objs:
        objs = nodes.selected(type=["transform", "mesh"])
        if not objs:
            raise RuntimeError("No object given and no 'transform' or 'mesh' selected")
        if len(objs) < 2:
            raise RuntimeError("Not enough object given or selected")
    objs = nodes.yams(objs)
    source = objs[0]
    targets = objs[1:]
    if isinstance(source, nodes.Transform):
        source = source.shape
        if not isinstance(source, nodes.Mesh):
            raise RuntimeError(f"First object '{source}' is not a 'mesh'")
    if ws:
        space = om.MSpace.kworld
    else:
        space = om.MSpace.kObject
    source_mfn = source.MFn
    for target in targets:
        if isinstance(target, nodes.Transform):
            target = target.shape
            if not isinstance(target, nodes.Mesh):
                cmds.warning(f"Cannot match '{target}' of type '{type(target).__name__}'")
                continue
        target_mfn = target.MFn
        for i in range(len(target)):
            x, y, z, _ = source_mfn.getClosestPoint(target_mfn.getPoint(i), space)[0]
            target.vtx[i].setPosition([x, y, z], ws=True)


def matrixMayaToRow(matrix):
    """Converts maya matrix (all values in a single list) to row matrix (list of list of values)"""
    return [matrix[0:4], matrix[4:8], matrix[8:12], matrix[12:16]]


def matrixRowToMaya(matrix):
    """Converts row matrix (list of list of values) to maya matrix (all values in a single list)"""
    maya_matrix = []
    for m in matrix:
        maya_matrix += m
    return maya_matrix


@decorators.keepsel
def getSymmetryTable(obj=None):
    """
    Returns a SymTable object containing the symmetry table of the given object.
    :param obj: the object to get the symmetry table from.
    :return: SymTable object
    """

    def selected():
        return cmds.ls(os=True, fl=True)

    def index(vtx):
        return int(vtx.split("[")[-1][:-1])

    if not obj:
        obj = selected()
        if not obj:
            raise RuntimeError("No mesh given")
        obj = obj[0]
    cmds.select(str(obj))
    table = SymTable()
    cmds.select(sys=1)
    r_vtxs = selected()
    for r_vtx in r_vtxs:
        cmds.select(r_vtx, sym=True)
        r_vtx_, l_vtx = selected()
        if not r_vtx == r_vtx_:
            r_vtx_, l_vtx = l_vtx, r_vtx_
        table[index(l_vtx)] = index(r_vtx)
    cmds.select(sys=2)
    l_vtxs = selected()
    for l_vtx in l_vtxs:
        cmds.select(l_vtx, sym=True)
        l_vtx_, r_vtx = selected()
        if not l_vtx == l_vtx_:
            l_vtx_, r_vtx = r_vtx, l_vtx_
        table[index(l_vtx)] = index(r_vtx)
    cmds.select(sys=0)
    mid_vtxs = selected()
    table.mids = [index(mid_vtx) for mid_vtx in mid_vtxs]
    return table


class SymTable(dict):
    """
    A class to represent a symmetry table. It is a dictionary with the keys being the left vertices index and the values
    being the right vertices index.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.axis_mult = [-1, 1, 1]
        self.mids = []

    def __repr__(self):
        return f"SymTable({super().__repr__()})"

    def __invert__(self):
        """
        Returns a flipped symmetry table.
        :return: the flipped symmetry table object
        """
        return SymTable({value: key for key, value in self.items()})

    def invert(self):
        """
        Flips the symmetry table.
        """
        inv = ~self
        self.clear()
        for key, value in inv.items():
            self[key] = value


def sortOutliner(objs=None, key=str):
    if not objs:
        objs = nodes.selected()
    grp = nodes.createNode("transform", name="grp_TEMP", ss=True)
    for i in sorted(objs, key=key):
        parent = i.parent
        grp.setXform(m=i.getXform(m=True, ws=True), ws=True)
        i.parent = grp
        i.parent = parent
    cmds.delete(grp.name)


def getActiveCamera():
    return nodes.yam(
        cmds.modelEditor(cmds.getPanel(withFocus=True), q=True, activeView=True, camera=True)
    ).shape


def unlockTRSV(
    objs=None, unlock=True, breakConnections=True, keyable=True, t=True, r=True, s=True, v=True
):
    if not objs:
        objs = nodes.selected()
    else:
        objs = nodes.yams(objs)

    trs = ""
    trs += "t" if t else ""
    trs += "r" if r else ""
    trs += "s" if s else ""

    for obj in objs:
        for attr in trs:
            if unlock:
                obj.attr(attr).locked = False
            if keyable:
                obj.attr(attr).keyable = True
            if breakConnections:
                obj.attr(attr).breakConnections()
            for xyz in "xyz":
                if unlock:
                    obj.attr(attr + xyz).locked = False
                if keyable:
                    obj.attr(attr + xyz).keyable = True
                if breakConnections:
                    obj.attr(attr + xyz).breakConnections()
        if v:
            if unlock:
                obj.v.locked = False
            if keyable:
                obj.v.keyable = True
            if breakConnections:
                obj.v.breakConnections()


def createPolyNgon(name="pNgon1", radius=0.1, sides=3, upAxis="y", parent=None):
    """
    Creates a single face regular polygon with given number of sides.

    @param name: str, name for the created polygon
    @param radius: flaot, the radius for the polygon.
    @param sides: int, the number of sides for the polygon.
    @param upAxis: 'x', 'y', or 'z', up axis to which the polygon face normal will point to.
    @param parent: the parent transform in which to put the created shape.

    @return: Mesh polygon shape node.
    """
    if sides < 3:
        raise ValueError(
            f"A polygon can not have less than 3 sides; numbr of sides given : {sides}"
        )

    if config.undoable:
        ngon = cmds.polyCone(
            radius=radius, subdivisionsX=sides, height=0, constructionHistory=False, name=name
        )[0]
        cmds.delete(f"{ngon}.f[1:{sides}]")
        cmds.polyNormal(ngon, normalMode=0, constructionHistory=False)
        ngon = nodes.yam(ngon).shape
        if parent:
            parent = nodes.yam(parent)
            ngon.parent = parent

    else:
        mfn = om.MFnMesh()
        coordinates = utils.getRegularPolygonCoordinates(sides, radius)
        coordinates.insert("xyz".index(upAxis), 0)
        u, v = zip(*coordinates)
        points = [om.MPoint(coordinate) for coordinate in coordinates]
        create_args = [points, [sides], list(range(sides)), u, v]
        if parent:
            parent = nodes.yam(parent)
            create_args.append(parent.MObject)
        ngon = mfn.create(*create_args)
        ngon = nodes.yam(ngon)
        ngon.name = name
        ngon = ngon.shape

    return ngon


def componentListToIndices(components):
    """
    Unpacks a list of components to a flat list of component indices.
    Maya's componentList attributes store components. Getting the value of this type of attributes returns a list of
    'packed' components, e.g.: ['vtx[0:333]', 'vtx[335:388]', 'vtx[390:38621]', 'vtx[38623:39242]', ...]
    """
    indices = []
    for pack in components:
        pack = pack.split("[")[-1][:-1]
        if ":" in pack:
            start, stop = pack.split(":")
            start, stop = int(start), int(stop)
            indices += list(range(start, stop + 1))
        else:
            indices.append(int(pack))
    return indices
