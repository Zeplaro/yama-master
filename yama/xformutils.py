# encoding: utf8

from maya import cmds
import maya.api.OpenMaya as om

from . import nodes, components, decorators, utils, checks


def align(objs=None, t=True, r=True):
    """
    Aligns and evenly spaces given or selected objects between the first and last objects.
    """
    if not objs:
        # Allows you to get components order of selection
        if cmds.selectPref(q=True, tso=True) == 0:
            cmds.selectPref(tso=True)
        objs = nodes.selected()
    if not objs and len(objs) > 2:
        raise ValueError("Not enough object selected.")
    objs = nodes.yams(objs)
    poses = [x.getPosition(ws=True) for x in objs]

    t_step, t_start, r_step, r_start = None, None, None, None
    if t:
        t_start, t_end = poses[0], poses[-1]
        t_step = [(t_end[x] - t_start[x]) / (len(objs) - 1.0) for x in range(3)]
    if r:
        r_start, r_end = [x.getXform(ro=True, ws=True) for x in (objs[0], objs[-1])]
        r_step = [(r_end[x] - r_start[x]) / (len(objs) - 1.0) for x in range(3)]

    for i, obj in enumerate(objs):
        if t:
            pos = [t_step[x] * i + t_start[x] for x in range(3)]
            obj.setPosition(pos, ws=True)
        if r:
            rot = [r_step[x] * i + r_start[x] for x in range(3)]
            obj.setXform(ro=rot, ws=True)

    if r and not t:
        for obj, pos in zip(objs, poses):
            obj.setPosition(pos, ws=True)


@decorators.mayaundo
@decorators.keepsel
def aimChain(
    objs=None,
    aimVector=(1, 0, 0),
    upVector=(0, 1, 0),
    worldUpType="scene",
    worldUpObject=None,
    worldUpVector=(0, 1, 0),
):
    """
    Aims the given or selected objects to each other in order. Last object gets the same orientation as the previous
    one.
    Args:
        objs: [str, ...]
        aimVector: [float, float, float]
        upVector: [float, float, float]
        worldUpType: str, One of 5 values : "scene", "object", "objectrotation", "vector", or "none"
        worldUpObject: str, Use for worldUpType "object" and "objectrotation"
        worldUpVector: [float, float, float], Use for worldUpType "scene" and "objectrotation"

    e.g.:
    >>> aimChain(objs='locator1', aimVector=(1, 0, 0), upVector=(0, 1, 0), worldUpType='scene', worldUpObject=None,
    >>> worldUpVector=(0, 1, 0))
    """
    if worldUpType == "object":
        if not worldUpObject:
            raise Exception("worldUpObject is required for worldUpType 'object'.")
        checks.objExists(worldUpObject, raiseError=True)
    if not objs:
        objs = nodes.selected()
    if not objs and len(objs) > 1:
        raise ValueError("Not enough object selected.")
    objs = nodes.yams(objs)
    poses = [x.getXform(t=True, ws=True) for x in objs]
    # Using try to make sure nulss transforms are deleted
    try:
        nulls = nodes.YamList(nodes.createNode("transform") for _ in objs)
        for null, pos in zip(nulls, poses):
            null.setXform(t=pos, ws=True)
        world_null = nodes.createNode("transform", name="world_null")

        for obj, null, pos in zip(objs[:-1], nulls[1:], poses):
            obj.setXform(t=pos, ws=True)
            if worldUpType == "scene":
                cmds.delete(
                    cmds.aimConstraint(
                        null.name,
                        obj.name,
                        aimVector=aimVector,
                        upVector=upVector,
                        worldUpType="objectrotation",
                        worldUpObject=world_null.name,
                        worldUpVector=worldUpVector,
                        mo=False,
                    )
                )
            elif worldUpType == "object":
                cmds.delete(
                    cmds.aimConstraint(
                        null.name,
                        obj.name,
                        aimVector=aimVector,
                        upVector=upVector,
                        worldUpType="object",
                        worldUpObject=worldUpObject,
                        mo=False,
                    )
                )
            elif worldUpType == "objectrotation":
                cmds.delete(
                    cmds.aimConstraint(
                        null.name,
                        obj.name,
                        aimVector=aimVector,
                        upVector=upVector,
                        worldUpType="objectrotation",
                        worldUpObject=worldUpObject,
                        worldUpVector=worldUpVector,
                        mo=False,
                    )
                )
            else:
                raise NotImplementedError

        objs[-1].setXform(t=poses[-1], ro=objs[-2].getXform(ro=True, ws=True), ws=True)

    finally:
        try:
            cmds.delete(nulls.names)
            cmds.delete(world_null.name)
        except NameError:
            pass


def match(source=None, targets=None, t=True, r=True, s=False, m=False, ws=True):
    """
    Match the position, rotation and scale of the first object to the following objects.
    :param source: str or None to get first selected object.
    :param targets: [str, ...] or None to get selected objects.
    :param t: bool, True to match translation.
    :param r: bool, True to match rotation.
    :param s: bool, True to match scale.
    :param m: bool, True to match matrix.
    :param ws:  bool, True to match in world space.
    """
    if isinstance(targets, (str, nodes.Yam)):
        targets = [targets]

    if not targets:
        targets = nodes.selected()
    if not source:
        source = targets.pop(0) if targets else None

    if not targets or not source:
        raise ValueError("Not enough object selected.")

    source = nodes.yam(source)
    targets = nodes.yams(targets)

    if isinstance(source, components.Component):
        r = False
        s = False

    # Prevent unnecessary getting and setting of t, r and s when matching matrix
    if m:
        t, r, s = False, False, False

    pos, rot, scale, matrix = None, None, None, None
    if t:
        pos = source.getPosition(ws=ws)
    if r:
        rot = source.getXform(ro=True, ws=ws)
    if s:
        scale = source.getXform(s=True, ws=ws)
    if m:
        matrix = source.getXform(m=True, ws=ws)

    for target in targets:
        if isinstance(target, nodes.Transform):
            if t:
                target.setXform(t=pos, ws=ws)
            if r:
                target.setXform(ro=rot, ws=ws)
            if s:
                target.setXform(s=scale, ws=ws)
            if m:
                target.setXform(m=matrix, ws=ws)
        elif isinstance(target, components.Component):
            if t:
                target.setPosition(pos, ws=ws)
        else:
            raise RuntimeError(f"Cannot match '{target}' of type '{type(target).__name__}'.")


@decorators.keepsel
def matchComponents(components=None, target=None, ws=False):
    """
    Match the position of the given/selected components to the target object.
    :param components: [str, ...] or None to get selected components.
    :param target: str, Name of the target object or None to get last selected object.
    :param ws: bool, True to match in world space.
    """
    if not components or not target:
        components = nodes.selected(fl=False)
        target = components.pop(-1)
    else:
        components = nodes.yams(components)
        target = nodes.yam(target)

    for comp in components:
        target.cp[comp.index].setPosition(comp.getPosition(ws=ws), ws=ws)


def getCenter(objs):
    objs = nodes.yams(objs)
    xs, ys, zs = [], [], []
    for obj in objs:
        x, y, z = obj.getPosition(ws=True)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    x = (min_x + max_x) / 2
    y = (min_y + max_y) / 2
    z = (min_z + max_z) / 2
    return [x, y, z]


def snapAlongCurve(objs=None, curve=None, reverse=False):
    """
    Snap a list of given objects along a given curve.
    If no objects or curve are not given : works on current selection with last selected being the curve.
    :param objs: list, A list of objects to be snapped along the curve.
    :param curve: nodes.NurbsCurve, A nurbs curve object along which the objects will be snapped.
    :param reverse: bool, If set to True, the objects will be snapped in reverse order. Default is False.
    """
    if not objs or not curve:
        objs = nodes.selected()
        if not objs:
            raise RuntimeError("No object given and object selected.")
        if len(objs) < 2:
            raise RuntimeError("Not enough object given or selected.")
        curve = objs.pop()

    if isinstance(curve, nodes.Transform):
        curve = curve.shape

    if not isinstance(curve, nodes.NurbsCurve):
        raise TypeError(f"No NurbsCurve found under given curve : {curve}.")

    if reverse:
        objs = objs[::-1]
    step = curve.MFn.findParamFromLength(curve.arclen()) / (len(objs) - 1.0)
    for i, obj in enumerate(objs):
        x, y, z, _ = curve.MFn.getPointAtParam(step * i, om.MSpace.kWorld)
        obj.setPosition([x, y, z], ws=True)


def mirrorPos(obj, table):
    for l_cp in table:
        l_pos = obj.cp[l_cp].getPosition()
        r_pos = utils.mulLists([l_pos, table.axis_mult])
        obj.cp[table[l_cp]].setPosition(r_pos)

    mid_mult = table.axis_mult[:]
    mid_mult["xyz".index(table.axis)] *= 0
    for mid in table.mids:
        pos = utils.mulLists([obj.cp[mid].getPosition(), mid_mult])
        obj.cp[mid].setPosition(pos)


def flipPos(obj, table, reverse_face_normal=True):
    for l_cp in table:
        l_pos = obj.cp[l_cp].getPosition()
        r_pos = obj.cp[table[l_cp]].getPosition()

        obj.cp[l_cp].setPosition(utils.mulLists([l_pos, table.axis_mult]))
        obj.cp[table[l_cp]].setPosition(utils.mulLists([r_pos, table.axis_mult]))

    for mid in table.mids:
        pos = obj.cp[mid].getPosition()
        pos = utils.mulLists([pos, table.axis_mult])
        obj.cp[mid].setPosition(pos)

    if reverse_face_normal:
        # Flipping the face normals
        cmds.polyNormal(obj.name, normalMode=3, constructionHistory=False)


def extractXYZ(neutral, pose, axis=("y", "xz"), ws=False):
    """
    Extracts the vertex position difference between two shapes per each given axis or axis combination.
    :param neutral: neutral shape.
    :param pose: posed shape.
    :param axis: the axis or combination of axis to extract from.
    :param ws: Space in which to query the positions.
    :return: dictionary containing list of vertex positions per extracted axis

    TODO: Add support for other than world axis
    """
    neutral = nodes.yam(neutral)
    pose = nodes.yam(pose)
    if not isinstance(neutral, nodes.Mesh) or not isinstance(pose, nodes.Mesh):
        raise TypeError(f"Given neutral and/or pose object are not mesh objects; {neutral}, {pose}")
    n_pose = neutral.vtx.getPositions(ws=ws)
    pose_pos = pose.vtx.getPositions(ws=ws)
    data = {x: [] for x in axis}
    for n_vtx_pos, p_vtx_pos in zip(n_pose, pose_pos):
        for i in axis:
            pos = []
            for xyz, n_value, p_value in zip("xyz", n_vtx_pos, p_vtx_pos):
                if xyz in i:
                    pos.append(p_value)
                else:
                    pos.append(n_value)
            data[i].append(pos)
    return data


@decorators.keepsel
def makePlanar(
    objs, firstPointIndex=0, secondPointIndex=-1, thirdPointIndex=1, aimObjsChain=True, **aimKwargs
):
    """
    Aligns the given objects on a three point plane defined by the first, last and objs[midIndex] given objects.
    All objects must be in a parented hierarchy and given in hierarchical order.
    Minimum three objects must be given.

    :param objs: list [str, ...] Objects to align on a plane.
    :param firstPointIndex: int, the index in the given object list for the first point of the plane.
    :param secondPointIndex: int, the index in the given object list for the second point of the plane.
    :param thirdPointIndex: int, the index in the given object list for the third point of the plane.
    :param aimObjsChain: bool, if True: aims the objects to each other in order.
    :aimKwargs: all kwargs passed to aimChain call if aimObjsChain is True.
    """
    if len(objs) < 3:
        raise ValueError(
            "Not enough object given. Minimum 3 object needed to align them on the same plane"
        )
    if not -len(objs) < firstPointIndex < len(objs) - 1:
        raise ValueError(
            f"Must be True : -len(objs) < firstPointIndex < len(objs)-1; "
            f"Given firstPointIndex : {firstPointIndex}."
        )
    if not -len(objs) < secondPointIndex < len(objs) - 1:
        raise ValueError(
            f"Must be True : -len(objs) < secondPointIndex < len(objs)-1; "
            f"Given secondPointIndex : {secondPointIndex}."
        )
    if not -len(objs) < thirdPointIndex < len(objs) - 1:
        raise ValueError(
            f"Must be True : -len(objs) < thirdPointIndex < len(objs)-1; "
            f"Given thirdPointIndex : {thirdPointIndex}."
        )

    # Getting positive indices for corresponding amount of objs.
    length = len(objs)
    firstPointIndex = slice(firstPointIndex, None).indices(length)[0]
    secondPointIndex = slice(secondPointIndex, None).indices(length)[0]
    thirdPointIndex = slice(thirdPointIndex, None).indices(length)[0]

    # Using try to make sure nulls transforms are deleted
    try:
        # Creating temporary transforms to work with.
        nulls = nodes.YamList()
        for obj in objs:
            null = nodes.createNode("transform", name=obj.shortName + "_TEMP_NULL")
            null.setXform(m=obj.getXform(m=True, ws=True), ws=True)
            nulls.append(null)

        # Creating aim transform and matching it to defined up object.
        aimNull = nodes.createNode("transform", name="AIM_NULL")
        aimNull.setXform(m=objs[thirdPointIndex].getXform(m=True, ws=True), ws=True)

        # Aiming the first object to the last object with aimNull for up.
        aimChain(
            [nulls[firstPointIndex], nulls[secondPointIndex]],
            worldUpType="object",
            worldUpObject=aimNull,
        )

        # Parenting all nulls to first null and setting z to 0.
        nulls_except_first = nulls[:firstPointIndex] + nulls[firstPointIndex + 1 :]
        cmds.parent(nulls_except_first, nulls[firstPointIndex])
        for null in nulls_except_first:
            null.tz.value = 0

        # Matching the new positions from nulls to objects.
        for obj, null in zip(objs, nulls):
            obj.setPosition(null.getPosition(ws=True), ws=True)

        # Aiming objects to each others
        if aimObjsChain:
            # Matching the aimNull to defined up object
            aimNull.setXform(ro=nulls[firstPointIndex].getXform(ro=True, ws=True), ws=True)
            # Moving the aimNull up by same amount as distance between first and last object
            dist = utils.distance(objs[0].getPosition(ws=True), objs[-1].getPosition(ws=True)) * 10
            cmds.move(dist, aimNull.name, objectSpace=True, relative=True, moveY=True)

            aimKwargs.setdefault("worldUpType", "object")
            aimKwargs.setdefault("worldUpObject", aimNull)
            aimChain(objs, **aimKwargs)

    finally:
        try:
            cmds.delete(nulls.names)
            cmds.delete(aimNull.name)
        except NameError:
            pass
