# encoding: utf8

from copy import copy
from maya import cmds, mel

from . import nodes, decorators, components, config, weightlist, checks


def getSkinCluster(obj, firstOnly=True):
    # type: (nodes.Yam, bool) -> nodes.SkinCluster | nodes.YamList[nodes.SkinCluster] | None
    """TODO: which way is better : listHistory or ls ?"""
    skinClusters = cmds.listHistory(str(obj), pdo=True)
    skinClusters = cmds.ls(skinClusters, type="skinCluster")
    if firstOnly:
        return nodes.yam(skinClusters[0]) if skinClusters else None
    return nodes.yams(skinClusters) if skinClusters else nodes.YamList()


def getSkinClusters(objs, firstOnly=True):
    return nodes.YamList([getSkinCluster(obj, firstOnly=firstOnly) for obj in objs])


def skinAs(
    objs=None,
    sourceNamespace=None,
    targetNamespace=None,
    useObjectNamespace=False,
    createMissing=True,
    prompt=True,
):
    # type: ([str | nodes.Transform | nodes.Shape], str, str, bool, bool, bool) -> nodes.YamList[nodes.SkinCluster] | None
    """
    Copies the skinning and skinCLuster settings of one skinned object to any other objects with a different topology.
    First object given in objs is the skinned object to copy from and the rest being the objects to copy the skin to, if
    no objects are given then this works on current scene selection.

    If the first object influences have a namespace and the other objects influences have a different namespace or don't
    have one then you can use sourceNamespace to remove from the original influences, and/or targetNamespace to add to
    the new influences. To get the namespace from the list of objects then set useObjectNamespace to True, in this case
    the sourceNamespace and targetNamespace are ignored and the namespace to remove from the original influences and to
    add to the new influences is determined per each object namespace.
    :param objs: List of objects. The first object's skinCluster is copied to the other objects.
    :param sourceNamespace: Namespace of the joints skinning the source object.
    :param targetNamespace: Namespace to use to find the joints that will skin the target objects.
    :param useObjectNamespace: If True: uses each objects namespace to determine the source and target namespaces.
    :param createMissing: If True and useObjecNamespace or targetNamespace is True: will create any missing target
                          influences that match the source influences.
    :param prompt: If True: will show a dialogue box asking to delete the intermediate shapes before skinning if
                   intermediate shapes are found on the target objects
    """
    from . import mayautils as mut

    if isinstance(objs, str):
        raise RuntimeError(
            f"first arg objs='{objs}' is of type {type(objs).__name__} instead of expected type: list"
        )
    if not objs:
        objs = nodes.ls(sl=True, tr=True, fl=True)

        def getSkinnable(objs_):
            skinnable = []
            for obj in objs_:
                if obj.shapes(type="controlPoint") and not obj.shapes(type="nurbsSurface"):
                    skinnable.append(obj)
                skinnable.extend(getSkinnable(obj.children(type="transform")))
            return skinnable

        objs = [objs[0]] + getSkinnable(objs[1:])
    if len(objs) < 2:
        raise ValueError("Please select at least two skinnable objects")
    source = nodes.yam(objs[0])
    targets = mut.hierarchize(objs[1:], reverse=True)

    free_targets = []
    for target in targets:
        # Checking for already attached skin on target
        if getSkinCluster(target):
            if config.verbose:
                cmds.warning(f"{target} already has a skinCluster attached")
            continue
        else:
            free_targets.append(target)

        # Checking for intermediate shapes on target
        intermediate_shapes = [
            shape for shape in target.shapes(noIntermediate=False) if shape.intermediateObject.value
        ]
        if intermediate_shapes and prompt:
            result = cmds.confirmDialog(
                title="Confirm",
                message=f"These targets already have intermediate shapes on them : "
                f"{intermediate_shapes}\n"
                f"Do you want to continue ?",
                button=["Yes", "Cancel", "Delete them"],
                defaultButton="Yes",
                cancelButton="Cancel",
                dismissString="Cancel",
            )
            if result == "Cancel":
                cmds.warning("SkinAs operation cancelled")
                return
            elif result == "Delete them":
                cmds.delete(intermediate_shapes)
                return skinAs(objs, sourceNamespace, targetNamespace, useObjectNamespace)

    source_skn = getSkinCluster(source)
    if not source_skn:
        raise ValueError("First object as no skinCluster attached")
    source_influences = source_skn.influences()
    target_influences = source_influences

    if useObjectNamespace:
        sourceNamespace = ":".join(source.name.split(":")[:-1])
        if not sourceNamespace and config.verbose:
            cmds.warning("useObjectNamespace is True but no namespace was found on source object")
    elif sourceNamespace or targetNamespace:
        if sourceNamespace:
            replace_args = [sourceNamespace + ":", ""]
            if targetNamespace:
                replace_args[1] = targetNamespace + ":"
            target_influences = nodes.yams(
                inf.name.replace(*replace_args) for inf in source_influences
            )
        else:
            target_influences = nodes.yams(targetNamespace + ":" + inf for inf in source_influences)

    skinClusters = nodes.YamList()
    kwargs = {
        "skinMethod": source_skn.skinningMethod.value,
        "maximumInfluences": source_skn.maxInfluences.value,
        "normalizeWeights": source_skn.normalizeWeights.value,
        "obeyMaxInfluences": source_skn.maintainMaxInfluences.value,
        "weightDistribution": source_skn.weightDistribution.value,
        "includeHiddenSelections": True,
        "toSelectedBones": True,
    }
    for target in free_targets:
        if useObjectNamespace:  # getting target influences namespace per target
            targetNamespace = ":".join(target.name.split(":")[:-1])
            if sourceNamespace:
                replace_args = [sourceNamespace + ":", ""]
                if targetNamespace:
                    replace_args[1] = targetNamespace + ":"
                target_influences = []
                for inf in source_influences:
                    target_name = inf.name.replace(*replace_args)
                    if not checks.objExists(target_name) and createMissing:
                        target_inf = nodes.createNode("joint", name=target_name)
                    else:
                        target_inf = nodes.yam(target_name)
                    target_influences.append(target_inf)
            elif targetNamespace:
                target_influences = nodes.yams(
                    targetNamespace + ":" + inf for inf in source_influences
                )
            else:
                target_influences = source_influences

        nodes.select(target_influences, target)
        target_skinCluster = nodes.yam(
            cmds.skinCluster(name=f"{target.shortName}_SKN", **kwargs)[0]
        )
        cmds.copySkinWeights(
            ss=source_skn.name,
            ds=target_skinCluster.name,
            nm=True,
            sa="closestPoint",
            smooth=True,
            ia=("oneToOne", "label", "closestJoint"),
        )
        if config.verbose:
            print(target + " skinned -> " + target_skinCluster.name)
        skinClusters.append(target_skinCluster)
    return skinClusters


def reskin(objs=None):
    if not objs:
        objs = nodes.selected()
        if not objs:
            raise RuntimeError("No object given and no object selected")
    objs = nodes.yams(objs)
    for skn in getSkinClusters(objs):
        skn.reskin()


def mirrorWeights(weights, table):
    """
    Mirror weights according to the given mirror table.
    :param weights: dict of {'index': weight}
    :param table: mirror table from mayautils.getSymmetryTable
    :return: dict of {'index': weight}
    """
    mir_weights = copy(weights)
    for l_cp in table:
        mir_weights[table[l_cp]] = weights[l_cp]
    return mir_weights


def flipWeights(weights, table):
    flip_weights = copy(weights)
    for l_cp in table:
        flip_weights[l_cp] = weights[table[l_cp]]
        flip_weights[table[l_cp]] = weights[l_cp]
    return flip_weights


@decorators.keepsel
def copyDeformerWeights(source, target, sourceGeo=None, destinationGeo=None):
    """
    For two geometries with different topologies, copies a given deformer weight to another given deformer.
    :param source: the deformer to copy the weights from, the object needs to have a 'weights' attribute
    :param target: the deformer to copy the weights on, the object needs to have a 'weights' attribute
    :param sourceGeo: the geo on which the sourceDeformer is applied; if None, the geo is taken from the sourceDeformer
    :param destinationGeo: the geo on which the destinationDeformer is applied; if None, the geo is taken from the
                           destinationDeformer
    """
    source, target = nodes.yams((source, target))
    if sourceGeo is None:
        sourceGeo = source.geometry
    else:
        sourceGeo = nodes.yam(sourceGeo)
    if destinationGeo is None:
        destinationGeo = target.geometry
    else:
        destinationGeo = nodes.yam(destinationGeo)

    if not hasattr(source, "weights"):
        raise TypeError(
            f"'{source}' of type '{type(source).__name__}' has no 'weights' attributes."
        )
    if not hasattr(target, "weights"):
        raise TypeError(
            f"'{target}' of type '{type(target).__name__}' has no 'weights' attributes."
        )

    # Creating temp geos
    temp_source_geo, temp_dest_geo = nodes.yams(cmds.duplicate(sourceGeo.name, destinationGeo.name))
    temp_source_geo.name = "temp_source_geo"
    temp_dest_geo.name = "temp_dest_geo"
    # Removing shapeOrig
    cmds.delete(
        [x.name for x in temp_source_geo.shapes(noIntermediate=False) if x.intermediateObject.value]
    )
    cmds.delete(
        [x.name for x in temp_dest_geo.shapes(noIntermediate=False) if x.intermediateObject.value]
    )

    # Creating temp skinClusters to copy weights
    source_jnt = nodes.createNode("joint", name="temp_source_jnt")
    nodes.select(source_jnt, temp_source_geo)
    source_skn = nodes.yam(
        cmds.skinCluster(
            name="temp_source_skn",
            skinMethod=0,
            normalizeWeights=0,
            obeyMaxInfluences=False,
            weightDistribution=0,
            includeHiddenSelections=True,
            toSelectedBones=True,
        )[0]
    )
    source_skn.envelope.value = 0

    destination_jnt = nodes.createNode("joint", name="temp_destination_jnt")
    nodes.select(destination_jnt, temp_dest_geo)
    destination_skn = nodes.yam(
        cmds.skinCluster(
            name="temp_dest_skn",
            skinMethod=0,
            normalizeWeights=0,
            obeyMaxInfluences=False,
            weightDistribution=0,
            includeHiddenSelections=True,
            toSelectedBones=True,
        )[0]
    )
    destination_skn.envelope.value = 0

    source_weights = source.weights
    source_skn.weights = [[value] for value in source_weights]

    cmds.copySkinWeights(
        sourceSkin=source_skn.name,
        destinationSkin=destination_skn.name,
        noMirror=True,
        surfaceAssociation="closestPoint",
        influenceAssociation=("oneToOne", "label", "closestJoint"),
        smooth=True,
        normalize=False,
    )

    destination_skn_weights = destination_skn.weights
    target.weights = weightlist.WeightList([value[0] for value in destination_skn_weights])

    cmds.delete(source_skn.name, destination_skn.name)
    cmds.delete(source_jnt.name, destination_jnt.name)
    cmds.delete(temp_source_geo.name, temp_dest_geo.name)


@decorators.keepsel
def hammerShells(vertices=None, reverse=False):
    """
    Hammer weights on the selected or given vertices even if they're part of different shells or mesh.
    :param vertices: list of MeshVertex components. If None, uses selected vertices.
    :param reverse: False by default, hammer weights on the selected vertices. If True hammer weights on all other not
                    selected vertices per shell.
    """
    if not vertices:
        vertices = nodes.selected()
        vertices.keepType(components.Component)
        if not vertices:
            raise RuntimeError("No vertices given or selected")

    data = {}
    for vtx in vertices:
        node_name = vtx.node.name  # Uses name as dict key instead of node itself for performance.
        if node_name in data:
            data[node_name][1].append(vtx)
        else:
            data[node_name] = [vtx.node, [vtx]]

    for node_name, (node, vtxs) in data.items():
        # Uses indexOnly for 'index in/not in shell' for much faster performance
        shells = node.shells(indexOnly=True)
        if config.verbose:
            print(f"Found {len(shells)} shells for '{node_name}'")
        for shell in shells:
            if reverse:
                vtxs_indexes = {vtx.index for vtx in vtxs}
                shell_vtxs = [node.vtx[index] for index in shell if index not in vtxs_indexes]
                if len(shell) == len(shell_vtxs):
                    continue
            else:
                shell_vtxs = [vtx for vtx in vtxs if vtx.index in shell]
            if shell_vtxs:
                nodes.select(shell_vtxs)
                cmds.WeightHammer()


def transferInfluenceWeights(skinCluster, influences, target_influence, add_missing_target=True):
    """
    Transfer weights from a list of influences to a target influence.
    :param skinCluster: any type that can be passed to nodes.yam; the skinCluster to work on.
    :param influences: list; influences to transfer weights from.
    :param target_influence: any type that can be passed to nodes.yam; the influence to transfer weights to.
    :param add_missing_target: bool; if True, adds missing target influences to the skinCluster.
    """
    skinCluster = nodes.yam(skinCluster)
    influence_indexes = []
    all_influences = skinCluster.influences()

    # Checks the target influence is not missing from the skinCluster
    target_influence = nodes.yam(target_influence)
    if target_influence in all_influences:
        target_index = all_influences.index(target_influence)
    elif add_missing_target:
        cmds.skinCluster(
            skinCluster.name,
            e=True,
            addInfluence=target_influence.name,
            lockWeights=True,
            weight=0.0,
        )
        all_influences = skinCluster.influences()
        target_index = all_influences.index(target_influence)
    else:
        if config.verbose:
            cmds.warning(
                f"Target influence not found in skinCluster '{skinCluster}': '{target_influence}'"
            )
        return False

    # Checks the influences are not missing from the skinCluster
    missing_influences = []
    for inf in influences:
        inf = nodes.yam(inf)
        if inf in all_influences:
            influence_indexes.append(all_influences.index(inf))
        else:
            missing_influences.append(inf)
    if not influence_indexes:
        if config.verbose:
            cmds.warning(
                f"None of the influences were found in skinCluster '{skinCluster}': {influences}"
            )
        return False
    if missing_influences:
        if config.verbose:
            cmds.warning(
                f"Influences not found in skinCluster '{skinCluster}': {missing_influences}"
            )

    # Transferring the weights
    weights = skinCluster.weights
    for vtx in range(len(skinCluster.geometry)):
        for index in influence_indexes:
            weights[vtx][target_index] += weights[vtx][index]
            weights[vtx][index] = 0.0
    skinCluster.weights = weights

    """
    OLD best way without undo queue hacking for setting weights; must use skinCluster.indexForInfluenceObject to get the
    target and source indices if using the old way.
    """
    # if config.undoable:
    #     # If undoable, faster this way most times especially when the influences influence only part of the vertices.
    #     for inf, index in influence_indexes:
    #         for vtx in skinCluster.getPointsAffectedByInfluence(inf):
    #             skinCluster.weightList[vtx.index].weights[target_index].value += skinCluster.weightList[vtx.index].weights[index].value
    #             skinCluster.weightList[vtx.index].weights[index].value = 0.0
    # else:  # Faster this way if not undoable since OpenMaya has a way to set weights per influences
    #     target_weights = skinCluster.getInfluenceWeights(target_index)
    #     zero_weights = weightlist.WeightList.fromLengthValue(len(skinCluster.geometry), 0.0)
    #     for inf, index in influence_indexes:
    #         target_weights += skinCluster.getInfluenceWeights(index)
    #         skinCluster.setInfluenceWeights(index, zero_weights)
    #     skinCluster.setInfluenceWeights(target_index, target_weights)

    return True


def skinAsNurbs(source=None, targets=None):
    if not isinstance(targets, (list, tuple)):
        targets = [targets]
    if not source or not targets:
        source, *targets = nodes.selected()
    else:
        source, *targets = nodes.yams([source] + targets)

    if not source.shape:
        raise RuntimeError(f"Given source {source} does not have a shape.")
    no_shape = []
    no_nurbs = []
    for target in targets:
        if not target.shape:
            no_shape.append(target)
        if not isinstance(target.shape, nodes.NurbsSurface):
            no_nurbs.append(no_nurbs)
    if no_shape:
        raise RuntimeError(f"Given targets {targets} do not have a shape.")
    if no_nurbs:
        raise TypeError(f"Given targets {no_nurbs} do not contain a nurbsSurface.")

    for target in targets:
        plane, poly = nodes.yams(cmds.polyPlane(sx=target.shape.lenU(), sy=target.shape.lenV()))

        # Using a try except do be sure to delete the temp plane
        try:
            plane.shape.vtx.setPositions(target.shape.cv.getPositions(ws=True), ws=True)

            verbose = config.verbose
            config.verbose = False
            plane_skin = skinAs([source, plane])[0]
            config.verbose = verbose
            target_skn = getSkinCluster(target)
            if target_skn:
                raise RuntimeError(
                    f"Given target already has a skinCluster attached : {target}; {target_skn}."
                )

            target_skin = skinAs([source, target])
            target_skin = target_skin[0]

            for i, _ in enumerate(target.shape.cv):
                for j, _ in enumerate(plane_skin.influences()):
                    target_skin.weightList[i].weights[j].value = (
                        plane_skin.weightList[i].weights[j].value
                    )
        except Exception as e:
            raise e
        # Making sure the temp plane gets deleted
        finally:
            cmds.delete(plane.name)


def localizeSkinClusterInfluence(skinCluster, influence):
    """
    Localizing the influence of a skinCluster make the skinCluster only accounts for the local position change of the
    influence and does not take into account the position change of the influence parents.

    Here are three methods to localize the influence of a skinCluster:

    -First method : Create a transform located at the world origin, parent the new transform under the influence parent,
      then parent the influence under the new transform, finally change the connection between the influence and the
      skinCluster to have the .matrix (instead of its .worldMatrix) of the influence connect to the corresponding
      .matrix of the skinCluster.
      This way has the disadvantage of confusing maya about its current influences, because the modified .matrix inputs
      are not connected from a .worldMatrix output. Some cmds and even OpenMaya functions will not return the correct
      influences info anymore.
      e.g. : using cmds.skinCluster('skinCluster1', q=True, wi=True) to list weighted influences will only return the
      influences that still have their .worldMatrix connected to the skinCluster, and same thing when using
      oma.MFnSkinCluster.influenceObjects which will also only return the influences that still have their .worldMatrix
      connected.
      To 'reset' the skinCluster to the current influence positions : set the .bindPreMatrix to value of the influence
      .inverseMatrix.

    -Second method : Multiply (in that order) the .matrix of the influence by the original parent world matrix and plug
      the result directly into the corresponding .matrix input of the skinCluster.
      This way has the disadvantage of confusing maya about its current influences, because the modified .matrix inputs
      are not connected from a .worldMatrix output. Some cmds and even OpenMaya functions will not return the correct
      influences info anymore.
      e.g. : using cmds.skinCluster('skinCluster1', q=True, inf=True) to list influences will only return the influences
      that still have their .worldMatrix connected to the skinCluster, and same thing when using
      oma.MFnSkinCluster.influenceObjects which will also only return the influences that still have their .worldMatrix
      connected.
      To 'reset' the skinCluster to the current influence positions : set the original parent world matrix value to its
      current state and set the .bindPreMatrix to the inverse matrix value of the multiplication result.

    -Third method : Involves using the skinCluster .bindPreMatrix to localize the influence but keeping the .worldMatrix
      of the influence connected to the corresponding .matrix of the skinCluster. First multiply (in that order) the
      .matrix of the influence by the original parent world matrix (the values at bind pose not the plug), multiply
      (in that order) the original world inverse matrix (the values at bind pose not the plug) by the result of the
      first multiplication, then multiply (in that order) the result of the second multiplication by the
      .worldInverseMatrix of the influence, finally plug the result of the last multiplication into the corresponding
      .bindPreMatrix of the skinCluster.
      This way has the advantage of keeping the influences .worldMatrix directly plugged into the .matrix attribute of
      the skinCluster, which allows you to still use many of the cmds and OpenMaya functions without the unexpected
      behaviour that the other methods creates.
      To 'reset' the skinCluster to the current influence positions. Set the stored matrix values for the original
      parent world matrix and the original world inverse matrix to their current corresponding values for the influence.

    This function uses the third method.
    """
    skinCluster, influence = nodes.yams([skinCluster, influence])
    if influence not in skinCluster.influences():
        raise RuntimeError(
            f"Given influence '{influence}' is not an influence of skinCluster '{skinCluster}'"
        )

    # Getting the index at which the influence is connected to the skinCluster
    influence_index = skinCluster.indexForInfluenceObject(influence)

    # Getting the world matrix for the influence as if the parent had not moved in the world
    local_world_mult = nodes.createNode("multMatrix", name=f"{influence}_localWorld_MMX")
    influence.matrix.connectTo(local_world_mult.matrixIn[0])
    local_world_mult.matrixIn[1].value = influence.parentMatrix.value

    # Getting a matrix of the difference between the original world inverse matrix and the live world matrix if the
    # parent had not moved
    localWorldDiff_localWorldInverse_mult = nodes.createNode(
        "multMatrix", name=f"{influence}_wimWmDiff_MMX"
    )
    localWorldDiff_localWorldInverse_mult.matrixIn[0].value = influence.worldInverseMatrix.value
    local_world_mult.matrixSum.connectTo(localWorldDiff_localWorldInverse_mult.matrixIn[1])

    # Getting a matrix of the difference between the calculated live matrix difference and the current live world
    # inverse matrix
    worldInverse_diff_mult = nodes.createNode("multMatrix", name=f"{influence}_wimDiff_MMX")
    localWorldDiff_localWorldInverse_mult.matrixSum.connectTo(worldInverse_diff_mult.matrixIn[0])
    influence.worldInverseMatrix.connectTo(worldInverse_diff_mult.matrixIn[1])

    bindPreMatrix_attr = skinCluster.bindPreMatrix[influence_index]

    # Undo is often buggy and sets random value back to the bindPreMatrix plug if used, to avoid that we manually set
    # the values before doing the connection.
    if not bindPreMatrix_attr.input():
        bindPreMatrix_attr.value = bindPreMatrix_attr.value

    # Connecting the result to the influence
    worldInverse_diff_mult.matrixSum.connectTo(bindPreMatrix_attr, force=True)
