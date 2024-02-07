# encoding: utf8

import json


def exportData(data, path):
    """
    Exports given data in a given json path.
    :param data: any json valid data.
    :param path: full json file path.
    """
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def importData(path):
    """
    Imports data from a given json path.
    :param path: full json file path.
    :return: data from json file.
    """
    with open(path) as json_file:
        data = json.load(json_file)
    return data
