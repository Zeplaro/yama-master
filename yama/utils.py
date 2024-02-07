# encoding: utf8

"""
Contains non maya specific utils.
"""

from string import ascii_lowercase
import math
from functools import reduce
from operator import mul


def mulLists(lists):
    """
    Zips given lists items and multiplies them together.
    :param lists: lists of list of multipliable items
    :return: list of multiplied items
    """
    # TODO: replace reduce(mul, ...) with python 3.8 math.prod when dropping python 2 support
    return [reduce(mul, x, 1) for x in zip(*lists)]


def addLists(lists):
    """
    Zips given lists items and adds them together.
    :param lists: lists of list of addable items
    :return: list of added items
    """
    return [sum(x) for x in zip(*lists)]


def decimalToAlphabetical(decimal):
    """
    Converts int to an alphabetical decimal.

    e.g.:
    >>> decimalToAlphabetical(0)
    'a'
    >>> decimalToAlphabetical(1)
    'b'
    >>> decimalToAlphabetical(2)
    'c'
    >>> decimalToAlphabetical(440414)
    'yama'

    :param decimal: int
    :return: str
    """
    if not isinstance(decimal, int):
        raise TypeError(f"Expected type -> int; got {decimal} -> {type(decimal).__name__}.")
    if decimal < 0:
        raise ValueError(f"Given decimal must be superior or equal to 0; got : {decimal}")

    alphanum = ""
    decimal += 1  # because alphabet hase no 0 and starts with 'a'
    while decimal:
        decimal -= 1  # 'a' needs to be used as next 'decimal' unit when reaching 'z':  ..., 'y', 'z', 'aa', 'ab', ...
        left = decimal % 26
        decimal = decimal // 26
        alphanum = ascii_lowercase[left] + alphanum
    return alphanum


def alphabeticalToDecimal(alpha):
    """
    Converts str to an int index.
    Converts all letter to lowercase. All letters must be from ascii_lowercase, no other character or number.

    e.g.:
    >>> alphabeticalToDecimal('a')
    0
    >>> alphabeticalToDecimal('b')
    1
    >>> alphabeticalToDecimal('c')
    2
    >>> alphabeticalToDecimal('yama')
    440414

    :param alpha: str
    :return: int
    """
    if not isinstance(alpha, str):
        raise TypeError(f"Expected type -> str; got {alpha} -> {type(alpha).__name__}.")
    if not alpha:
        raise ValueError("Cannot convert given empty string to decimal")

    alpha = alpha.lower()
    if not all(x in ascii_lowercase for x in alpha):
        raise ValueError(f"Expected ascii characters only; got {alpha}.")

    index = -1
    for step, letter in reversed(list(enumerate(reversed(alpha)))):
        letter_index = ascii_lowercase.index(letter)
        index += (letter_index + 1) * (26**step)
    return index


def decimalToRoman(decimal):
    """
    Converts int to roman numeral.
    Given decimal must be an integer superior to 0.
    """
    if not isinstance(decimal, int):
        raise TypeError(f"Expected type -> int; got {decimal} -> {type(decimal).__name__}.")
    if not decimal > 0:
        raise ValueError(f"Given index must be superior to 0; got : {decimal}.")

    roman = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    roman_num = ""
    while decimal:
        for value, num in roman:
            if decimal >= value:
                roman_num += num
                decimal -= value
                break
    return roman_num


def romanToDecimal(roman):
    """
    Converts roman numeral to int.
    Given roman numeral must be only contain valid roman letters : 'M', 'D', 'C', 'L', 'X', 'V', and 'I'.

    Warning : a string containing valid letters in an invalid order will still return a value even if the roman numeral
    is not valid; e.g. : 'CLVDXI' returns 666 but 666 is actually written 'DCLXVI'
    """
    if not isinstance(roman, str):
        raise TypeError(
            f"Expected type -> str, and not empty; got {roman} -> {type(roman).__name__}."
        )
    if not roman:
        raise ValueError("Cannot convert given empty string to roman numeral.")

    values = [
        ("M", 1000),
        ("CM", 900),
        ("D", 500),
        ("CD", 400),
        ("C", 100),
        ("XC", 90),
        ("L", 50),
        ("XL", 40),
        ("X", 10),
        ("IX", 9),
        ("V", 5),
        ("IV", 4),
        ("I", 1),
    ]
    roman = roman.upper()
    diff = set(roman) - set([x for x, _ in values])
    if diff:
        raise ValueError(f"Only valid roman character accepted; invalid characters : {diff}.")

    result = 0
    while roman:
        for sign, value in values:
            if roman.startswith(sign):
                result += value
                roman = roman[len(sign) :]
                break
    return result


def distance(vectorA, vectorB):
    # TODO: replace with python 3.8 math.dist when dropping python 2 support
    return math.sqrt(sum((x - y) ** 2.0 for x, y in zip(vectorA, vectorB)))


def getRegularPolygonCoordinates(sides=3, radius=0.5):
    angle = 0
    step = 2 * math.pi / sides

    coordinates = []
    for i in range(sides):
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        coordinates.append((x, y))
        angle -= step  # Substracting to have the face normal face up when used to build regular polygon in maya

    return coordinates


def recursive_map(func, iterable, *, recursiontypes=(tuple, list), forcerecursiontypes=False):
    """
    Recursive equivalent of map if an element of the iterable is an instance of given recursiontypes.

    @param func: the function to call with each element of the iterable as argument.
    @param iterable: the iterable to loop through to pass each element as argument to the given function.
    @param recursiontypes: list of types that trigger the recursion if the item type match any of these.

    @return: Return an iterator that applies function to every item of iterable, or its items if recursion.
    """
    for item in iterable:
        if isinstance(item, recursiontypes):

            if forcerecursiontypes:
                for type_ in recursiontypes:
                    if isinstance(item, type_):
                        yield type_(recursive_map(func, item))
                        break

            else:
                yield type(item)(recursive_map(func, item))
        else:
            yield func(item)
