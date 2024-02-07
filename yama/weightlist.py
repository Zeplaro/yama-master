# encoding: utf8

from . import io


class WeightList(list):
    def __init__(self, data=None, force_clamp=True, min_value=0.0, max_value=1.0, round_value=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.force_clamp = force_clamp
        self.round_value = round_value

        # Forces initial arg to go through __setitem__ to verify values are floats, clamped and rounded
        if data:
            for i in data:
                self.append(i)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __getitem__(self, item):
        if item.__class__ == slice:
            return self.emptyCopy(super().__getitem__(item))
        return super().__getitem__(item)

    @classmethod
    def fromLengthValue(
        cls, length, value=0.0, force_clamp=True, min_value=0.0, max_value=1.0, round_value=None
    ):
        if not isinstance(length, int):
            raise TypeError(
                "length must be int; got : {}, {}".format(length, type(length).__name__)
            )
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                "value must be float or string float; got : {}, {}".format(
                    value, type(value).__name__
                )
            )
        return cls(
            (value for _ in range(length)),
            force_clamp=force_clamp,
            min_value=min_value,
            max_value=max_value,
            round_value=round_value,
        )

    def _cleanValue(self, value):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                "Value must be float or string float; got : {}, {}".format(
                    value, type(value).__name__
                )
            )

        if self.force_clamp:
            value = max(min(value, self.max_value), self.min_value)
        if self.round_value is not None:
            value = round(value, self.round_value)

        return value

    def __setitem__(self, key, value):
        value = self._cleanValue(value)

        super().__setitem__(key, value)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        return self.emptyCopy(x + y for (x, y) in zip(self, other))

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        for i, y in enumerate(other):
            self[i] += y
        return self

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        return self.emptyCopy(x - y for (x, y) in zip(self, other))

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        for i, y in enumerate(other):
            self[i] -= y
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        return self.emptyCopy(x * y for (x, y) in zip(self, other))

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        for i, y in enumerate(other):
            self[i] *= y
        return self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        return self.emptyCopy(x / y for (x, y) in zip(self, other))

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other, force_clamp=False)
        for i, y in enumerate(other):
            self[i] /= y
        return self

    def __neg__(self):
        return self.emptyCopy(-x for x in self)

    def __invert__(self):
        return self.emptyCopy(1 - x for x in self)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = self.fromLengthValue(len(self), other)
        return super().__eq__(other)

    def __hash__(self):
        return hash(tuple(self))

    def get(self, k, value=None):
        try:
            return self[k]
        except IndexError:
            return value

    def append(self, value):
        value = self._cleanValue(value)
        super().append(value)

    def extend(self, other):
        for i in other:
            self.append(i)

    def insert(self, index, value):
        value = self._cleanValue(value)
        super().insert(index, value)

    def clamp(self, min_value=None, max_value=None):
        if min_value is None:
            min_value = self.min_value
        if max_value is None:
            max_value = self.max_value
        for i, value in enumerate(self):
            if not min_value < value < max_value:
                self[i] = max(min(value, max_value), min_value)

    def round(self, round_value=None):
        if round_value is None and self.round_value is not None:
            round_value = self.round_value
        else:
            raise RuntimeError("No round_value value provided or set on the object.")
        for i, value in enumerate(self):
            self[i] = round(value, round_value)

    def replaceWeights(self, weights):
        del self[:]
        for i in weights:
            self.append(i)

    def exportWeights(self, path):
        io.exportData(self, path)

    def importWeights(self, path, apply_data=True):
        data = io.importData(path)
        if apply_data:
            del self[:]
            for i in data:
                self.append(i)
        return data

    def emptyCopy(self, data=None):
        return type(self)(
            data,
            force_clamp=self.force_clamp,
            min_value=self.min_value,
            max_value=self.max_value,
            round_value=self.round_value,
        )

    def copy(self):
        return self.emptyCopy(self)


def normalizeWeights(weights, force_clamp=True, min_value=0.0, max_value=1.0, round_value=None):
    """
    Normalizes a list of weights.

    Args:
        weights (list): A list of weights. Each weight can be a WeightList object or a numeric value.
        force_clamp (bool, optional): True to force clamp the normalized values within the specified range. Default True
        min_value (float, optional): The minimum value for clamping. Defaults to 0.0.
        max_value (float, optional): The maximum value for clamping. Defaults to 1.0.
        round_value (int, optional): The number of decimal places to round the normalized values. Defaults to None.

    Returns:
        list: A list of normalized weights, where each weight is a WeightList object.

    Note:
        - If a weight in the 'weights' list is already a WeightList object, it will be used as is.
        - If 'force_clamp' is set to True, the normalized values will be clamped between 'min_value' and 'max_value'.
        - If 'round_value' is specified, the normalized values will be rounded to the specified decimal places.
    """
    # Checking that all given weights are WeightList
    weights = [
        (
            WeightList(
                weight,
                force_clamp=force_clamp,
                min_value=min_value,
                max_value=max_value,
                round_value=round_value,
            )
            if not isinstance(weight, WeightList)
            else weight
        )
        for weight in weights
    ]
    new_weights = [weight.emptyCopy() for weight in weights]
    for values in zip(*weights):
        mult = 1.0
        # Keeps everything to 0.0 if all values are at 0.0 and prevents a / by 0 error
        if any(values):
            try:
                mult = 1.0 / sum(values)
            except ZeroDivisionError as e:
                raise ZeroDivisionError(
                    f"The sum of values was equal to 0.0; Values : {values}; {e}"
                )
        for i, value in enumerate(values):
            new_weights[i].append(value * mult)
    return new_weights


def clampWeights(weights, min_value=0.0, max_value=1.0):
    if isinstance(weights, WeightList):
        round_value = weights.round_value
    else:
        round_value = None
    return WeightList(
        weights, force_clamp=True, min_value=min_value, max_value=max_value, round_value=round_value
    )


def roundWeights(weights, round_value=3):
    if isinstance(weights, WeightList):
        force_clamp = weights.force_clamp
        min_value = weights.min_value
        max_value = weights.max_value
    else:
        force_clamp, min_value, max_value = True, 0.0, 1.0
    return WeightList(
        weights,
        force_clamp=force_clamp,
        min_value=min_value,
        max_value=max_value,
        round_value=round_value,
    )
