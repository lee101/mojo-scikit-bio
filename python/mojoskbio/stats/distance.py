from __future__ import annotations

import numpy as np


class DistanceMatrix:
    def __init__(self, data, ids=None, validate=True, condensed=False):
        array = np.asarray(data, dtype=np.float64)
        if condensed or array.ndim == 1:
            length = len(array)
            size = int((1 + np.sqrt(1 + 8 * length)) / 2)
            if size * (size - 1) // 2 != length:
                raise ValueError("Invalid condensed distance matrix length.")
            square = np.zeros((size, size), dtype=np.float64)
            square[np.triu_indices(size, 1)] = array
            square += square.T
            array = square
        array = np.ascontiguousarray(array, dtype=np.float64)
        if validate:
            if array.ndim != 2 or array.shape[0] != array.shape[1]:
                raise ValueError("Data must be a square matrix.")
            if not np.allclose(array, array.T, equal_nan=True):
                raise ValueError("Data must be symmetric.")
            if not np.allclose(np.diag(array), 0.0):
                raise ValueError("Data must be hollow.")
        size = array.shape[0]
        if ids is None:
            ids = tuple(str(i) for i in range(size))
        else:
            ids = tuple(ids)
        if len(ids) != size:
            raise ValueError("The number of IDs must match the matrix size.")
        if len(set(ids)) != len(ids):
            raise ValueError("IDs must be unique.")
        self._data = array
        self._ids = ids
        self._id_index = {value: i for i, value in enumerate(ids)}

    @property
    def data(self):
        return self._data

    @property
    def ids(self):
        return self._ids

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return len(self._data)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 2:
            left, right = index
            if isinstance(left, (str, int)) and left in self._id_index:
                left = self._id_index[left]
            if isinstance(right, (str, int)) and right in self._id_index:
                right = self._id_index[right]
            return self._data[left, right]
        if isinstance(index, (str, int)) and index in self._id_index:
            return self._data[self._id_index[index]]
        return self._data[index]

    def condensed_form(self):
        return self._data[np.triu_indices(len(self), 1)].copy()

    def copy(self):
        return type(self)(self._data.copy(), self._ids, validate=False)

    def to_data_frame(self):
        import pandas as pd

        return pd.DataFrame(self._data.copy(), index=self._ids, columns=self._ids)

    def __eq__(self, other):
        return (
            isinstance(other, DistanceMatrix)
            and self.ids == other.ids
            and np.array_equal(self.data, other.data, equal_nan=True)
        )
