from __future__ import annotations

import math

import numpy as np

from .._lib import addr, lib
from ._sequence import Sequence


def _check_pair(seq1, seq2) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(seq1, Sequence) or not isinstance(seq2, Sequence):
        raise TypeError("seq1 and seq2 must be Sequence instances")
    if type(seq1) is not type(seq2):
        raise TypeError("seq1 and seq2 must have the same type")
    a = np.frombuffer(seq1._bytes, dtype=np.uint8)
    b = np.frombuffer(seq2._bytes, dtype=np.uint8)
    return a, b


def _nonempty_address(array: np.ndarray) -> tuple[np.ndarray, int]:
    if array.size:
        return array, addr(array, dtype=np.uint8)
    placeholder = np.zeros(1, dtype=np.uint8)
    return placeholder, addr(placeholder, dtype=np.uint8)


def hamming(seq1, seq2, proportion=True):
    a, b = _check_pair(seq1, seq2)
    if len(a) != len(b):
        raise ValueError("Sequences do not have equal length.")
    if len(a) == 0:
        return math.nan
    distance = int(
        lib().msb_hamming(
            addr(a, dtype=np.uint8), addr(b, dtype=np.uint8), len(a)
        )
    )
    return distance / len(a) if proportion else float(distance)


def kmer_distance(seq1, seq2, k, overlap=True):
    a, b = _check_pair(seq1, seq2)
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be greater than 0.")
    step = 1 if overlap else k
    candidates_a = max(0, (len(a) - k) // step + 1)
    candidates_b = max(0, (len(b) - k) // step + 1)
    candidates = candidates_a + candidates_b
    if candidates == 0:
        return math.nan
    capacity = 1
    while capacity < candidates * 2:
        capacity *= 2
    hashes = np.empty(capacity, dtype=np.uint64)
    starts = np.empty(capacity, dtype=np.int64)
    sources = np.empty(capacity, dtype=np.uint8)
    owners = np.zeros(capacity, dtype=np.uint8)
    a_owner, a_address = _nonempty_address(a)
    b_owner, b_address = _nonempty_address(b)
    return float(
        lib().msb_kmer_distance(
            a_address,
            len(a),
            b_address,
            len(b),
            k,
            step,
            addr(hashes, dtype=np.uint64, writable=True),
            addr(starts, dtype=np.int64, writable=True),
            addr(sources, dtype=np.uint8, writable=True),
            addr(owners, dtype=np.uint8, writable=True),
            capacity,
        )
    )
