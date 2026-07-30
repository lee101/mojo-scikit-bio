from __future__ import annotations


class Sequence:
    def __init__(self, sequence):
        if isinstance(sequence, Sequence):
            sequence = str(sequence)
        if isinstance(sequence, bytes):
            sequence = sequence.decode("ascii")
        if not isinstance(sequence, str):
            raise TypeError("sequence must be a string or bytes")
        encoded = sequence.encode("ascii")
        self._string = sequence
        self._bytes = encoded

    def __str__(self) -> str:
        return self._string

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._string!r})"

    def __len__(self) -> int:
        return len(self._string)

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and str(self) == str(other)


class DNA(Sequence):
    pass


class RNA(Sequence):
    pass


class Protein(Sequence):
    pass
