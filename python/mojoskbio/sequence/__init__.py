from ._sequence import DNA, RNA, Protein, Sequence
from .distance import hamming, kmer_distance

__all__ = ["DNA", "RNA", "Protein", "Sequence", "hamming", "kmer_distance"]
