import numpy as np
import pytest

from skbio import DNA as SkDNA
from skbio import Sequence as SkSequence
from skbio.sequence.distance import hamming as sk_hamming
from skbio.sequence.distance import kmer_distance as sk_kmer_distance

from mojoskbio import DNA, Protein, RNA, Sequence
from mojoskbio.sequence.distance import hamming, kmer_distance


@pytest.mark.parametrize("length", [0, 1, 7, 64, 65, 1003])
@pytest.mark.parametrize("proportion", [True, False])
def test_hamming_random_parity(length, proportion):
    rng = np.random.default_rng(length)
    a = "".join(rng.choice(list("ACGTN-"), length))
    b = "".join(rng.choice(list("ACGTN-"), length))
    actual = hamming(Sequence(a), Sequence(b), proportion=proportion)
    expected = sk_hamming(
        SkSequence(a), SkSequence(b), proportion=proportion
    )
    assert actual == pytest.approx(expected, nan_ok=True)


def test_hamming_validates_inputs():
    with pytest.raises(TypeError):
        hamming("ACG", "ACT")
    with pytest.raises(TypeError):
        hamming(Sequence("ACG"), DNA("ACG"))
    with pytest.raises(ValueError):
        hamming(Sequence("AC"), Sequence("ACT"))
    assert hamming(DNA("ACG"), DNA("ACT"), proportion=False) == 1.0


@pytest.mark.parametrize("sequence_type", [Sequence, DNA, RNA, Protein])
def test_all_advertised_sequence_containers(sequence_type):
    left = sequence_type("ACGT")
    right = sequence_type("AGGT")
    assert hamming(left, right, proportion=False) == 1.0
    assert kmer_distance(left, right, 2) == pytest.approx(
        sk_kmer_distance(SkSequence("ACGT"), SkSequence("AGGT"), 2)
    )


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("k", [1, 2, 3, 8, 17, 40])
@pytest.mark.parametrize(
    "left,right",
    [
        ("ATCGGCGAT", "GCAGATGTG"),
        ("AAAAAAAAAAAA", "AAAAAAAATAAA"),
        ("ACGT", ""),
        ("", ""),
        ("ACGTACGTACGT", "ACGTACGTACGT"),
    ],
)
def test_kmer_distance_parity(left, right, k, overlap):
    actual = kmer_distance(Sequence(left), Sequence(right), k, overlap=overlap)
    expected = sk_kmer_distance(
        SkSequence(left), SkSequence(right), k, overlap=overlap
    )
    assert actual == pytest.approx(expected, nan_ok=True)


def test_kmer_distance_random_long_sequence():
    rng = np.random.default_rng(42)
    left = "".join(rng.choice(list("ACGT"), 5000))
    right = "".join(rng.choice(list("ACGT"), 5000))
    assert kmer_distance(Sequence(left), Sequence(right), 21) == pytest.approx(
        sk_kmer_distance(SkSequence(left), SkSequence(right), 21)
    )


def test_kmer_distance_subclass_and_validation():
    assert kmer_distance(DNA("ACGT"), DNA("AGGT"), 2) == pytest.approx(
        sk_kmer_distance(SkDNA("ACGT"), SkDNA("AGGT"), 2)
    )
    with pytest.raises(ValueError):
        kmer_distance(Sequence("A"), Sequence("A"), 0)
    with pytest.raises(TypeError):
        kmer_distance(Sequence("A"), Sequence("A"), 1.5)
