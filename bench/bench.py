from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import skbio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojoskbio as mojo  # noqa: E402
from mojoskbio.sequence.distance import hamming, kmer_distance  # noqa: E402
from skbio import Sequence as SkSequence  # noqa: E402
from skbio.diversity import alpha_diversity as sk_alpha_diversity  # noqa: E402
from skbio.diversity import beta_diversity as sk_beta_diversity  # noqa: E402
from skbio.sequence.distance import hamming as sk_hamming  # noqa: E402
from skbio.sequence.distance import kmer_distance as sk_kmer_distance  # noqa: E402


def time_best(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def dna(length, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(np.array(list("ACGT")), length))


def cases():
    left = dna(5_000_000, 0)
    right = dna(5_000_000, 1)
    mojo_left, mojo_right = mojo.Sequence(left), mojo.Sequence(right)
    sk_left, sk_right = SkSequence(left), SkSequence(right)
    yield (
        "hamming (5M bases)",
        lambda: hamming(mojo_left, mojo_right),
        lambda: sk_hamming(sk_left, sk_right),
    )

    left = dna(200_000, 2)
    right = dna(200_000, 3)
    mojo_left, mojo_right = mojo.Sequence(left), mojo.Sequence(right)
    sk_left, sk_right = SkSequence(left), SkSequence(right)
    yield (
        "kmer_distance (200k bases, k=21)",
        lambda: kmer_distance(mojo_left, mojo_right, 21),
        lambda: sk_kmer_distance(sk_left, sk_right, 21),
    )

    rng = np.random.default_rng(4)
    counts = rng.poisson(3, size=(25_000, 256))
    yield (
        "alpha_diversity shannon (25k x 256)",
        lambda: mojo.alpha_diversity("shannon", counts),
        lambda: sk_alpha_diversity("shannon", counts),
    )
    yield (
        "alpha_diversity observed (25k x 256)",
        lambda: mojo.alpha_diversity("observed_features", counts),
        lambda: sk_alpha_diversity("observed_features", counts),
    )

    counts = rng.poisson(3, size=(800, 256))
    yield (
        "beta_diversity braycurtis (800 x 256)",
        lambda: mojo.beta_diversity("braycurtis", counts),
        lambda: sk_beta_diversity("braycurtis", counts),
    )
    yield (
        "beta_diversity jaccard (800 x 256)",
        lambda: mojo.beta_diversity("jaccard", counts),
        lambda: sk_beta_diversity("jaccard", counts),
    )


def machine():
    model = platform.processor()
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main():
    print(f"Machine: {machine()}")
    print(f"NumPy: {np.__version__}; scikit-bio: {skbio.__version__}")
    print()
    print("| case | mojo-scikit-bio | scikit-bio | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_time = time_best(ours)
        upstream_time = time_best(upstream)
        ratio = upstream_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
