# mojo-scikit-bio

`mojo-scikit-bio` is a focused port of compute-heavy
[scikit-bio](https://scikit.bio/) sequence-distance and ecological-diversity
operations to Mojo. Its Python API mirrors the covered scikit-bio 0.7.3 names
and signatures, while all numerical kernels run in one compiled shared
library.

```python
from mojoskbio import Sequence, alpha_diversity, beta_diversity
from mojoskbio.sequence.distance import hamming, kmer_distance

left = Sequence("ATCGGCGAT")
right = Sequence("GCAGATGTG")

print(hamming(left, right))
print(kmer_distance(left, right, k=3))
print(alpha_diversity("shannon", [[3, 0, 4, 2]], ids=["sample-a"]))
print(beta_diversity("braycurtis", [[3, 0, 4], [1, 5, 2]],
                     ids=["sample-a", "sample-b"]).data)
```

## Coverage

The sequence API provides `Sequence`, `DNA`, `RNA`, and `Protein` containers
for the covered operations, plus:

| API | behavior |
| --- | --- |
| `sequence.distance.hamming(seq1, seq2, proportion=True)` | exact character distance, including gaps and ambiguous characters |
| `sequence.distance.kmer_distance(seq1, seq2, k, overlap=True)` | exact set-based k-mer distance; counts are intentionally ignored |

The directly callable alpha-diversity functions are:

`observed_features`, `sobs`, `singles`, `doubles`, `osd`, `chao1`,
`berger_parker_d`, `dominance`, `simpson_d`, `shannon`, `simpson`,
`inv_simpson`, `enspie`, `goods_coverage`, `margalef`, `menhinick`,
`mcintosh_d`, `mcintosh_e`, `pielou_e`, `simpson_e`, `heip_e`, `robbins`,
`hill`, `renyi`, and `tsallis`.

`alpha_diversity` applies every named metric above except the tuple-valued
`osd` helper to count matrices in one batched Mojo call and returns a
`pandas.Series`, matching scikit-bio. It also supports callable metrics.

`beta_diversity` supports:

| family | metrics |
| --- | --- |
| quantitative | `euclidean`, `sqeuclidean`, `cityblock`, `manhattan`, `braycurtis`, `canberra`, `chebyshev`, `correlation`, `cosine`, `hamming`, `minkowski`, `jensenshannon` |
| qualitative | `dice`, `jaccard`, `matching`, `rogerstanimoto`, `russellrao`, `sokalsneath`, `yule` |

It returns the standalone `DistanceMatrix` included here, with `.data`,
`.ids`, `.condensed_form()`, ID indexing, copying, and
`.to_data_frame()`. Callable metrics and a custom `pairwise_func` are also
accepted.

This is intentionally not a port of all scikit-bio. Sequence alignment,
metadata, trees, ordination, statistics, and I/O are outside scope. The alpha
metrics that use estimators, repeated sampling, confidence intervals, or
sorting are not included. Phylogenetic `faith_pd`, `phydiv`, weighted and
unweighted UniFrac are also not included because they require the tree layer.
The lightweight sequence classes only implement the surface needed by the
covered distance functions.

## Install

The repository carries its own pinned Mojo nightly through Pixi:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-scikit-bio.so`. The Python wrapper will
also rebuild a missing or stale library on first use. Set
`MOJOSK_BIO_LIB=/absolute/path/to/libmojo-scikit-bio.so` to use a prebuilt
library.

After `pixi install`, run the example at the top of this README with:

```bash
pixi run python - <<'PY'
from mojoskbio import Sequence, alpha_diversity, beta_diversity
from mojoskbio.sequence.distance import hamming, kmer_distance

left = Sequence("ATCGGCGAT")
right = Sequence("GCAGATGTG")

print(hamming(left, right))
print(kmer_distance(left, right, k=3))
print(alpha_diversity("shannon", [[3, 0, 4, 2]], ids=["sample-a"]))
print(beta_diversity("braycurtis", [[3, 0, 4], [1, 5, 2]],
                     ids=["sample-a", "sample-b"]).data)
PY
```

Run benchmarks only through the locked Pixi task:

```bash
pixi run bench
```

## Performance

These are real best-of-five results printed by `pixi run bench`, including
input validation, Python result construction, and scratch allocation. They
were measured on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic, Python 3.13.14, NumPy 2.5.1, and scikit-bio 0.7.3.

| case | mojo-scikit-bio | scikit-bio | result |
| --- | ---: | ---: | ---: |
| hamming (5M bases) | 0.99 ms | 9.55 ms | 9.67x faster |
| kmer_distance (200k bases, k=21) | 40.18 ms | 1287.49 ms | 32.04x faster |
| alpha_diversity shannon (25k x 256) | 119.80 ms | 532.16 ms | 4.44x faster |
| alpha_diversity observed (25k x 256) | 194.24 ms | 248.01 ms | 1.28x faster |
| beta_diversity braycurtis (800 x 256) | 16.05 ms | 94.82 ms | 5.91x faster |
| beta_diversity jaccard (800 x 256) | 102.09 ms | 279.09 ms | 2.73x faster |

The pairwise kernels parallelize independent sample rows, while named batched
alpha metrics avoid Python-per-sample dispatch. No GPU path is provided or
claimed.

## How it works

`src/kernels.mojo` is one compilation unit. Each exported function uses
`@export("name")` and the C ABI. NumPy buffers cross `ctypes` as integer
addresses because parameterized Mojo pointers cannot be exported. The Mojo
wrapper reconstructs `UnsafePointer[..., AnyOrigin[mut=True]]` values inside
the function.

Sequence data is contiguous ASCII `uint8`. Count tables and distance matrices
are C-contiguous, row-major `float64`. Python owns input, output, and scratch
memory; Mojo never retains a pointer or allocates an FFI-visible buffer. Alpha
metrics reduce each row in the compiled kernel. Beta metrics calculate the
upper triangle and mirror it into a dense symmetric matrix.

The k-mer kernel uses a caller-owned open-addressed hash table. Each occupied
slot stores a representative sequence position, and equal hashes are verified
byte-for-byte. Hash collisions therefore affect probing cost, not correctness.

The test suite contains 314 tests and parameterized parity cases against real
scikit-bio 0.7.3. Results using the nightly Mojo `log`, `exp`, or `pow`
implementations are checked with tight numerical tolerances.

## License

MIT
