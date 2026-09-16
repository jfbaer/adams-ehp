# The unstable Adams E2-page

Computes the homology of the lambda algebra via the Curtis algorithm, and
the operations on the unstable Adams E2-page that are consumed by the python
pipeline: products, suspension
E, Hopf H, Whitehead product P, and Mahowald's map to Lambda(C2). The Curtis
procedure and Mahowald's map are described in the [preprint](../paper/lambda.pdf),
*The unstable Adams E₂-page*.

The Curtis algorithm reduces admissible monomials by tags; the map stage then
emits the products, the E/H/P maps, and the C2 map as `E2_*.csv` and
`E2_names.json`.

## Build, test, run

```bash
cargo build --release
cargo test --release
cargo run --release -- generate -d 70          # full pipeline, CSVs to cwd
cargo run --release -- curtis -d 70            # Curtis reduction only
cargo run --release -- c2 -d 70                # C2 map + rank only (fast)
cargo run --release -- c2-all -d 70            # C2 map + filtration-1 products
                                               #   + rank + names (the full
                                               #   Lambda(C2) data set)
cargo run --release -- c2-products -d 70       # products only (skips the map)
cargo run --release -- c2-names -d 70          # Lambda(C2) basis dictionary only
                                               #   (recover names from a dead run)
cargo run --release -- generate --help         # all flags
```

(Single-map slices `products`, `h`, and `p` also exist; see `--help`.)

To regenerate the full E2 dataset used by the python propagator and run it
end to end:

```bash
cargo run --release -- generate -d 77 -o ../python/data/E2
cd ../python && sage -python run.py 76
```

(A `-d N` table supports propagation through total degree `N - 1`, so
`-d 77` reproduces the shipped range; bare `run.py` defaults to 70, and the
loader reads the table's actual coverage and lowers its bound to fit.)

To regenerate only the Lambda(C2) data and copy the outputs into
`../python/data/E2/`:

```bash
cargo run --release -- c2-all -d 75 -o out75
cp out75/E2_C2.csv out75/E2_C2_products.csv ../python/data/E2/
rm -rf ../python/data/E2/d2      # the cached d2 was computed from the old data
```

## Output contract

`generate` writes, to `--out-dir` (default: the current directory):

| File | Contents | Consumed by |
|---|---|---|
| `E2_rank.csv` | dimension at each tridegree `(n, s, f)`, including the n=0 Lambda(C2) column | propagator loader, charts |
| `E2_relations.csv` | multiplication table | Leibniz rule |
| `E2_E.csv`, `E2_H.csv`, `E2_P.csv` | suspension, Hopf, and Whitehead-product map data | naturality constraints |
| `E2_C2.csv` | Mahowald's map to Lambda(C2) | naturality from the stable/C2 inputs |
| `E2_C2_products.csv` | filtration-1 (h₀–h₃) products on the n=0 column | filtration-1 Leibniz rule |
| `E2_names.json` | class name → admissible-monomial word | chart labels |

Classes are named `n_s_f` (one-dimensional bidegrees) or `n_s_f_i` (indexed
basis); map values may be **sums** of basis classes, written `a + b + …`.
In particular `E2_C2.csv` computes the full induced map on E2 (rows with a
zero or incomplete image are omitted, so the solver treats them as unknown
rather than zero). The precise CSV formats are documented in `src/lib.rs`
and `src/io/csv.rs`.

`--mult-floor N` restricts the multiplication table to product targets with
s+f > N, for incrementally extending an existing table (`--mult-ceil` caps
it); the default computes the full table.

## Resource use

Apple Silicon laptop, `curtis` subcommand:

| degree bound | time | RAM |
|---|---|---|
| 60 | ~27 s | 450 MB |
| 70 | ~3.2 min | 1.45 GB |
| 79 | ~15 min | a few GB |

Time and memory grow steeply with the bound. A degree-80 run wants tens of
GB, so a high-memory machine helps, but nothing about the run is
cluster-specific.

## Module map

- `curtis.rs`: the Curtis algorithm, monomial generation and tag reduction;
  streams results into on-disk working storage as it goes
- `mon.rs`, `poly.rs`, `lambda.rs`: admissible monomials, polynomials over
  F2, and the lambda algebra differential
- `packed.rs`, `store.rs`: front-coded packed encoding of polynomials, and
  the append-only mmap-backed poly store they live in
- `data.rs`: `Tags`, `Cocycles`, `E2`, the views the operations read
- `sweep.rs`: run consolidation used during reduction
- `maps/`: `products.rs` (multiplication table), `hopf_p.rs` (E, H, P),
  `c2.rs` (Mahowald's map)
- `io/cache.rs`, `io/csv.rs`: on-disk working storage and CSV/JSON exports
- `census.rs`, `grading.rs`: page statistics, grading types

---

<sub>Part of [adams-ehp](../README.md). Curtis algorithm → <b>E2 data</b> → [propagator](../python/README.md) → [charts](../charts/README.md).</sub>
