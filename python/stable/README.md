# Input stable Adams differentials

These files are the external mathematical input to the whole pipeline:
every nontrivial differential the propagator deduces terminates, through
its proof chain, in entries from these tables.

| File | What it holds | Where it applies |
|---|---|---|
| `stable_sphere_diffs.csv` | stable Adams d_r for the sphere, r = 2..5 | every sphere in the stable range (n > s + 1) |
| `c2_diffs.csv` | Adams d_r for the cofiber of 2 (S/2), r = 2..5 | the n = 0 column (Lambda(C2), the target of Mahowald's map) |

## Provenance

The paper assumes knowledge of all differentials in the Adams spectral
sequences for the sphere and for S/2 through total degree 80, following:

- D. Isaksen, G. Wang, Z. Xu, *Stable stems* (the published stable range)
- W. Lin, G. Wang, Z. Xu, *Machine proofs for Adams differentials and
  extension problems among CW spectra*,
  [arXiv:2412.10876](https://arxiv.org/abs/2412.10876)
- the accompanying machine-verified dataset:
  [doi:10.5281/zenodo.14875701](https://doi.org/10.5281/zenodo.14875701)
  (version pinned in the paper; concept DOI
  [10.5281/zenodo.14272279](https://doi.org/10.5281/zenodo.14272279)
  resolves to the latest)

The tables here are the entry-level form of those differentials in this
repository's basis. They can therefore look quite different from the cited
references: the sources state each differential in their own generator
names, while these tables re-express the same differential in the lambda
algebra basis. The mathematical content is identical; only the coordinates
changed.

## Format

Both files share one CSV schema:

```csv
r,n,s,f,row,col,value
2,0,17,3,0,0,1
```

One line is one matrix entry of a known d_r. The differential at tridegree
`(n, s, f)` is a matrix from the basis of `(n, s, f)` to the basis of
`(n, s − 1, f + r)`; the line pins the entry with **source index `col`** and
**target index `row`** to `value` (over F2).

Three conventions matter:

- **Explicit zeros are information.** A `value = 0` line asserts that matrix
  entry vanishes; it is imposed as a constraint just like a 1.
- **Absent tridegrees stay open.** A tridegree with no lines is *not*
  assumed zero. It is left to naturality, the Leibniz rule, and d² = 0
  during `compute()`. Only what the tables actually determine is imposed.
- **Sphere rows are stable classes.** A row of `stable_sphere_diffs.csv` is
  recorded at one representative n, but `check_stable()` applies it to every
  sphere with n > s + 1; that is what stability means here. Rows of
  `c2_diffs.csv` apply only at n = 0.

Entries whose `row`/`col` exceed the loaded dataset's dimensions (possible
when running at a reduced degree range) are skipped with a warning; loading
happens once per page in `differentials.py` (`KNOWN_DIFF_CSVS`, `check_stable()`).

---

<sub>Part of [adams-ehp](../../README.md). These tables + [E2 data](../../rust/README.md) feed the [propagator](../README.md).</sub>
