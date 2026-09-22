# The preprint

*Automated proofs of unstable Adams differentials* — LaTeX source and the
built [`lambda.pdf`](lambda.pdf).

## Building

With a TeX Live installation (2024 or later; needs `biber` for the
biblatex bibliography):

```bash
latexmk -pdf lambda.tex
```

`lambda.tex` pulls in `legend.tex` (the why-graph legend figure),
`9_37_7_3.tex` (the worked proof-tree figure), `E8_bounds_true.tex`
(the results tables), and `good.bib`.

---

<sub>Part of [adams-ehp](../README.md): the computations behind the paper
live in [`rust/`](../rust/README.md), [`python/`](../python/README.md), and
[`charts/`](../charts/README.md).</sub>
