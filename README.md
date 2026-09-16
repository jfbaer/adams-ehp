# Adams-EHP

Companion code and data for the preprint *Automated proofs of unstable Adams
differentials*. The two work products are the paper, available
as a PDF in [`paper/`](paper/lambda.pdf), and the interactive
unstable Adams charts with proof diagrams for each differential.

<p align="center">
  <a href="https://jfbaer.github.io/adams-ehp/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/assets/chart-hero-mocha.svg">
      <img alt="The unstable Adams spectral sequence for the 5-sphere with differentials through the 45-stem" src=".github/assets/chart-hero-latte.svg" width="900">
    </picture>
  </a>
</p>

<p align="center">
  <sub>above: the unstable Adams spectral sequence for S⁵ with differentials
  through the 45-stem</sub>
</p>

<p align="center">
  <a href="https://jfbaer.github.io/adams-ehp/">
    <img src="https://img.shields.io/badge/View_unstable_Adams_charts-1e66f5?style=for-the-badge&logoColor=white" alt="View unstable Adams charts">
  </a>
</p>

## The idea

The propagator takes the stable Adams differentials for the sphere and the
cofiber of 2, available from Weinan Lin's
[machine-verified computations](https://arxiv.org/abs/2412.10876)
(Lin–Wang–Xu; data on
[Zenodo](https://doi.org/10.5281/zenodo.14875701)), and uses naturality of
the EHP maps and the unstable Leibniz rule to compute *unstable* Adams
differentials.

Concretely: a Rust implementation of the Curtis algorithm computes the E₂
page of the unstable Adams spectral sequence for every sphere at once (the
lambda algebra), together with the maps connecting neighboring spheres. A
SageMath engine then treats each unknown unstable Adams differential as an
affine subspace of possible matrices and shrinks those subspaces by
intersecting with constraints imposed by the algebraic EHP maps and algebraic
compositions, recording a replayable proof chain for each deduction. The
method is developed in Section 3 of the [preprint](paper/lambda.pdf), *The
computation of unstable Adams differentials*.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/flow-mocha.svg">
    <img alt="Pipeline: the rust-generated unstable Adams E2-page and the input stable Adams differentials feed the computation of unstable Adams differentials, which produces proof diagrams and interactive charts" src=".github/assets/flow-latte.svg" width="720">
  </picture>
</p>

## Quick start: reproduce the computation

With a Rust toolchain (`cargo`) and [SageMath](https://doc.sagemath.org/html/en/installation/)
9.0+ on your `PATH`, two commands run the whole pipeline:

```bash
( cd rust   && cargo run --release -- generate -d 50 -o ../build/E2_50 )
( cd python && sage -python run.py 50 --data ../build/E2_50 )
```

The first generates the E2 page and its product tables (Curtis
algorithm); the second computes the d2–d5 differentials and chart CSVs
(into `python/charts/`). The loader reads the generated table's actual
coverage and lowers its bound to fit, so the two degrees need not be
matched by hand. Bound 50 takes a few minutes; the shipped dataset's full
range is total degree 76, reproducible with `-d 77` / `run.py 76` (the
rust and python stages each take several hours). The shipped data is
never modified; the fresh E2 lands in `build/`, and the propagator writes
its page data to `python/data/E{r}/` and chart CSVs to `python/charts/`.

To then browse your run as interactive charts:

```bash
( cd charts && poetry install && poetry run python generate_charts.py --charts-dir ../python/charts )
```

(add `--sidebyside` to also build the split-screen EHP map views; they are
the slow part of a chart build)

## Repository map

| Directory | What it is |
|---|---|
| [`rust/`](rust/) | the unstable Adams E₂-page: Curtis algorithm, products, EHP maps, Mahowald's map to Λ(C2) |
| [`python/`](python/) | the propagator: a SageMath engine computing d2–d5 with proof chains, plus the `why.py` proof-diagram tool |
| [`python/stable/`](python/stable/) | the stable inputs: known Adams differentials for the sphere and the cofiber of 2 (Lin–Wang–Xu, Isaksen–Wang–Xu) |
| [`charts/`](charts/) | the chart generator: turns propagator CSVs into the interactive HTML charts (bundled trimmed [SeqSee](https://github.com/JoeyBF/SeqSee)) |
| [`paper/`](paper/lambda.pdf) | the preprint (PDF) |

## Citing this work

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff). The
preprint is not yet posted; this BibTeX entry will be completed with the
arXiv ID when it is:

```bibtex
@misc{baer-adams-ehp,
  author = {Baer, Jake Francis},
  title  = {Automated proofs of unstable {Adams} differentials},
  year   = {2026},
  note   = {arXiv ID pending. Code and data: \url{https://github.com/jfbaer/adams-ehp}}
}
```

## License and acknowledgments

Licensed under the Apache License, Version 2.0; see [`LICENSE`](LICENSE).

The chart renderer is a trimmed, bundled copy of
[SeqSee](https://github.com/JoeyBF/SeqSee) by Joey Beauvais-Feisthauer and
Dan Isaksen (its license ships in [`charts/seqsee/`](charts/seqsee/)). The stable
Adams differentials seeding the propagator come from the machine-verified
computations of Weinan Lin, Guozhen Wang, and Zhouli Xu
([arXiv:2412.10876](https://arxiv.org/abs/2412.10876),
[Zenodo](https://doi.org/10.5281/zenodo.14875701)).
