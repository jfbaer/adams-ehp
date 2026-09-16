//! Lambda-algebra E2-page computations for the EHP spectral sequence.
//!
//! This library computes the E2 page of the lambda algebra via the Curtis
//! algorithm and several operations on it, and writes them as CSV files
//! consumed by the SageMath pipeline in `python/`.
//!
//! # Pipeline
//!
//! 1. [`curtis::curtis`] runs the Curtis algorithm up to a total-degree
//!    (s+f) bound, producing [`data::Tags`] (non-survivors with their
//!    target/tag pairs), [`data::Cocycles`] (survivor → cocycle
//!    representative), and the page [`data::E2`] itself.
//!    [`io::cache::load_or_compute_curtis`] runs it fresh each invocation.
//! 2. [`maps::products`] computes the multiplication table,
//!    [`maps::hopf_p`] the Hopf map H, Whitehead product P, and suspension E,
//!    and [`maps::c2`] Mahowald's map to Λ(C2) — each by applying the
//!    operation to a class's cocycle representative and completing back into
//!    the E2 basis ([`poly::Poly::complete`]), so values are sums of basis
//!    classes.
//! 3. [`io::csv`] writes the results; [`verify`] independently checks the C2
//!    map against a brute-force F2 homology oracle.
//!
//! # CSV contract (consumed by `python/`)
//!
//! Classes are named `"n_s_f"` (single basis element at that tridegree) or
//! `"n_s_f_i"`; `"0"` is the zero class; sums are joined with `" + "`.
//! Files: `E2_rank.csv` (n, s, f, dimension — including the n = 0 Λ(C2)
//! column), `E2_relations.csv` (factor1, factor2, result), and
//! `E2_{E,H,P,C2}.csv` (element, image).
//!
//! Note: release builds use `panic = "abort"`; invariant violations (documented
//! per function) abort the process, which is intended behavior for a batch CLI.

pub mod census;
pub mod curtis;
pub mod data;
pub mod grading;
pub mod io;
pub mod lambda;
pub mod maps;
pub mod mon;
pub mod packed;
pub mod poly;
pub mod store;
pub(crate) mod sweep;
pub mod verify;

/// Boxed error alias used across the library (Send + Sync so results can cross
/// rayon and anyhow boundaries).
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Per-dimension operation results: class -> sum of image basis elements.
pub type OpResults =
    std::collections::HashMap<i32, std::collections::HashMap<Vec<i32>, Vec<Vec<i32>>>>;
/// Per-dimension product decompositions: (factor, factor) -> product classes.
pub type ProductTable =
    std::collections::HashMap<i32, std::collections::HashMap<(Vec<i32>, Vec<i32>), Vec<Vec<i32>>>>;
/// Coordinate-level operation results (used by the C2 map): class id -> sum.
pub type CoordMap = std::collections::HashMap<grading::ClassId, Vec<grading::ClassId>>;
/// Class-name dictionary: dimension -> monomial -> `"n_s_f[_i]"`.
pub type Names = std::collections::HashMap<i32, std::collections::HashMap<Vec<i32>, String>>;
