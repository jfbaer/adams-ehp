//! Mahowald's map to Λ(C2) on the E2 page, and the module structure over the
//! sphere: the filtration-1 products (·h0, ·h1, ·h2, ·h3 = ·λ0, ·λ1, ·λ3, ·λ7).
//!
//! Λ(C2) = Cone(h0 = λ0·) over the lambda algebra; its E2 page is
//! ker(h0) [top cell e_{2n}, stem shifted +1] ⊕ coker(h0) [bottom cell
//! e_{2n-1}], stored in the shared `E2` map as the n = 0 column with
//! monomials `[cell, β…]`, cell ∈ {0 = e_{2n-1}, 1 = e_{2n}}.
//!
//! The chain-level map (including its correction term) is `Poly::to_c2`; this
//! module builds the C2 column (`compute_c2_e2`), reduces C2 cochains into a
//! SUM of basis names (`reduce_c2`), and exposes three computations sharing
//! that reduction:
//!   - `process_odd_spheres_csv`: Mahowald's map from odd spheres to Λ(C2);
//!   - `compute_c2_products`: filtration-1 products on the Λ(C2) homology;
//!   - `compute_all`: both of the above plus the rank and names of the Λ(C2)
//!     homology (the n = 0 column), written as CSV/JSON.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::Path;

use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::data::{Cocycles, Tags, E2};
use crate::grading::{tri, ClassId, Trigrade};
use crate::mon::{Idx, Monomial, BOT, TOP};
use crate::poly::Poly;

/// Backstop for the reduction loops: no legitimate reduction takes anywhere
/// near this many steps; hitting it means a broken tag table (a cycle), and
/// the loop bails rather than hanging.
const REDUCTION_STEP_LIMIT: u32 = 1_000_000;

/// Reducing a filtration-1 product `λ_I * λ_i` that lands at total degree `P`
/// reads tags/cocycles through degree `P + 1` and no higher: the lambda
/// differential and `Tags::resolve` reductions are total-degree-preserving,
/// and the only degree-raising step is the λ0 threading of the top-cell
/// terms (+1). With a curtis table to degree `N` (all tags through `N`
/// materialized), products landing at total degree `N - 1` are therefore
/// fully supported, so the cap is `N - MARGIN` with margin 1. (A product
/// that does run off the end of the tags warns loudly rather than corrupting
/// silently.)
const PRODUCT_TAG_MARGIN: i32 = 1;

/// The filtration-1 generators of the sphere acting on Λ(C2): the Hopf
/// elements h_k = λ_{2^k − 1}, i.e. λ0, λ1, λ3, λ7 (Hopf invariant one). The
/// label is the CSV column value for `factor2`.
pub const FILT1_GENERATORS: [(&str, Idx); 4] = [("h0", 0), ("h1", 1), ("h2", 3), ("h3", 7)];

/// Lines of `path` that are TERMINATED by '\n'. A kill mid-flush can leave a
/// torn final line; excluding unterminated tails keeps resume from misreading
/// a truncated record (e.g. `0_49_2` from `0_49_25`) as a complete one.
/// Missing/unreadable file = no lines.
fn complete_lines(path: &Path) -> Vec<String> {
    match std::fs::read_to_string(path) {
        Ok(s) => match s.rfind('\n') {
            Some(end) => s[..end].lines().map(str::to_string).collect(),
            None => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Atomically replace `path` with `header` + `lines` (tmp file + rename), so a
/// resume never appends after torn tail bytes left by a kill.
fn rewrite_csv(path: &Path, header: &str, lines: &[String]) -> crate::Result<()> {
    let tmp = path.with_extension("csv.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        writeln!(f, "{header}")?;
        for l in lines {
            writeln!(f, "{l}")?;
        }
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Sort a completed incremental CSV's data rows in place (atomic rewrite).
/// The parallel writers emit rows in rayon completion order, so without this
/// the byte order of `E2_C2.csv`/`E2_C2_products.csv` — though never the row
/// set — varied run to run, making byte-level diffs look changed.
fn sort_csv_rows(path: &Path, header: &str) -> crate::Result<()> {
    let content = std::fs::read_to_string(path)?;
    let mut rows: Vec<String> = content.lines().skip(1).map(str::to_string).collect();
    rows.sort();
    rewrite_csv(path, header, &rows)
}

/// Build the Λ(C2) column of the E2 page (stored at dimension key n = 0) as
/// ker(h0) ⊕ coker(h0) of the h0-multiplication map on the sphere E2 page at
/// dimension `dim`.
///
/// With `max_filt = Some(F)` (a filtration-capped curtis table, exact through
/// F) the column is built only from sphere classes at filtration ≤ F−1: the
/// ker(h0) test completes λ0·coc(β) against the sphere basis one filtration
/// up, so a class at filtration F would be gated against the missing
/// filtration-F+1 row and mis-classified.
pub fn compute_c2_e2(
    dim: i32,
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &mut E2,
    max_filt: Option<i32>,
) {
    let max_t = pages
        .0
        .keys()
        .filter(|k| k.n == dim)
        .map(|k| k.s + k.f)
        .max()
        .unwrap_or(0);
    let mut h0map = FxHashMap::default();
    for t in 0..=max_t {
        compute_c2_e2_degree(dim, t, tags, cocycles, pages, &mut h0map, max_filt);
    }
}

/// One sphere-source total degree `t` of the h0 LES analysis at dimension
/// `dim`: extends the rolling ker(h0) map with this degree's classes, then
/// registers their top/bottom-cell contributions to the n = 0 column.
///
/// `h0map` is cumulative state — entries for all degrees < `t` must already
/// be present (the coker test reads the degree-(t−1) classes' h0 images).
/// [`compute_c2_e2`] folds this over t in ascending order, which also fixes
/// a canonical n = 0 basis order (source classes sorted by (s, f) within
/// each degree) independent of hash-map iteration.
///
/// Correctness needs the curtis table complete through degree t+1: the
/// ker(h0) gate completes λ0·coc(β) — a degree-(t+1) chain — against the
/// basis one filtration up.
pub fn compute_c2_e2_degree(
    dim: i32,
    t: i32,
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &mut E2,
    h0map: &mut FxHashMap<Monomial, Vec<Monomial>>,
    max_filt: Option<i32>,
) {
    let in_range = |f: i32| max_filt.is_none_or(|cap| f < cap);
    // The fundamental class (the empty word) is the h0 tower's base:
    // λ0·ι completes to the λ0 class one filtration up. Without this seed
    // the ker/coker tests below misread the bottom of the column — a
    // spurious top-cell class at (1, 0), a spurious bottom-cell class at
    // (0, 1), and no bottom-cell fundamental class at (0, 0).
    if t == 0 {
        if let Some(res) =
            Poly::from_monomial(vec![0]).complete(dim, tags, cocycles, pages)
        {
            h0map.insert(Monomial::from(Vec::<i32>::new()), res);
        }
    }
    // Build h0map with parallel iterator chain
    // First collect entries to be processed
    let entries_to_process: Vec<Monomial> = pages
        .iter()
        .filter(|(k, _)| k.n == dim && in_range(k.f) && k.s + k.f == t)
        .flat_map(|(_, entries)| entries.iter().filter(|e| !e.is_empty()).cloned())
        .collect();

    // Process entries in parallel
    let new_h0: Vec<(Monomial, Vec<Monomial>)> = entries_to_process
        .par_iter()
        .filter_map(|entry| {
            cocycles
                .get(entry)?
                .unpack()
                .prepend(vec![0])
                .complete(dim, tags, cocycles, pages)
                .map(|res| (entry.clone(), res))
        })
        .collect();
    h0map.extend(new_h0);

    // Collect (s, f, el) into Vec to avoid borrowing `pages` twice; sort so
    // the n = 0 push order (hence the _i basis indices) is canonical.
    let mut items: Vec<(i32, i32, Monomial)> = pages
        .iter()
        .filter(|(k, _)| k.n == dim && in_range(k.f) && k.s + k.f == t)
        .flat_map(|(k, entries)| entries.iter().map(move |el| (k.s, k.f, el.clone())))
        .collect();
    items.sort_by_key(|(s, f, _)| (*s, *f));

    for (s, f, el) in items {
        if !h0map.contains_key(&el) {
            pages.add(tri(0, s + 1, f), el.prepended(TOP));
        }
        if f > 0 {
            let src = pages.basis(tri(dim, s, f - 1));
            if src
                .iter()
                .all(|x| h0map.get(x).map(|h| h.first() != Some(&el)).unwrap_or(true))
            {
                pages.add(tri(0, s, f), el.prepended(BOT));
            }
        } else {
            // Nothing lies below filtration 0, so every f = 0 class is
            // automatically a cokernel class: the bottom-cell fundamental
            // (the class named "0" at (0, 0)).
            pages.add(tri(0, s, f), el.prepended(BOT));
        }
    }
}

/// Primitive of a boundary `b` by RAW tag reduction: find γ with dγ = b by
/// repeatedly cancelling the leading term of `b` with its tag's differential and
/// accumulating the tags. Unlike the basis-gated completion, this works at
/// bidegrees that have no survivors (exactly where the λ₀-threading lands).
/// Returns None if `b` is not reducible (i.e. not a boundary) — should not
/// happen for the λ₀·top we feed it.
fn primitive_by_tags(mut b: Poly, tags: &Tags, ctx: &Monomial) -> Option<Poly> {
    let mut gamma = Poly::new();
    let mut guard = 0u32;
    while let Some(lead) = b.lead_mon().cloned() {
        guard += 1;
        if guard > REDUCTION_STEP_LIMIT {
            return None;
        }
        let (tag, dtag) = tags.resolve(&lead)?;
        gamma.add_poly(tag);
        b.add_poly(dtag); // must cancel `lead`
        // Litmus: one reduction round MUST strictly lower the lead. If `lead` is
        // still on top, the tag failed to cancel it — a broken reduction that
        // would otherwise spin to the step limit — so bail loudly, naming the
        // class under reduction.
        if b.lead_mon() == Some(&lead) {
            log::warn!(
                "STUCK in primitive_by_tags: lead {lead:?} unchanged after a round while reducing class {ctx:?}"
            );
            return None;
        }
    }
    Some(gamma)
}

/// Split a `[e, β]` C2 image into its top (e = 1, e_{2n}) and bottom
/// (e = 0, e_{2n-1}) Λ-coefficients.
fn split_cells(img: &Poly) -> (Poly, Poly) {
    let mut top = Poly::new();
    let mut bot = Poly::new();
    for m in &img.monomials {
        let Some(&e) = m.first() else { continue };
        let tail = Monomial::from(m.tail().to_vec());
        if e == TOP {
            top.toggle(tail);
        } else if e == BOT {
            bot.toggle(tail);
        }
    }
    (top, bot)
}

/// Reduce a Λ(C2) cochain `e_{2n}·top + e_{2n-1}·bot` into a SUM of C2 basis
/// names, returning the `[cell, β]` names whose sum is its homology class.
///
/// This is the shared reduction engine behind both Mahowald's map and the
/// filtration-1 products. It is DIMENSION-FREE: it recognizes leading terms
/// directly against the C2 column (`pages` at n = 0), reducing with the sphere
/// tags/cocycles.
///
///   - TOP cell: reduce `top`; a leading term L with `[1,L]` registered
///     (L ∈ ker h0) is recorded and peeled with its cocycle; a non-registered L
///     is reduced by its tag τ. BOTH the survivor peel and the tag carry the
///     cone's extra differential piece `λ0·(·)` into the bottom cell (threaded,
///     not lumped) — this is the coupling D(e_{2n}·w) ⊇ e_{2n-1}·λ0·w.
///   - BOTTOM cell: reduce `bot` with the plain differential (no extra piece),
///     recognizing `[0,L]` (L ∈ coker h0).
fn reduce_c2(
    mut top: Poly,
    mut bot: Poly,
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    ctx: &Monomial,
) -> Vec<Monomial> {
    // is a `[cell, L]` name a registered C2-column basis element?
    let registered = |name: &Monomial| -> bool {
        pages
            .basis(tri(0, name.stem(), name.len() as i32 - 1))
            .contains(name)
    };

    let mut result: Vec<Monomial> = Vec::new();

    // ---- TOP cell: recognize ker(h0), thread λ0 into the bottom ----
    let mut guard = 0u32;
    while let Some(l) = top.lead_mon().cloned() {
        guard += 1;
        if guard > REDUCTION_STEP_LIMIT {
            break;
        }
        let name = l.prepended(TOP);
        if registered(&name) {
            result.push(name);
            let Some(coc) = cocycles.get(&l) else { break };
            let coc = coc.unpack();
            // extra C2 piece: e_{2n-1}·primitive(λ0·cocycle(L))
            let lam0 = coc.prepend(vec![0]);
            top.add_poly(coc); // peel this survivor
            if !lam0.is_empty() {
                if let Some(beta) = primitive_by_tags(lam0, tags, ctx) {
                    bot.add_poly(beta);
                }
            }
        } else if let Some((tau, dtau)) = tags.resolve(&l) {
            top.add_poly(dtau); // reduce top by dτ
            bot.add_poly(tau.prepend(vec![0])); // extra C2 piece: λ0·τ
        } else {
            break; // leading term neither a ker(h0) survivor nor reducible
        }
        // Litmus: peeling/tagging must cancel `l`, so the top lead strictly drops.
        if top.lead_mon() == Some(&l) {
            log::warn!(
                "STUCK in reduce_c2 (top cell): lead {l:?} unchanged after a round while reducing class {ctx:?}"
            );
            break;
        }
    }

    // ---- BOTTOM cell: recognize coker(h0), plain differential ----
    let mut guard = 0u32;
    while let Some(l) = bot.lead_mon().cloned() {
        guard += 1;
        if guard > REDUCTION_STEP_LIMIT {
            break;
        }
        let name = l.prepended(BOT);
        if registered(&name) {
            result.push(name);
            let Some(coc) = cocycles.get(&l) else { break };
            bot.add_poly(coc.unpack()); // peel
        } else if let Some((_tau, dtau)) = tags.resolve(&l) {
            bot.add_poly(dtau);
        } else {
            break;
        }
        // Litmus: the bottom lead must strictly drop each round.
        if bot.lead_mon() == Some(&l) {
            log::warn!(
                "STUCK in reduce_c2 (bottom cell): lead {l:?} unchanged after a round while reducing class {ctx:?}"
            );
            break;
        }
    }

    result
}

/// The C2 map computed the H/P way: apply the (corrected) chain map `to_c2` to
/// the class's cocycle representative, then reduce the image into the C2 basis,
/// returning a SUM of `[e, β]` C2 names.
///
/// `img = to_c2(x)` is a cocycle (`d top = 0`, `d bot = λ0·top`); we split it
/// into cells and hand it to `reduce_c2`.
pub fn c2_complete_sum(
    x: &Poly,
    dim: i32,
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    ctx: &Monomial,
) -> Vec<Monomial> {
    let (top, bot) = split_cells(&x.to_c2(dim));
    reduce_c2(top, bot, tags, cocycles, pages, ctx)
}

/// The Λ(C2) cocycle representative of a basis class `name = [cell, β]`, as the
/// pair of Λ-coefficients `(top, bot)` of `z = e_{2n}·top + e_{2n-1}·bot`:
///
///   - BOTTOM cell `[0, β]` (β ∈ coker h0): `(0, coc(β))` — already a cocycle
///     (the bottom cell has no cone coupling), since coc(β) is a strict d-cycle.
///   - TOP cell `[1, β]` (β ∈ ker h0): `(coc(β), γ)` with `dγ = λ0·coc(β)`.
///     `e_{2n}·coc(β)` alone is NOT closed — `D(e_{2n}·coc(β)) = e_{2n-1}·λ0·coc(β)`
///     — so γ (a primitive of that boundary) closes it. The `e_{2n-1}·γ` term is
///     exactly where the NONTRIVIAL h0-products of H(Λ(C2)) live (top-cell ker
///     classes supporting h0-extensions into the bottom cell).
///
/// Computed ONCE per class — the γ primitive is the expensive part, and all four
/// filtration-1 products reuse this pair. Returns None if β has no stored
/// cocycle, or (top cell) no primitive exists — an out-of-range / non-ker(h0)
/// edge class, whose products are dropped.
fn c2_representative(
    name: &Monomial,
    tags: &Tags,
    cocycles: &Cocycles,
) -> Option<(Poly, Poly)> {
    let cell = *name.first()?;
    // β = the Λ-coefficient (tail); its cocycle representative in the sphere.
    // The fundamental class is its own representative (the empty word), which
    // the cocycle store never holds — without this fallback the bottom-cell
    // class "0" at (0, 0) gets no products at all.
    let tail = name.tail();
    let coc = if tail.is_empty() {
        Poly::from_monomial(Vec::new())
    } else {
        cocycles.get(tail)?.unpack()
    };

    if cell == BOT {
        debug_assert!(
            coc.differential().is_empty(),
            "c2_representative: coc({name:?}) is not a d-cycle; bottom-cell rep is not closed",
        );
        return Some((Poly::new(), coc));
    }

    // TOP cell: z = e_{2n}·coc(β) + e_{2n-1}·γ, dγ = λ0·coc(β). γ exists because
    // β ∈ ker(h0) makes λ0·coc(β) a boundary.
    match primitive_by_tags(coc.prepend(vec![0]), tags, name) {
        Some(gamma) => {
            // A single γ closes z exactly iff D(z) = 0, which splits as
            //   e_{2n}·d(coc(β))          — zero iff coc(β) is a strict d-cycle, and
            //   e_{2n-1}·(λ0·coc(β) + dγ) — zero iff dγ = λ0·coc(β).
            // Both hold by construction (strict-cycle cocycles + the tag invariant
            // d(tag)=target that primitive_by_tags telescopes); assert it holds.
            #[cfg(debug_assertions)]
            {
                debug_assert!(
                    coc.differential().is_empty(),
                    "c2_representative: coc({name:?}) is not a d-cycle; z not closed on top cell",
                );
                let mut defect = gamma.differential();
                defect.add_poly(coc.prepend(vec![0])); // dγ + λ0·coc(β), cancels to 0 (mod 2)
                debug_assert!(
                    defect.is_empty(),
                    "c2_representative: dγ ≠ λ0·coc({name:?}); a single γ does not close z",
                );
            }
            Some((coc, gamma))
        }
        None => {
            // λ0·coc(β) should reduce to 0 for a ker(h0) class; if it does not
            // (e.g. an h0-tower class whose h0-image runs off the top of the
            // computed range), the representative is incomplete — warn and drop.
            log::warn!(
                "c2_representative: no primitive for λ0·coc({name:?}); top-cell class dropped"
            );
            None
        }
    }
}

/// Coordinates in the n = 0 column for a `[cell, β]` C2 name (None if it is not
/// a registered basis element).
fn c2_name_to_coords(name: &Monomial, pages: &E2) -> Option<ClassId> {
    let stem = name.stem();
    let filt = name.len() as i32 - 1;
    pages
        .basis(tri(0, stem, filt))
        .iter()
        .position(|v| v == name)
        .map(|pos| ClassId::new(0, stem, filt, pos as i32))
}

/// Optional restriction of the odd-spheres→C2 map to a sub-region of SOURCE
/// classes: a single sphere dimension and/or a stem cap. `None` leaves the
/// axis unrestricted. Only the map's source enumeration is filtered — the C2
/// column itself is always built in full (image naming needs the whole basis).
#[derive(Clone, Copy, Default)]
pub struct SourceFilter {
    pub sphere: Option<i32>,
    pub max_stem: Option<i32>,
}

impl SourceFilter {
    fn admits(&self, k: &Trigrade) -> bool {
        self.sphere.is_none_or(|n| k.n == n) && self.max_stem.is_none_or(|s| k.s <= s)
    }
}

/// Apply the C2 map to every odd-sphere E2 class, returning per-class sums of
/// target coordinates in the n = 0 column. With `max_filt = Some(F)`, only
/// source classes at filtration ≤ F−1 are mapped (their images land in the
/// part of the C2 column that a capped table computes completely).
pub fn process_odd_spheres_csv(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    max_filt: Option<i32>,
    filter: SourceFilter,
) -> crate::Result<crate::CoordMap> {
    // Parallel over total degrees (each degree's classes run in parallel
    // within). In this one-shot path the n = 0 column is final before the
    // map runs and each class's computation is read-only into a keyed map,
    // so no cross-degree ordering is needed; the pipeline path drives
    // process_odd_spheres_degree directly, which does carry an ordering
    // contract (see its doc).
    let max_t = pages
        .0
        .keys()
        .filter(|k| k.n % 2 != 0)
        .map(|k| k.s + k.f)
        .max()
        .unwrap_or(0);
    let mut results = crate::CoordMap::new();
    let ts: Vec<i32> = (0..=max_t).collect();
    let per_degree: Vec<crate::CoordMap> = ts
        .into_par_iter()
        .map(|t| process_odd_spheres_degree(t, tags, cocycles, pages, max_filt, filter))
        .collect::<crate::Result<_>>()?;
    for m in per_degree {
        results.extend(m);
    }
    Ok(results)
}

/// The C2 map on the odd-sphere source classes of total degree exactly `t`
/// (keyed results, so folding over t reproduces [`process_odd_spheres_csv`]).
///
/// Correctness needs the curtis table complete through degree t+1 and the
/// n = 0 column of `pages` final through C2-degree t (i.e.
/// [`compute_c2_e2_degree`] run through source degree t): each image is
/// reduced within degree ≤ t and named against the registered n = 0 basis.
pub fn process_odd_spheres_degree(
    t: i32,
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    max_filt: Option<i32>,
    filter: SourceFilter,
) -> crate::Result<crate::CoordMap> {
    let mut results = crate::CoordMap::new();

    let odd_sphere_data = odd_sphere_sources(pages, max_filt, filter, Some(t));

    // Compute the C2 map the H/P way: complete the corrected chain image into a
    // SUM of C2 basis names, then convert each name to target coordinates.
    let parallel_results: Vec<_> = odd_sphere_data
        .par_iter()
        .map(|(dim, s, f, i, el)| {
            c2_image_of(ClassId::new(*dim, *s, *f, *i), el, tags, cocycles, pages)
        })
        .collect();

    // Insert results into the results HashMap
    for (element_coords, result_coords) in parallel_results {
        results.insert(element_coords, result_coords);
    }

    Ok(results)
}

/// The admitted odd-sphere source classes — every `(n, s, f, i, monomial)`
/// the Mahowald map is computed on. `t = Some(d)` restricts to total degree
/// `s + f = d` (the pipeline/per-degree contract); `None` enumerates all.
fn odd_sphere_sources(
    pages: &E2,
    max_filt: Option<i32>,
    filter: SourceFilter,
    t: Option<i32>,
) -> Vec<(i32, i32, i32, i32, Monomial)> {
    pages
        .0
        .iter()
        .filter(|(k, _entries)| {
            k.n % 2 != 0
                && max_filt.is_none_or(|cap| k.f < cap)
                && t.is_none_or(|d| k.s + k.f == d)
                && filter.admits(k)
        })
        .sorted_by_key(|(k, _)| **k)
        .flat_map(|(k, entries)| {
            entries
                .iter()
                .enumerate()
                .filter(move |(_, el)| match el.first() {
                    Some(&first) => first as i32 == k.n - 1 || first as i32 == k.n - 2,
                    None => false, // skip empty monomials
                })
                .map(move |(i, el)| (k.n, k.s, k.f, i as i32, el.clone()))
        })
        .collect()
}

/// The Mahowald-map image of ONE odd-sphere class: complete the corrected
/// chain image into a SUM of C2 basis names, then convert each name to target
/// coordinates. `vec![ClassId::ZERO]` marks "zero / could not complete" for
/// the downstream solver (omitted from the CSV, treated as unknown).
fn c2_image_of(
    id: ClassId,
    el: &Monomial,
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
) -> (ClassId, Vec<ClassId>) {
    // full cocycle representative of the class named `el`
    let Some(rep) = cocycles.get(el) else {
        return (id, vec![ClassId::ZERO]);
    };
    let rep = &rep.unpack();

    // each name [e, β] -> its coordinates in the n=0 C2 column
    let coords: Vec<ClassId> = c2_complete_sum(rep, id.grade.n, tags, cocycles, pages, el)
        .iter()
        .filter_map(|name| c2_name_to_coords(name, pages))
        .collect();

    if coords.is_empty() {
        (id, vec![ClassId::ZERO])
    } else {
        (id, coords)
    }
}

/// Compute the Mahowald map and write `E2_C2.csv` INCREMENTALLY: one flat
/// parallel pass over every source class, each class's row appended and
/// flushed as it completes, with a `E2_C2.done` sidecar line (the class name)
/// appended AFTER the row flush. A kill or wall-time limit therefore loses at
/// most the in-flight classes — never hours of finished reductions.
///
/// With `resume`, classes named in the sidecar or already present as rows are
/// skipped; zero-image classes killed between "computed" and "sidecar line"
/// are recomputed (correct, just redundant). Rows are one per class, so any
/// '\n'-terminated line is a complete record; torn tail bytes are truncated
/// before appending. The final row SET equals a one-shot run's (order differs;
/// no loader depends on order).
pub fn process_odd_spheres_csv_incremental(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    max_filt: Option<i32>,
    filter: SourceFilter,
    path: &Path,
    resume: bool,
) -> crate::Result<()> {
    let done_path = path.with_extension("done");
    let mut done: HashSet<String> = HashSet::new();
    let mut kept: Vec<String> = Vec::new();
    if resume {
        done = complete_lines(&done_path).into_iter().collect();
        for line in complete_lines(path) {
            let Some((element, _)) = line.split_once(',') else {
                continue;
            };
            if element == "element" {
                continue; // header
            }
            done.insert(element.to_string());
            kept.push(line);
        }
        rewrite_csv(path, "element,image", &kept)?; // truncates torn tail bytes
        log::info!(
            "[c2-map] resume: {} source classes already on disk, skipping them",
            done.len()
        );
    }

    let sources: Vec<(i32, i32, i32, i32, Monomial)> =
        odd_sphere_sources(pages, max_filt, filter, None)
            .into_iter()
            .filter(|(n, s, f, i, _)| {
                !done.contains(&pages.name_from_coords(ClassId::new(*n, *s, *f, *i)))
            })
            .collect();
    log::info!("[c2-map] {} source classes to compute", sources.len());

    let appending = resume && path.exists();
    let file = if appending {
        std::fs::OpenOptions::new().append(true).open(path)?
    } else {
        std::fs::File::create(path)?
    };
    let mut header = csv::Writer::from_writer(file);
    if !appending {
        header.write_record(["element", "image"])?;
        header.flush()?;
    }
    let writer = std::sync::Mutex::new(header);
    let done_file = if appending {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&done_path)?
    } else {
        std::fs::File::create(&done_path)? // truncate any stale done-list
    };
    let done_writer = std::sync::Mutex::new(done_file);

    sources
        .par_iter()
        .try_for_each(|(dim, s, f, i, el)| -> crate::Result<()> {
            let (coords, image) =
                c2_image_of(ClassId::new(*dim, *s, *f, *i), el, tags, cocycles, pages);
            let element = pages.name_from_coords(coords);
            // ZERO (zero / could-not-complete) images are OMITTED from the CSV
            // — the solver must treat them as unknown, not an explicit zero —
            // but still get a sidecar line so resume never redoes them.
            if image != vec![ClassId::ZERO] {
                let image_name = image
                    .iter()
                    .map(|c| pages.name_from_coords(*c))
                    .collect::<Vec<_>>()
                    .join(" + ");
                let mut w = writer.lock().expect("c2-map writer lock");
                w.write_record([element.as_str(), image_name.as_str()])?;
                w.flush()?; // flush so completed classes survive a kill
            }
            {
                let mut d = done_writer.lock().expect("c2-map done-list lock");
                writeln!(d, "{element}")?;
            }
            Ok(())
        })?;
    // Deterministic artifact: the rows were appended in completion order.
    sort_csv_rows(path, "element,image")?;
    log::info!("[c2-map] all source classes done");
    Ok(())
}

/// Compute every filtration-1 product `x · h_k` (h_k = λ0, λ1, λ3, λ7) for each
/// basis class `x` of the Λ(C2) homology (the n = 0 column) whose product LANDS
/// within the trustworthy range, writing `(factor1, generator, result)` rows to
/// `path` incrementally as each class finishes. Must run after `compute_c2_e2`.
///
/// `degree` is the curtis-table bound. A product of a class at `(0, s, f)` with
/// `h_gen` lands at total degree `s + f + 1 + gen`, and reducing it (the λ0
/// threading reaches one degree higher still) needs tags/survivors above that.
/// So products are computed only when they land at total degree ≤
/// `degree - PRODUCT_TAG_MARGIN` — e.g. a table to degree 72 yields products
/// landing at 71 or below. Products above the cap are neither computed (avoiding
/// expensive off-the-edge reductions that run out of tags) nor emitted.
///
/// Classes run in one flat parallel pass. Every product is announced BEFORE it
/// is reduced (`computing FACTOR * hK`), so a stall names exactly which
/// `λ_I * h_i` is in flight; the reduction routines also warn if a leading term
/// fails to drop after a round. Each class's `c2_representative` — whose γ
/// primitive is the expensive part — is built ONCE and reused for all four
/// generators. Rows are appended and flushed under a lock as each class
/// finishes (and sorted at the end), so a kill or timeout leaves every
/// completed class on disk.
pub fn compute_c2_products(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    degree: i32,
    path: &Path,
    max_filt: Option<i32>,
    resume: bool,
) -> crate::Result<()> {
    let prod_bound = degree - PRODUCT_TAG_MARGIN;

    // Group in-range classes by source total degree (s + f), ascending. A class
    // is skipped entirely when even its cheapest product (·h0, landing s+f+1)
    // exceeds the cap — no wasted γ work on the boundary shell. On a
    // filtration-capped table every product lands one filtration up, and the
    // landing class must lie in the trustworthy C2 column (filtration ≤ F−1),
    // so factors are further restricted to filtration ≤ F−2.
    let mut by_degree: std::collections::BTreeMap<i32, Vec<(ClassId, Monomial)>> =
        std::collections::BTreeMap::new();
    for (k, entries) in pages.iter().filter(|(k, _)| {
        k.n == 0 && k.s + k.f < prod_bound && max_filt.is_none_or(|cap| k.f + 1 < cap)
    }) {
        for (i, el) in entries.iter().enumerate() {
            if !el.is_empty() {
                by_degree
                    .entry(k.s + k.f)
                    .or_default()
                    .push((ClassId::new(0, k.s, k.f, i as i32), el.clone()));
            }
        }
    }
    let total: usize = by_degree.values().map(Vec::len).sum();
    log::info!(
        "[c2-products] {total} classes over {} source degrees; products land ≤ {prod_bound}",
        by_degree.len()
    );

    // RESUME: reload what an interrupted run already banked. Rows are flushed
    // per class (all of a class's rows in one lock hold), and the sidecar
    // `.done` line for a class is appended only AFTER its rows hit the disk —
    // so every factor in the CSV except possibly the LAST is complete, and
    // anything the sidecar vouches for is complete regardless of position.
    // Zero-product classes leave no rows, only a sidecar line; resuming a
    // pre-sidecar run recomputes them (correct, just redundant).
    let done_path = path.with_extension("done");
    let mut done: HashSet<String> = HashSet::new();
    if resume {
        let sidecar: HashSet<String> = complete_lines(&done_path).into_iter().collect();
        let mut kept: Vec<String> = Vec::new();
        let mut order: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for line in complete_lines(path) {
            let Some((factor, _)) = line.split_once(',') else {
                continue;
            };
            if factor == "factor1" {
                continue; // header
            }
            if seen.insert(factor.to_string()) {
                order.push(factor.to_string());
            }
            kept.push(line);
        }
        // The final batch is the only one a kill can have half-written; drop
        // and recompute that class unless the sidecar vouches for it.
        if let Some(last) = order.last().cloned() {
            if !sidecar.contains(&last) {
                kept.retain(|l| l.split_once(',').is_some_and(|(f, _)| f != last));
                seen.remove(&last);
                log::info!("[c2-products] resume: recomputing possibly-torn last class {last}");
            }
        }
        rewrite_csv(path, "factor1,generator,result", &kept)?;
        done = seen;
        done.extend(sidecar);
        log::info!(
            "[c2-products] resume: {} classes already on disk, skipping them",
            done.len()
        );
    }

    // Incremental CSV: header on a fresh run, rows appended (and flushed) per
    // class; the sidecar records every completed class (including all-zero ones).
    let appending = resume && path.exists();
    let file = if appending {
        std::fs::OpenOptions::new().append(true).open(path)?
    } else {
        std::fs::File::create(path)?
    };
    let mut header = csv::Writer::from_writer(file);
    if !appending {
        header.write_record(["factor1", "generator", "result"])?;
        header.flush()?;
    }
    let writer = std::sync::Mutex::new(header);
    let done_file = if appending {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&done_path)?
    } else {
        std::fs::File::create(&done_path)? // truncate any stale done-list
    };
    let done_writer = std::sync::Mutex::new(done_file);

    // One flat parallel pass over EVERY class, no per-degree barrier: each
    // class's products depend only on the (read-only) tags/cocycles/pages,
    // and a degree-by-degree barrier would serialize the deep-filtration
    // reduce_c2 grinds (hours each at high total degree); flat, they run
    // concurrently. Rows land in completion order and are sorted at the end,
    // so the artifact is deterministic.
    let flat: Vec<&(ClassId, Monomial)> = by_degree
        .values()
        .flatten()
        .filter(|(cid, _)| !done.contains(&pages.name_from_coords(*cid)))
        .collect();
    if resume {
        log::info!("[c2-products] {} of {total} classes remain", flat.len());
    }
    {
        flat.par_iter()
            .try_for_each(|(cid, name)| -> crate::Result<()> {
                // The class representative (the expensive γ) is built ONCE and
                // reused across all four generators.
                let factor = pages.name_from_coords(*cid);
                let mut records: Vec<[String; 3]> = Vec::new();
                if let Some((top, bot)) = c2_representative(name, tags, cocycles) {
                    for (label, gen) in FILT1_GENERATORS {
                        // Landing total degree = (s + gen) + (f + 1); keep only ≤ cap.
                        if cid.grade.s + cid.grade.f + 1 + gen as i32 > prod_bound {
                            continue;
                        }
                        // Announce BEFORE reducing: a stall leaves this as the last
                        // line for its thread with no follow-up.
                        log::info!("[c2-products]   computing  {factor} * {label}");
                        let gp = Poly::from_monomial(vec![gen as i32]);
                        let result = reduce_c2(
                            top.multiply(&gp),
                            bot.multiply(&gp),
                            tags,
                            cocycles,
                            pages,
                            name,
                        )
                        .iter()
                        .filter_map(|nm| c2_name_to_coords(nm, pages))
                        .map(|c| pages.name_from_coords(c))
                        .collect::<Vec<_>>()
                        .join(" + ");
                        // c2_name_to_coords never yields ZERO, so a non-empty
                        // joined result is a genuine sum of basis names.
                        if !result.is_empty() {
                            records.push([factor.clone(), label.to_string(), result]);
                        }
                    }
                }
                if !records.is_empty() {
                    let mut w = writer.lock().expect("products writer lock");
                    for r in &records {
                        w.write_record(r)?;
                    }
                    w.flush()?; // flush so completed classes survive a kill
                }
                // Sidecar AFTER the row flush: a class is vouched complete only
                // once every row is durably on disk. Zero-product classes get a
                // line too, so a resume never redoes their expensive γ work.
                {
                    let mut d = done_writer.lock().expect("products done-list lock");
                    writeln!(d, "{factor}")?;
                }
                Ok(())
            })?;
    }
    // Deterministic artifact: the rows were appended in completion order.
    sort_csv_rows(path, "factor1,generator,result")?;
    log::info!("[c2-products] all {total} classes done");

    Ok(())
}

/// Build the C2 column, then compute the C2 map on every odd-sphere class
/// admitted by `filter` (`SourceFilter::default()` = all of them).
pub fn compute_c2_images(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &mut E2,
    max_filt: Option<i32>,
    filter: SourceFilter,
) -> crate::Result<crate::CoordMap> {
    log_filt_clamp(max_filt);
    let max_dim = pages.max_dimension();
    compute_c2_e2(max_dim, tags, cocycles, pages, max_filt);
    let results = process_odd_spheres_csv(tags, cocycles, pages, max_filt, filter)?;
    Ok(results)
}

/// One log line stating the clamps a filtration-capped run applies.
fn log_filt_clamp(max_filt: Option<i32>) {
    if let Some(cap) = max_filt {
        log::info!(
            "[max-filt {cap}] curtis table exact through filtration {cap}; \
             C2 column/map clamped to filtration ≤ {}, products to landing filtration ≤ {}",
            cap - 1,
            cap - 1
        );
    }
}

/// Options for [`compute_all`]: what to write and how the run is parameterized.
pub struct ComputeAllOpts {
    /// Compute Mahowald's odd-spheres→Λ(C2) map (the expensive pass) and write
    /// `E2_C2.csv`; when false, only products/rank/names are written.
    pub include_map: bool,
    /// Total-degree filtration cap (`None` = full range).
    pub max_filt: Option<i32>,
    /// Which source classes to include.
    pub filter: SourceFilter,
    /// Resume from a partially-written run (skip already-emitted classes).
    pub resume: bool,
}

/// The full Λ(C2) pipeline: build the n = 0 column, then write
/// - `E2_C2.csv`          — Mahowald's map (odd spheres → Λ(C2)); only when
///   `include_map` is set (it is the expensive pass);
/// - `E2_C2_products.csv` — filtration-1 products (factor1, generator, result);
/// - `E2_C2_rank.csv`     — rank of the Λ(C2) homology (n = 0 column);
/// - `E2_C2_names.json`   — names → `[cell, β]` word for the n = 0 basis.
///
/// With `include_map = false` this skips the odd-spheres→C2 map entirely and
/// writes only the products, rank, and names — the map is a separate pass that
/// nothing else depends on.
pub fn compute_all(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &mut E2,
    out_dir: &Path,
    degree: i32,
    opts: ComputeAllOpts,
) -> crate::Result<()> {
    let ComputeAllOpts {
        include_map,
        max_filt,
        filter,
        resume,
    } = opts;
    log_filt_clamp(max_filt);
    // 1. Build the Λ(C2) column (ker(h0) ⊕ coker(h0)) — needed by everything below.
    let max_dim = pages.max_dimension();
    log::info!("Building the Λ(C2) column...");
    compute_c2_e2(max_dim, tags, cocycles, pages, max_filt);

    // 2+3. Mahowald's map and the filtration-1 products BOTH read only the
    // finished column (plus tags/cocycles) and write different files, so
    // they run under one rayon::join sharing the worker pool: while the
    // map's few deep reductions grind, idle workers drain the products
    // queue instead of waiting at a stage boundary.
    let pages_ref: &E2 = pages;
    let products_path = out_dir.join("E2_C2_products.csv");
    log::info!(
        "Computing the C2 map and filtration-1 products (landing degree ≤ {}) concurrently...",
        degree - PRODUCT_TAG_MARGIN
    );
    let (map_res, prod_res) = rayon::join(
        || -> crate::Result<()> {
            if !include_map {
                return Ok(());
            }
            log::info!("Computing the C2 map (odd spheres → Λ(C2)) → E2_C2.csv (incremental)...");
            process_odd_spheres_csv_incremental(
                tags,
                cocycles,
                pages_ref,
                max_filt,
                filter,
                &out_dir.join("E2_C2.csv"),
                resume,
            )
        },
        || {
            compute_c2_products(
                tags,
                cocycles,
                pages_ref,
                degree,
                &products_path,
                max_filt,
                resume,
            )
        },
    );
    prod_res?;
    map_res?;

    // 4. Rank and names of the Λ(C2) homology (the n = 0 column).
    log::info!("Writing Λ(C2) homology rank and names...");
    write_c2_rank_csv(pages, &out_dir.join("E2_C2_rank.csv"))?;
    write_c2_names_json(pages, &out_dir.join("E2_C2_names.json"))?;

    if include_map {
        log::info!("Done: E2_C2.csv, E2_C2_products.csv, E2_C2_rank.csv, E2_C2_names.json");
    } else {
        log::info!("Done: E2_C2_products.csv, E2_C2_rank.csv, E2_C2_names.json (map skipped)");
    }
    Ok(())
}

/// Write the rank (basis dimension) of every bidegree in the Λ(C2) homology
/// (the n = 0 column) as `n, s, f, dimension`.
pub fn write_c2_rank_csv(pages: &E2, path: &Path) -> crate::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(["n", "s", "f", "dimension"])?;

    let mut keys: Vec<&Trigrade> = pages.keys().filter(|k| k.n == 0).collect();
    keys.sort();
    for k in keys {
        let rank = pages.basis(*k).len();
        wtr.write_record([
            k.n.to_string(),
            k.s.to_string(),
            k.f.to_string(),
            rank.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

/// Write the names of the Λ(C2) homology basis (n = 0 column) as JSON, mapping
/// each class name `"0_s_f[_i]"` to its `[cell, β…]` word (space-separated; the
/// leading entry is the cell marker, 1 = e_{2n}, 0 = e_{2n-1}).
pub fn write_c2_names_json(pages: &E2, path: &Path) -> crate::Result<()> {
    let mut keys: Vec<&Trigrade> = pages.keys().filter(|k| k.n == 0).collect();
    keys.sort();

    let mut json = serde_json::Map::new();
    for k in keys {
        let basis = pages.basis(*k);
        for (i, m) in basis.iter().enumerate() {
            let name = if basis.len() == 1 {
                format!("0_{}_{}", k.s, k.f)
            } else {
                format!("0_{}_{}_{}", k.s, k.f, i)
            };
            let word = m
                .to_i32_vec()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            json.insert(name, serde_json::Value::String(word));
        }
    }
    std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}
