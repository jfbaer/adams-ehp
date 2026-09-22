//! The Hopf map H, the Whitehead product P, and the suspension E on the E2
//! page. H and P apply the corresponding algebraic operation (`Poly::hopf`,
//! `Poly::p`) to a class's cocycle representative and complete the result
//! into the E2 basis, returning per-class images as sums of basis elements
//! (an empty sum = zero / not computed); E is a pure basis-membership test.
//!
//! Dimension conventions (tridegrees (n, s, f)):
//!   - H: sphere n → sphere 2n − 1;
//!   - P: sphere n (n odd) → ((n − 1)/2, s + (n − 1)/2 − 1, f + 2);
//!   - E: (n, s, f) → (n + 1, s, f), same monomial.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::data::{Cocycles, Tags, E2};
use crate::grading::{tri, Trigrade};
use crate::poly::to_i32s;

/// Index of single-element Hopf images by target trigrade (the P map's
/// "not in the image of H" shortcut).
pub type HopfImageIndex = HashMap<Trigrade, std::collections::HashSet<Vec<i32>>>;

/// Compute the Hopf map H on every E2 class: apply `Poly::hopf` to the
/// class's cocycle representative at dimension n and complete the result into
/// the E2 basis at dimension 2n − 1. Returns the per-class images (an empty
/// image = zero / could not complete) and the [`HopfImageIndex`] of
/// single-element images consumed by [`compute_p`].
pub fn compute_hopf(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
) -> crate::Result<(crate::OpResults, HopfImageIndex)> {
    let mut keys: Vec<_> = pages.0.keys().collect();
    keys.sort();

    // The Hopf image of each class is independent, so compute the keys in
    // parallel and merge the per-key results in sorted-key order afterwards
    // (so output is deterministic). Each entry is (dim, per-class images,
    // single-element image contributions).
    type KeyOut = (
        i32,
        Vec<(Vec<i32>, Vec<Vec<i32>>)>,
        Vec<(Trigrade, Vec<i32>)>,
    );
    let per_key: Vec<KeyOut> = keys
        .par_iter()
        .map(|&&key| {
            let dim = key.n;
            let mut local_results: Vec<(Vec<i32>, Vec<Vec<i32>>)> = Vec::new();
            let mut local_image: Vec<(Trigrade, Vec<i32>)> = Vec::new();

            if let Some(vectors) = pages.0.get(&key) {
                for compact_mon in vectors.iter() {
                    let mon = to_i32s(compact_mon);
                    let poly = cocycles.get(compact_mon).unwrap().unpack();

                    if let Some(completed) =
                        poly.hopf(dim).complete(2 * dim - 1, tags, cocycles, pages)
                    {
                        let processed_result: Vec<Vec<i32>> =
                            completed.iter().map(|m| m.to_i32_vec()).collect();

                        // Single-element images feed the P map's "not in the
                        // image of H" shortcut, indexed by target bidegree.
                        if processed_result.len() == 1 {
                            for result_vector in &processed_result {
                                let target_dim = 2 * dim - 1;
                                let target_stem = result_vector.iter().sum::<i32>();
                                let target_filt = result_vector.len() as i32;
                                local_image.push((
                                    tri(target_dim, target_stem, target_filt),
                                    result_vector.clone(),
                                ));
                            }
                        }
                        local_results.push((mon, processed_result));
                    } else {
                        local_results.push((mon, vec![]));
                    }
                }
            }
            (dim, local_results, local_image)
        })
        .collect();

    let mut hopf_results = crate::OpResults::new();
    let mut hopf_image_by_bidegree = HopfImageIndex::new();
    for (dim, local_results, local_image) in per_key {
        if !local_results.is_empty() {
            hopf_results.entry(dim).or_default().extend(local_results);
        }
        for (target, result_vector) in local_image {
            hopf_image_by_bidegree
                .entry(target)
                .or_default()
                .insert(result_vector);
        }
    }

    Ok((hopf_results, hopf_image_by_bidegree))
}

/// Compute the Whitehead product P on every odd-dimension E2 class: apply
/// `Poly::p` to the cocycle representative of a class at (n, s, f) (n odd)
/// and complete it at the target tridegree
/// ((n − 1)/2, s + (n − 1)/2 − 1, f + 2). An empty image records "zero or
/// not computed" — shortcut hits, out-of-range targets, and failed
/// completions alike.
///
/// Two skips avoid needless completions: a class already in the image of H
/// (single-element images only, via `hopf_image_by_bidegree`) has P = 0, and
/// a target bidegree whose basis consists entirely of suspensions cannot be
/// hit by P.
pub fn compute_p(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    hopf_image_by_bidegree: &HopfImageIndex,
) -> crate::Result<crate::OpResults> {
    let p_results = Arc::new(Mutex::new(HashMap::new()));

    let mut keys: Vec<_> = pages.0.keys().collect();
    keys.sort();

    keys.par_iter()
        .filter(|&&k| k.n % 2 != 0)
        .for_each(|&&key| {
            let (dim, stem, filt) = (key.n, key.s, key.f);
            let mut local_results: HashMap<Vec<i32>, Vec<Vec<i32>>> = HashMap::new();
            log::debug!("{}, {}, {}", dim, stem, filt);

            if let Some(vectors) = pages.0.get(&key) {
                for compact_mon in vectors.iter() {
                    let mon = to_i32s(compact_mon);

                    // Shortcut 1: P can only take nontrivial values on
                    // classes NOT in the image of H.
                    if let Some(hopf_set) = hopf_image_by_bidegree.get(&key) {
                        if hopf_set.contains(&mon) {
                            local_results.insert(mon, vec![]);
                            continue;
                        }
                    }

                    let poly = cocycles.get(compact_mon).unwrap().unpack();

                    let target_dim = (dim - 1) / 2;

                    // Target coordinates: (n, s, f) -> ((n-1)/2, s+(n-1)/2-1, f+2).
                    let target_stem_predicted = stem + (dim - 1) / 2 - 1;
                    let target_filt_predicted = filt + 2;

                    if target_stem_predicted > pages.max_stem() - 1
                        || target_filt_predicted > pages.max_filt() - 1
                    {
                        local_results.insert(mon, vec![]);
                        continue;
                    }

                    let p_poly = poly.p(dim);

                    // Shortcut 2: P can only hit non-suspending classes,
                    // so a target where the whole basis suspends receives 0.
                    let target_stem = p_poly.stem();
                    let target_filt = p_poly.filtration();
                    if all_basis_elements_suspend(target_dim, target_stem, target_filt, pages) {
                        local_results.insert(mon, vec![]);
                        continue;
                    }

                    if let Some(completed) = p_poly.complete(target_dim, tags, cocycles, pages) {
                        local_results
                            .insert(mon, completed.iter().map(|m| m.to_i32_vec()).collect());
                    } else {
                        local_results.insert(mon, vec![]);
                    }
                }
            }

            {
                let mut global_results = p_results.lock().unwrap();
                global_results
                    .entry(dim)
                    .or_insert_with(HashMap::new)
                    .extend(local_results);
            }
        });

    let results = Arc::try_unwrap(p_results).unwrap().into_inner().unwrap();
    Ok(results)
}

/// Does EVERY basis monomial at (dim, stem, filt) suspend (k = 1)?
/// Mathematical principle behind the caller's skip: P can only hit
/// non-suspending vectors, so a bidegree where everything suspends
/// receives nothing.
pub fn all_basis_elements_suspend(dim: i32, stem: i32, filt: i32, pages: &E2) -> bool {
    // Vacuously true when the bidegree is empty.
    pages
        .basis(tri(dim, stem, filt))
        .iter()
        .all(|mon| pages.suspends(dim, stem, filt, 1, mon))
}

/// Compute the suspension map E on every E2 class: a class suspends when the
/// same monomial is a basis element one dimension up.
pub fn compute_e(pages: &E2) -> crate::Result<crate::OpResults> {
    let mut results = crate::OpResults::new();

    let mut keys: Vec<_> = pages.keys().collect();
    keys.sort();

    for &key in &keys {
        let n = key.n;
        if let Some(vectors) = pages.0.get(key) {
            for compact_vector in vectors.iter() {
                let vector = to_i32s(compact_vector);

                results.entry(n).or_default();

                let suspends = pages
                    .basis(key.suspend(1))
                    .iter()
                    .any(|v| to_i32s(v) == vector);
                if suspends {
                    results
                        .get_mut(&n)
                        .unwrap()
                        .insert(vector.clone(), vec![vector]);
                } else {
                    results.get_mut(&n).unwrap().insert(vector, vec![]);
                }
            }
        }
    }

    Ok(results)
}
