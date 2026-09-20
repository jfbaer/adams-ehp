//! Products on the E2 page: pairwise decompositions of classes as products,
//! their suspensions, and the full multiplication table written incrementally
//! to CSV.
//!
//! Conventions at this module's boundaries:
//!   - a class at tridegree (n, s, f) is a basis λ-monomial (as `Vec<i32>` of
//!     λ-indices: stem s = sum of the indices, filtration f = the length);
//!   - a product of classes at (n, s, f) and (n + s + f, s', f') is completed
//!     into the E2 basis at (n, s + s', f + f') and recorded as a SUM of basis
//!     monomials (an empty sum = the product completed to zero);
//!   - "suspends by k" means the same monomial is a basis element at
//!     dimension n + k, same (s, f) (`E2::suspends`).

use rayon::prelude::*;

use crate::data::{Cocycles, Tags, E2};
use crate::grading::tri;
use crate::io::csv::write_decompositions_csv_append;
use crate::mon::Monomial;

/// Horizon for propagating product relations to suspended dimensions.
const SUSPENSION_DIM_HORIZON: i32 = 80;

/// Find all products of pairs of E2 classes landing at (target_stem,
/// target_filt), completing each product into the E2 basis.
///
/// For each factor at (n, stem, filt) the cofactor is drawn from
/// (n + stem + filt, target_stem − stem, target_filt − filt), and the product
/// of the two cocycle representatives is completed at dimension n. Pairs in
/// which BOTH factors are suspensions (tested on leading λ-indices:
/// `element[0] < n − 1` and `source[0] < n + stem − 1`) are skipped here and
/// expected to arrive via suspension propagation in
/// [`compute_suspensions_vector_based`] rather than being recomputed. (Note:
/// if the lower-dimensional product is never completed, or non-suspending
/// summands are dropped during propagation, the skipped pair may not be fully
/// recovered.)
pub fn decompositions(
    target_stem: i32,
    target_filt: i32,
    pages: &E2,
    cocycles: &Cocycles,
    tags: &Tags,
) -> crate::Result<crate::ProductTable> {
    let mut results = crate::ProductTable::new();

    // Collect all work items that need expensive computation
    let mut work_items = Vec::new();

    // Iterate through all potential element positions
    for n in 2..pages.max_tot() {
        for stem in 0..=target_stem {
            for filt in 0..target_filt {
                {
                    let elements = pages.basis(tri(n, stem, filt));
                    for element in elements.iter() {
                        // Skip empty elements to avoid index out of bounds
                        if element.is_empty() {
                            continue;
                        }

                        // The cofactor: complementary (stem, filt), at the
                        // element's dimension plus its total degree s + f.
                        let source_dim = n + stem + filt;
                        let source_stem = target_stem - stem;
                        let source_filt = target_filt - filt;

                        if source_stem >= 0 && source_filt >= 0 {
                            {
                                let sources =
                                    pages.basis(tri(source_dim, source_stem, source_filt));
                                for source in sources.iter() {
                                    // Skip empty sources to avoid index out of bounds
                                    if source.is_empty() {
                                        continue;
                                    }

                                    // Admit only sources whose leading λ-index
                                    // is < n + stem.
                                    if (source[0] as i32) < n + stem {
                                        // Compute only when the same pair does
                                        // not exist one dimension down (then it
                                        // arrives via suspension propagation
                                        // from that base instead). Tested by
                                        // basis membership, not leading index:
                                        // λ0-led classes look like suspensions
                                        // at every n — the h0-towers — yet have
                                        // no lower base in the table, and a
                                        // leading-index test starved their
                                        // products out of the table entirely.
                                        // The descent must also keep the
                                        // pairing admissible (source leading
                                        // index < (n-1) + stem), or the lower
                                        // "base" never gets computed.
                                        let pair_descends = (source[0] as i32)
                                            < n - 1 + stem
                                            && pages
                                                .basis(tri(n - 1, stem, filt))
                                                .contains(element)
                                            && pages
                                                .basis(tri(
                                                    source_dim - 1,
                                                    source_stem,
                                                    source_filt,
                                                ))
                                                .contains(source);
                                        if !pair_descends {
                                            work_items.push((n, element.clone(), source.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Process work items in parallel
    let parallel_results: Vec<_> = work_items
        .par_iter()
        .filter_map(|(n, element, source)| {
            // Compute base product
            let element_poly = cocycles.get(element)?.unpack();
            let source_poly = cocycles.get(source)?.unpack();

            // Calculate target dimension for the product
            let target_dim = *n;

            if let Some(product) = element_poly
                .multiply(&source_poly)
                .complete(target_dim, tags, cocycles, pages)
            {
                let product: Vec<Vec<i32>> = product.iter().map(|m| m.to_i32_vec()).collect();
                Some((
                    target_dim,
                    element.to_i32_vec(),
                    source.to_i32_vec(),
                    product,
                ))
            } else {
                None
            }
        })
        .collect();

    // Collect all results into final HashMap
    for (target_dim, element, source, product) in parallel_results {
        results
            .entry(target_dim)
            .or_default()
            .insert((element, source), product);
    }

    Ok(results)
}

/// Propagate fundamental products to suspended dimensions: a relation computed
/// at target dimension d is re-emitted at d + k for k = 1, 2, … as long as
/// both factors still suspend (same monomial present as a basis element k
/// dimensions up) and at least one product summand does, stopping at
/// `SUSPENSION_DIM_HORIZON`. The factor/product monomials themselves are
/// unchanged by suspension — only the dimension key moves.
pub fn compute_suspensions_vector_based(
    fundamental_results: &crate::ProductTable,
    pages: &E2,
) -> crate::Result<crate::ProductTable> {
    let mut suspension_results = crate::ProductTable::new();

    // Collect all fundamental products for parallel processing
    let mut work_items = Vec::new();
    for (target_dim, dim_results) in fundamental_results {
        for ((element_vector, source_vector), product_vectors) in dim_results {
            work_items.push((
                *target_dim,
                element_vector.clone(),
                source_vector.clone(),
                product_vectors.clone(),
            ));
        }
    }

    // Process suspensions in parallel
    let parallel_suspension_results: Vec<_> = work_items
        .par_iter()
        .filter_map(
            |(target_dim, element_vector, source_vector, product_vectors)| {
                let mut local_suspensions = Vec::new();

                // Calculate element coordinates for suspension checking
                let element_stem: i32 = element_vector.iter().sum();
                let element_filt = element_vector.len() as i32;
                let source_stem: i32 = source_vector.iter().sum();
                let source_filt = source_vector.len() as i32;
                let source_dim = target_dim + element_stem;

                let element_mon = Monomial::from(element_vector.clone());
                let source_mon = Monomial::from(source_vector.clone());
                let product_mons: Vec<Monomial> = product_vectors
                    .iter()
                    .map(|v| Monomial::from(v.clone()))
                    .collect();

                // Check suspensions for k = 1, 2, 3, ...
                let mut k = 1;
                while pages.suspends(*target_dim, element_stem, element_filt, k, &element_mon)
                    && pages.suspends(source_dim, source_stem, source_filt, k, &source_mon)
                    && target_dim + k < SUSPENSION_DIM_HORIZON
                {
                    // Re-emit the row at dimension d+k by applying E^k to
                    // the base sum summand-wise: E(x*y) = E(x)*E(y), and E
                    // acts on the tag basis as "same tag if it survives,
                    // else zero" (the tag system's defining semantics), so
                    // dropping the non-surviving summands IS the suspension
                    // of the product, not a truncation of it.
                    let mut suspended_product_vectors = Vec::new();

                    for (product_vec, product_mon) in
                        product_vectors.iter().zip(product_mons.iter())
                    {
                        let prod_stem: i32 = product_vec.iter().sum();
                        let prod_filt = product_vec.len() as i32;

                        if pages.suspends(*target_dim, prod_stem, prod_filt, k, product_mon) {
                            suspended_product_vectors.push(product_vec.clone());
                        }
                    }

                    if !suspended_product_vectors.is_empty() {
                        // The factor words are unchanged by suspension; only
                        // the dimension key (target_dim + k) moves.
                        let suspended_element = element_vector.clone();
                        let suspended_source = source_vector.clone();

                        local_suspensions.push((
                            target_dim + k,
                            suspended_element,
                            suspended_source,
                            suspended_product_vectors,
                        ));
                    }
                    k += 1;
                }

                if local_suspensions.is_empty() {
                    None
                } else {
                    Some(local_suspensions)
                }
            },
        )
        .flatten()
        .collect();

    // Collect results into final HashMap
    for (suspended_dim, element_vec, source_vec, product_vecs) in parallel_suspension_results {
        suspension_results
            .entry(suspended_dim)
            .or_default()
            .insert((element_vec, source_vec), product_vecs);
    }

    Ok(suspension_results)
}

/// Compute the multiplication table for product targets with total degree
/// s+f above `floor` (0 = the full table) and, when `ceil` is set, at most
/// `ceil`, writing rows to `path` incrementally per (stem, filt) target.
/// A positive floor extends an existing table without recomputing its
/// low-degree rows — note `path` is created fresh and holds only the
/// above-floor rows; combining with previously computed rows happens
/// outside this function. Targets are processed in ascending total degree,
/// descending stem within a degree.
pub fn mult_table(
    tags: &Tags,
    cocycles: &Cocycles,
    pages: &E2,
    names: &crate::Names,
    path: &std::path::Path,
    floor: i32,
    ceil: Option<i32>,
) -> crate::Result<()> {
    // Collect unique (stem, filt) pairs from all pages
    let mut stem_filt_pairs = std::collections::HashSet::new();

    for k in pages.keys() {
        if k.s + k.f > floor && ceil.is_none_or(|c| k.s + k.f <= c) {
            stem_filt_pairs.insert((k.s, k.f));
        }
    }

    // Convert to sorted vector: first by total degree (s + f), then by descending s within each degree
    let mut pairs: Vec<(i32, i32)> = stem_filt_pairs.into_iter().collect();
    pairs.sort_by_key(|(s, f)| (s + f, -s));

    // Create the output CSV file with headers
    let file = std::fs::File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(["factor1", "factor2", "result"])?;
    wtr.flush()?;
    drop(wtr); // Close the file so we can append to it

    // Run decompositions for each (stem, filt) pair sequentially
    for (stem, filt) in pairs.iter() {
        log::info!(
            "Computing decompositions for ({}, {}) at total degree {}",
            stem,
            filt,
            stem + filt
        );

        let results = decompositions(*stem, *filt, pages, cocycles, tags)?;

        // Compute suspensions for this batch
        log::info!("Computing suspensions for ({}, {})...", stem, filt);
        let suspension_results = compute_suspensions_vector_based(&results, pages)?;

        // Write fundamental products to CSV immediately
        log::info!(
            "Writing fundamental products for ({}, {}) to CSV...",
            stem,
            filt
        );
        write_decompositions_csv_append(&results, names, path)?;

        // Write suspension products to CSV immediately
        log::info!(
            "Writing suspension products for ({}, {}) to CSV...",
            stem,
            filt
        );
        write_decompositions_csv_append(&suspension_results, names, path)?;
    }

    Ok(())
}
