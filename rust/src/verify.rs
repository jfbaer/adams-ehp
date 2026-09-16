//! Independent regression check for the production map to Λ(C2)
//! (`maps::c2::c2_complete_sum`), run via `--verify-c2 <deg>`.
//!
//! Λ(C2) = Cone(h0 = λ0·), with E2 = ker(h0) [top cell e_{2n}, stem +1] ⊕
//! coker(h0) [bottom cell e_{2n-1}]. For each odd-sphere E2 class we take the
//! corrected chain image `x.to_c2(dim)` and, by brute-force F2 linear algebra
//! (local to one bidegree), reduce it modulo d_C2-boundaries. We then check that
//! the production map's sum of C2 names represents exactly that class
//! (`img ~ Σ rep(name)`), and cross-check its leading name against the older
//! single-term `complete_in_c2`. This machinery is deliberately independent of
//! the production completion logic, so it can catch regressions in it.

use std::collections::{BTreeSet, HashMap};

use crate::curtis::{add_evens, curtis};
use crate::data::{Cocycles, Tags, E2};
use crate::maps::c2::{c2_complete_sum, complete_in_c2, compute_c2_e2, primitive_by_tags};
use crate::mon::Monomial;
use crate::poly::{to_i32s, Poly};

type Mon = Vec<i32>; // a legacy C2 monomial [e, β…]
type Chain = BTreeSet<Mon>; // an F2 chain = set of legacy monomials (mod 2)

fn symdiff(a: &Chain, b: &Chain) -> Chain {
    a.symmetric_difference(b).cloned().collect()
}

/// Append `e_marker · poly` (in legacy `[e, ·]` form) into `out`, mod 2.
fn push_legacy(out: &mut Chain, e: i32, poly: &Poly) {
    for m in &poly.monomials {
        let mut v = vec![e];
        v.extend(to_i32s(m));
        if !out.remove(&v) {
            out.insert(v);
        }
    }
}

fn legacy_chain(poly: &Poly) -> Chain {
    poly.monomials.iter().map(|m| to_i32s(m)).collect()
}

/// All admissible Λ-monomials of exact `length` and `sum`, leading ≤ `max_lead`.
fn admissibles_fixed(max_lead: i32, length: i32, sum: i32) -> Vec<Mon> {
    if length <= 0 {
        return if length == 0 && sum == 0 {
            vec![vec![]]
        } else {
            vec![]
        };
    }
    if sum < 0 {
        return vec![];
    }
    let mut out = Vec::new();
    for a in 0..=max_lead.min(sum) {
        for mut rest in admissibles_fixed(2 * a, length - 1, sum - a) {
            let mut v = Vec::with_capacity(length as usize);
            v.push(a);
            v.append(&mut rest);
            out.push(v);
        }
    }
    out
}

/// d_C2 of `e_cell · β`, as legacy `[e, ·]` monomials:
///   cell 1 (e_{2n}):   e_{2n-1}·(λ0 β) + e_{2n}·(dβ)
///   cell 0 (e_{2n-1}):                   e_{2n-1}·(dβ)
fn dc2_legacy(cell: i32, beta: &[i32]) -> Chain {
    let b = Poly::from_monomial(beta.to_vec());
    let mut out = Chain::new();
    if cell == 1 {
        push_legacy(&mut out, 0, &b.prepend(vec![0]));
        push_legacy(&mut out, 1, &b.differential());
    } else {
        push_legacy(&mut out, 0, &b.differential());
    }
    out
}

/// d_C2 images of every chain monomial at source bidegree (src_stem, src_filt),
/// each landing in (src_stem-1, src_filt+1). `None` if the source basis exceeds
/// `cap` (too large to reduce by brute force).
fn c2_boundaries(src_stem: i32, src_filt: i32, cap: usize) -> Option<Vec<Chain>> {
    let top = admissibles_fixed(src_stem - 1, src_filt, src_stem - 1); // cell 1
    let bot = admissibles_fixed(src_stem, src_filt, src_stem); // cell 0
    if top.len() + bot.len() > cap {
        return None;
    }
    let mut out = Vec::with_capacity(top.len() + bot.len());
    for beta in top {
        out.push(dc2_legacy(1, &beta));
    }
    for beta in bot {
        out.push(dc2_legacy(0, &beta));
    }
    Some(out)
}

fn build_echelon(boundaries: Vec<Chain>) -> (Vec<Chain>, HashMap<Mon, usize>) {
    let mut ech: Vec<Chain> = Vec::new();
    let mut piv: HashMap<Mon, usize> = HashMap::new();
    for mut v in boundaries {
        while let Some(lead) = v.iter().next_back().cloned() {
            match piv.get(&lead) {
                Some(&i) => v = symdiff(&v, &ech[i]),
                None => {
                    piv.insert(lead, ech.len());
                    ech.push(v);
                    break;
                }
            }
        }
    }
    (ech, piv)
}

/// Reduce a chain modulo the boundary echelon; empty result ⟺ it is a boundary.
fn reduce_mod(ech: &[Chain], piv: &HashMap<Mon, usize>, mut v: Chain) -> Chain {
    while let Some(lead) = v.iter().next_back().cloned() {
        match piv.get(&lead) {
            Some(&i) => v = symdiff(&v, &ech[i]),
            None => break,
        }
    }
    v
}

/// Independent cocycle representative (legacy chain) of a C2 basis name `[e, s]`:
///   [1, s] -> e_{2n}·cocycle(s) + e_{2n-1}·primitive(λ0·cocycle(s))
///   [0, s] -> e_{2n-1}·cocycle(s)
fn c2_name_rep(name: &Monomial, tags: &Tags, cocycles: &Cocycles, pages: &E2) -> Chain {
    let _ = pages;
    let mut out = Chain::new();
    let Some((&e, s)) = name.split_first() else {
        return out;
    };
    let Some(scoc) = cocycles.get(s) else {
        return out;
    };
    let scoc = &scoc.unpack();
    if e == 1 {
        push_legacy(&mut out, 1, scoc);
        let lam0 = scoc.prepend(vec![0]);
        if !lam0.is_empty() {
            if let Some(prim) = primitive_by_tags(lam0, tags, name) {
                push_legacy(&mut out, 0, &prim);
            }
        }
    } else {
        push_legacy(&mut out, 0, scoc);
    }
    out
}

/// Run `curtis(deg)`, build the C2 column, and verify `c2_complete_sum` on every
/// odd-sphere E2 class: (1) its sum of names is homologous to the image via the
/// NF oracle, and (2) its leading name matches `complete_in_c2`.
///
/// With `max_filt = Some(F)` curtis is filtration-capped and only classes at
/// filtration ≤ F−1 are verified (higher ones would fail for lack of tags, not
/// because the map is wrong).
pub fn verify_c2(deg: i32, max_filt: Option<i32>) -> crate::Result<()> {
    const CAP: usize = 400_000;

    log::info!("Running curtis({deg})...");
    let db_dir = std::env::temp_dir().join(format!(
        "lambda_e2_verify_{}",
        crate::io::cache::db_stem(deg, max_filt)
    ));
    let (mut tags, mut cocycles, mut pages) = curtis(deg, None, &db_dir, max_filt)?;
    add_evens(&mut cocycles, &mut pages, &mut tags);
    tags.seal()?;
    cocycles.seal()?;
    compute_c2_e2(pages.max_dimension(), &tags, &cocycles, &mut pages, max_filt);

    let mut keys: Vec<_> = pages.keys().copied().collect();
    keys.sort();

    let mut checked = 0usize;
    let mut sum_ok = 0usize; // production sum is homologous to the image
    let mut sum_bad = 0usize;
    let mut bottom_unit = 0usize; // stem-0 e_{2n-1}·1 = [0] edge case (both maps read 0)
    let mut lead_ok = 0usize; // leading name matches complete_in_c2
    let mut lead_bad = 0usize;
    let mut multi = 0usize; // classes whose image is a genuine multi-term sum
    let mut skipped = 0usize; // bidegree too large to reduce
    let mut shown = 0usize;

    type Echelon = (Vec<Chain>, HashMap<Mon, usize>);
    let mut ech_cache: HashMap<(i32, i32), Option<Echelon>> = HashMap::new();
    let mut seen: BTreeSet<(i32, Monomial)> = BTreeSet::new();

    for key in keys {
        let (dim, s, f) = (key.n, key.s, key.f);
        if dim <= 0 || dim % 2 == 0 || max_filt.is_some_and(|cap| f >= cap) {
            continue;
        }
        let basis = pages.basis(key).to_vec();
        for mon in basis {
            if mon.is_empty() || !seen.insert((dim, mon.clone())) {
                continue;
            }
            let Some(x) = cocycles.get(&mon) else {
                continue;
            };
            let x = &x.unpack();
            let img = legacy_chain(&x.to_c2(dim));
            if img.is_empty() {
                continue;
            }
            checked += 1;

            let cone = c2_complete_sum(x, dim, &tags, &cocycles, &pages, &mon);
            if cone.len() > 1 {
                multi += 1;
            }

            // (1) NF-oracle check: img ~ Σ rep(name)
            let lead = img.iter().next_back().unwrap();
            let stem: i32 = lead.iter().sum();
            let filt: i32 = lead.len() as i32 - 1;
            let mut witness = Chain::new();
            for name in &cone {
                witness = symdiff(&witness, &c2_name_rep(name, &tags, &cocycles, &pages));
            }
            let combined = symdiff(&img, &witness);
            let entry = ech_cache
                .entry((stem, filt))
                .or_insert_with(|| c2_boundaries(stem + 1, filt - 1, CAP).map(build_echelon));
            match entry {
                Some((ech, piv)) => {
                    let nf = reduce_mod(ech, piv, combined);
                    if nf.is_empty() {
                        sum_ok += 1;
                    } else if nf.len() == 1 && nf.contains(&vec![0]) {
                        bottom_unit += 1; // e_{2n-1}·1, read as 0 by both maps
                    } else {
                        sum_bad += 1;
                        if shown < 20 {
                            println!("  SUM WRONG [dim={dim} class={mon:?}] -> {cone:?}");
                            shown += 1;
                        }
                    }
                }
                None => skipped += 1,
            }

            // (2) leading-name cross-check against complete_in_c2
            let cone_lead = cone.iter().max().cloned();
            let pipe = complete_in_c2(&mon, dim, s, f, &tags, &cocycles, &pages);
            let pipe_norm = match pipe {
                None => None,
                Some(v) if v.len() == 1 && v[0] == 0 => None,
                Some(v) => Some(v),
            };
            if cone_lead == pipe_norm {
                lead_ok += 1;
            } else {
                lead_bad += 1;
            }
        }
    }

    println!("\n=== C2 map verification (curtis deg {deg}) ===");
    println!("  classes checked:                {checked}");
    println!("  genuine multi-term sums:        {multi}");
    println!("  NF oracle: sum homologous:      {sum_ok}");
    println!("  NF oracle: sum WRONG:           {sum_bad}");
    println!("  bottom-unit edge (e_2n-1·1):    {bottom_unit}");
    println!("  leading == complete_in_c2:      {lead_ok}");
    println!("  leading DIFFERS:                {lead_bad}");
    println!("  skipped (bidegree too large):   {skipped}");
    if sum_bad == 0 && lead_bad == 0 {
        println!("  => c2_complete_sum verified on every reduced class.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correction term for the λ_{2n}λ_{4n} family lives in the e_{2n-1}
    /// (bottom) cell as λ_{4n+1}, NOT in e_{2n}. Locks `Poly::to_c2`.
    #[test]
    fn to_c2_correction_is_in_the_bottom_cell() {
        for n in 1..=6 {
            let dim = 2 * n + 1;
            let img = legacy_chain(&Poly::from_monomial(vec![2 * n, 4 * n]).to_c2(dim));
            assert!(img.contains(&vec![1, 4 * n]), "n={n}: missing e_2n·λ_4n");
            assert!(
                img.contains(&vec![0, 4 * n + 1]),
                "n={n}: correction not in e_2n-1"
            );
            assert!(
                !img.contains(&vec![1, 4 * n + 1]),
                "n={n}: correction wrongly in e_2n"
            );
        }
    }

    // The ORIGINAL (uncorrected) readoff map, kept to demonstrate that the
    // correction term is required.
    fn f_clean(x: &Poly, dim: i32) -> (Poly, Poly) {
        let (mut top, mut bot) = (Poly::new(), Poly::new());
        for m in &x.monomials {
            let v = to_i32s(m);
            let Some(&first) = v.first() else { continue };
            let tail = v[1..].to_vec();
            if first == dim - 1 {
                top.add(tail);
            } else if first == dim - 2 {
                bot.add(tail);
            }
        }
        (top, bot)
    }

    // + the e_{2n-1}·λ_{4n+1} correction on the λ_{2n}λ_{4n} family.
    fn f_corrected(x: &Poly, dim: i32) -> (Poly, Poly) {
        let (top, mut bot) = f_clean(x, dim);
        let (two_n, four_n) = (dim - 1, 2 * (dim - 1));
        for m in &x.monomials {
            let v = to_i32s(m);
            if v.len() >= 2 && v[0] == two_n && v[1] == four_n {
                let mut c = vec![four_n + 1];
                c.extend_from_slice(&v[2..]);
                bot.add(c);
            }
        }
        (top, bot)
    }

    // d_C2(top·e_{2n} + bot·e_{2n-1}) = (d top, λ0·top + d bot).
    fn dc2(top: &Poly, bot: &Poly) -> (Poly, Poly) {
        let t = top.differential();
        let mut b = top.prepend(vec![0]);
        b.add_poly(bot.differential());
        (t, b)
    }

    /// On the λ_{2n}λ_{4n} family the ORIGINAL (uncorrected) map fails to be a
    /// chain map, and the e_{2n-1}·λ_{4n+1} correction fixes it. A chain map
    /// satisfies f(dx) = d(f x); the residual checked below is f(dx) − d(f x).
    #[test]
    fn original_map_fails_and_correction_fixes_it() {
        let examples: Vec<(i32, Vec<i32>)> = vec![
            (3, vec![2, 4]),
            (3, vec![2, 4, 1]),
            (3, vec![2, 4, 1, 1, 1]),
            (5, vec![4, 8]),
            (5, vec![4, 8, 3]),
            (7, vec![6, 12]),
        ];
        for (dim, mon) in &examples {
            let x = Poly::from_monomial(mon.clone());
            let dx = x.differential();

            let (mut res_t, mut res_b) = f_clean(&dx, *dim);
            let (ct, cb) = f_clean(&x, *dim);
            let (rt, rb) = dc2(&ct, &cb);
            res_t.add_poly(rt);
            res_b.add_poly(rb);
            assert!(
                !res_t.is_empty() || !res_b.is_empty(),
                "dim={dim}, x={mon:?}: expected the uncorrected map to fail here"
            );

            let (mut cres_t, mut cres_b) = f_corrected(&dx, *dim);
            let (ct2, cb2) = f_corrected(&x, *dim);
            let (rt2, rb2) = dc2(&ct2, &cb2);
            cres_t.add_poly(rt2);
            cres_b.add_poly(rb2);
            assert!(
                cres_t.is_empty() && cres_b.is_empty(),
                "dim={dim}, x={mon:?}: corrected map is not a chain map here"
            );
        }
    }
}
