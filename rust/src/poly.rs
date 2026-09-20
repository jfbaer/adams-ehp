//! F2 polynomials in admissible lambda-algebra monomials, and their
//! reduction/completion machinery.
//!
//! A `Poly` is a set of `Monomial`s (mod-2 coefficients: present = 1). The
//! lex-largest monomial is the *lead* term; `reduce` rewrites everything into
//! admissible normal form via the Adem relations; `complete` expresses a
//! cocycle in the E2 survivor basis by repeatedly peeling recognized leads.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashSet;

use crate::data::{Cocycles, ResolvedParts, Tags, E2};
use crate::lambda::{is_admissible_place, leibniz, reduce_pair};
use crate::mon::{Idx, Monomial};
use crate::sweep::{head_equals, Run};

/// Index-slice -> signed vector (for the io/name boundary; `Monomial` derefs
/// to `[Idx]`, so `&mon` works directly).
pub fn to_i32s(s: &[Idx]) -> Vec<i32> {
    s.iter().map(|&x| x as i32).collect()
}

/// Sort ascending and cancel equal pairs (mod-2 normalization).
fn normalize(mut v: Vec<Monomial>) -> Vec<Monomial> {
    v.sort_unstable();
    let mut out = Vec::with_capacity(v.len());
    let mut it = v.into_iter().peekable();
    while let Some(m) = it.next() {
        if it.peek() == Some(&m) {
            it.next(); // cancel the pair mod 2
        } else {
            out.push(m);
        }
    }
    out
}

// FIELD ORDER FROZEN: bincode cache compat (a sorted Vec serializes exactly
// like the BTreeSet it replaced: a length-prefixed ascending sequence).
// INVARIANT: `monomials` is sorted ascending with no duplicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Poly {
    pub monomials: Vec<Monomial>,
}

impl Default for Poly {
    fn default() -> Self {
        Self::new()
    }
}

impl Poly {
    /// The zero polynomial.
    pub fn new() -> Self {
        Poly {
            monomials: Vec::new(),
        }
    }

    /// Number of monomials (terms with coefficient 1).
    pub fn len(&self) -> usize {
        self.monomials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Build from a vec that is ALREADY sorted ascending and duplicate-free
    /// (checked in debug builds). Used by fast paths that preserve order by
    /// construction, e.g. prepending one common admissible prefix.
    pub(crate) fn from_sorted(monomials: Vec<Monomial>) -> Self {
        debug_assert!(monomials.windows(2).all(|w| w[0] < w[1]));
        Poly { monomials }
    }

    /// Build from arbitrary (unsorted, possibly duplicated) terms, mod 2.
    pub fn from_terms(terms: Vec<Monomial>) -> Self {
        Poly {
            monomials: normalize(terms),
        }
    }

    /// Constructor from a single Monomial.
    pub fn from_mon(monomial: Monomial) -> Self {
        Poly {
            monomials: vec![monomial],
        }
    }

    /// Constructor from a single monomial given as a signed index vector.
    pub fn from_monomial(monomial: Vec<i32>) -> Self {
        Poly::from_mon(Monomial::from(monomial))
    }

    /// The lexicographically largest monomial (lead term), as a signed vector.
    pub fn lead(&self) -> Option<Vec<i32>> {
        self.monomials.last().map(|m| m.to_i32_vec())
    }

    /// Lead term without conversion (borrowed).
    pub fn lead_mon(&self) -> Option<&Monomial> {
        self.monomials.last()
    }

    /// Stem of the lead term (0 for the zero polynomial).
    pub fn stem(&self) -> i32 {
        self.lead_mon().map_or(0, |lead| lead.stem())
    }

    /// Filtration of the lead term = its length (0 for the zero polynomial).
    pub fn filtration(&self) -> i32 {
        self.lead_mon().map_or(0, |lead| lead.len() as i32)
    }

    /// Toggle a monomial mod 2.
    pub fn toggle(&mut self, mon: Monomial) {
        match self.monomials.binary_search(&mon) {
            Ok(i) => {
                self.monomials.remove(i);
            }
            Err(i) => self.monomials.insert(i, mon),
        }
    }

    /// Toggle a monomial given as a signed index vector.
    pub fn add(&mut self, mon: Vec<i32>) {
        self.toggle(Monomial::from(mon));
    }

    /// Add a poly mod 2 into self (symmetric-difference merge of two sorted
    /// sequences — the hot operation of the Curtis loop).
    pub fn add_poly(&mut self, poly: Poly) {
        if poly.monomials.is_empty() {
            return;
        }
        if self.monomials.is_empty() {
            self.monomials = poly.monomials;
            return;
        }
        let a = std::mem::take(&mut self.monomials);
        let mut out = Vec::with_capacity(a.len() + poly.monomials.len());
        let mut ia = a.into_iter().peekable();
        let mut ib = poly.monomials.into_iter().peekable();
        loop {
            match (ia.peek(), ib.peek()) {
                (Some(x), Some(y)) => match x.cmp(y) {
                    Ordering::Less => out.push(ia.next().unwrap()),
                    Ordering::Greater => out.push(ib.next().unwrap()),
                    Ordering::Equal => {
                        ia.next();
                        ib.next(); // equal terms cancel mod 2
                    }
                },
                (Some(_), None) => out.push(ia.next().unwrap()),
                (None, Some(_)) => out.push(ib.next().unwrap()),
                (None, None) => break,
            }
        }
        self.monomials = out;
    }

    /// Product in the lambda algebra: concatenate every pair of monomials
    /// (mod 2), then rewrite into admissible normal form via `reduce`.
    pub fn multiply(&self, other: &Poly) -> Poly {
        let mut raw = Vec::with_capacity(self.monomials.len() * other.monomials.len());
        for mon1 in &self.monomials {
            for mon2 in &other.monomials {
                let mut concat: Vec<Idx> = Vec::with_capacity(mon1.len() + mon2.len());
                concat.extend_from_slice(mon1);
                concat.extend_from_slice(mon2);
                raw.push(Monomial::from(concat));
            }
        }
        Poly {
            monomials: normalize(raw),
        }
        .reduce()
    }

    /// Left-multiply by a single monomial: `mon · self`, reduced.
    pub fn prepend(&self, mon: Vec<i32>) -> Poly {
        let mon_poly = Poly::from_monomial(mon);
        mon_poly.multiply(self)
    }

    /// Express a cocycle in the E2 survivor basis: the returned list are the
    /// basis classes whose sum is homologous to `self`.
    ///
    /// Fused descending sweep (one pass; no restart per recognized survivor,
    /// no materialized tag differentials): pending terms live in a heap of
    /// runs (see `crate::sweep`); a popped lead that is a basis element is
    /// recorded and its cocycle representative peeled as a streaming run; a
    /// reducible lead pushes the stored d(tag) run. Leads are emitted in the
    /// canonical descending sequence of maxima.
    pub fn complete(
        &self,
        dim: i32,
        tags: &Tags,
        cocycles: &Cocycles,
        pages: &E2,
    ) -> Option<Vec<Monomial>> {
        let basis = pages.basis(crate::grading::tri(dim, self.stem(), self.filtration()));
        // Empty basis: nothing can ever be recognized.
        if basis.is_empty() {
            return None;
        }
        let basis_set: FxHashSet<&[Idx]> = basis.iter().map(|m| &**m).collect();
        // Stop condition: reduction halts when the current lead is strictly
        // BELOW every basis element — it can then never reach one.
        let basis_min: &Monomial = basis.iter().min().expect("non-empty basis");

        let mut heap: BinaryHeap<Run<'_>> = BinaryHeap::new();
        if !self.monomials.is_empty() {
            heap.push(Run::slice(Vec::new(), &self.monomials));
        }

        let mut out: Vec<Monomial> = Vec::new();
        let mut lead: Vec<Idx> = Vec::new();

        'sweep: while let Some(mut top) = heap.pop() {
            if top.head().is_none() {
                continue;
            }
            top.head_into(&mut lead);
            top.advance();
            if top.head().is_some() {
                heap.push(top);
            }
            // parity cancel across runs
            let mut count = 1u32;
            while let Some(peek) = heap.peek() {
                if !head_equals(peek, &lead) {
                    break;
                }
                let mut run = heap.pop().expect("peeked run");
                run.advance();
                count += 1;
                if run.head().is_some() {
                    heap.push(run);
                }
            }
            if count.is_multiple_of(2) {
                continue;
            }

            // survivor recognition (the in_lambda break-path below also
            // records the lead and peels its cocycle)
            let mut take_as_survivor = basis_set.contains(&lead[..]);

            if !take_as_survivor {
                // stop when the lead has fallen below the entire basis
                if lead[..] < **basis_min {
                    break 'sweep;
                }
                match tags.resolve_parts(&lead) {
                    Some(ResolvedParts::Stored {
                        prefix_len,
                        target,
                        tag,
                    }) => {
                        // in_lambda(dim): first index of the prefixed tag lead
                        let first = if prefix_len > 0 {
                            lead[0]
                        } else {
                            tag.lead_first().unwrap_or(0)
                        };
                        if first as i32 >= dim {
                            take_as_survivor = true; // lead lies outside Λ(dim)
                        } else {
                            let prefix = lead[..prefix_len].to_vec();
                            let mut run = Run::packed(prefix.clone(), target);
                            debug_assert!(head_equals(&run, &lead));
                            run.advance();
                            if run.head().is_some() {
                                heap.push(run);
                            }
                            if prefix_len > 0 {
                                let dprefix = Poly::from_terms(leibniz(&Monomial::from(
                                    prefix.clone(),
                                )));
                                let extra = dprefix.multiply(&tag.unpack());
                                if !extra.monomials.is_empty() {
                                    heap.push(Run::owned(Vec::new(), extra.monomials));
                                }
                            }
                            continue 'sweep;
                        }
                    }
                    Some(ResolvedParts::Synthesized { tag }) => {
                        let first = tag
                            .lead_mon()
                            .and_then(|l| l.first().copied())
                            .unwrap_or(0);
                        if first as i32 >= dim {
                            take_as_survivor = true;
                        } else {
                            let dtag = tag.differential();
                            let mut rest =
                                Vec::with_capacity(dtag.monomials.len().saturating_sub(1));
                            let mut cancelled = false;
                            for m in dtag.monomials {
                                if !cancelled && *m == *lead {
                                    cancelled = true;
                                } else {
                                    rest.push(m);
                                }
                            }
                            debug_assert!(cancelled);
                            if !rest.is_empty() {
                                heap.push(Run::owned(Vec::new(), rest));
                            }
                            continue 'sweep;
                        }
                    }
                    None => break 'sweep, // unresolvable lead: stop, keep survivors
                }
            }

            if take_as_survivor {
                let coc = cocycles.get(&lead).expect("cocycle must exist");
                out.push(Monomial::from(lead.clone()));
                // peel: the cocycle's lead IS this lead — skip that copy
                let mut run = Run::packed(Vec::new(), coc);
                debug_assert!(head_equals(&run, &lead));
                run.advance();
                if run.head().is_some() {
                    heap.push(run);
                }
            }
        }

        // Return Some(out) only if out is non-empty
        (!out.is_empty()).then_some(out)
    }

    /// Does the polynomial lie in Λ(dim), i.e. is the lead's first index
    /// < dim? (False for the zero polynomial or an empty lead monomial.)
    pub fn in_lambda(&self, dim: i32) -> bool {
        if let Some(lead_term) = self.lead_mon() {
            if let Some(&first_entry) = lead_term.first() {
                return (first_entry as i32) < dim;
            }
        }
        false
    }

    /// The algebraic Hopf map H: keep the monomials led by λ_{dim-1} and
    /// strip that leading generator; every other monomial maps to 0.
    pub fn hopf(&self, dim: i32) -> Self {
        let mut result = Poly::new();
        for monomial in &self.monomials {
            if monomial.first().map(|&f| f as i32) == Some(dim - 1) {
                result.toggle(Monomial::from(monomial.tail().to_vec()));
            }
        }
        result
    }

    /// The algebraic P map of the EHP sequence (detected by Whitehead
    /// products; the "Delta map" in Curtis' terminology): left-multiply
    /// by d(λ_{(dim-1)/2}).
    pub fn p(&self, dim: i32) -> Poly {
        let w = leibniz(&Monomial::from(vec![(dim - 1) / 2]));
        let mut w_poly = Poly::new();
        for m in w {
            w_poly.toggle(m);
        }
        w_poly.multiply(self)
    }

    /// Send a Poly through the Theorem-2.5 map to Λ(C2) = e_{2n}Λ ⊕ e_{2n-1}Λ,
    /// where `dim = 2n+1`. Result is in the `[e, β…]` encoding with
    /// `e = 1` marking the e_{2n} (κ₁, top) cell and `e = 0` the e_{2n-1}
    /// (κ₀, bottom) cell. Leading index `dim-1 (=2n) → top`, `dim-2 (=2n-1) →
    /// bottom`, `≤ dim-3 → dropped`.
    ///
    /// CORRECTION TERM: the clean readoff is not a chain map — on the
    /// `λ_{2n}λ_{4n}` family it must also emit `e_{2n-1}·λ_{4n+1}·β` into the
    /// BOTTOM cell (the e_{2n-1} summand, NOT e_{2n}), making the map an exact
    /// chain map. Because the correction is strictly lex-lower than the
    /// `[1, 4n, β]` readoff, it never changes a class's leading-term name.
    pub fn to_c2(&self, dim: i32) -> Poly {
        let two_n = (dim - 1) as Idx;
        let four_n = (2 * (dim - 1)) as Idx;
        let mut out = Poly::new();

        for mon in &self.monomials {
            let Some(&first) = mon.first() else { continue };
            if first == two_n {
                // e_{2n} · tail  ->  [1, tail…]
                out.toggle(Monomial::with_cell(1, mon.tail()));
                // correction: λ_{2n}λ_{4n}β  ->  e_{2n-1}·λ_{4n+1}β = [0, 4n+1, rest…]
                if mon.len() >= 2 && mon[1] == four_n {
                    let mut c: Vec<Idx> = vec![0, four_n + 1];
                    c.extend_from_slice(&mon[2..]);
                    out.toggle(Monomial::from(c));
                }
            } else if first as i32 == dim - 2 {
                // e_{2n-1} · tail  ->  [0, tail…]
                out.toggle(Monomial::with_cell(0, mon.tail()));
            }
        }

        out
    }

    pub fn differential(&self) -> Poly {
        let mut raw = Vec::with_capacity(self.monomials.len() * 4);
        for mon in &self.monomials {
            raw.extend(leibniz(mon));
        }
        Poly {
            monomials: normalize(raw),
        }
        .reduce()
    }

    /// Rewrite into admissible normal form via the Adem relations.
    ///
    /// Note: takes `&mut self` and DRAINS it (the terms seed the work stack);
    /// the reduced polynomial is the return value, and `self` is left empty.
    pub fn reduce(&mut self) -> Poly {
        let mut adm: Vec<Monomial> = Vec::new();

        let mut stack: Vec<Monomial> = std::mem::take(&mut self.monomials);

        while let Some(mon) = stack.pop() {
            let mut reduced = false;

            for i in 1..mon.len() {
                if !is_admissible_place(i, &mon) {
                    for r in reduce_pair(mon.get_i32(i - 1), mon.get_i32(i)) {
                        stack.push(mon.splice(i - 1, 2, &r));
                    }
                    reduced = true;
                    break;
                }
            }

            if !reduced {
                adm.push(mon);
            }
        }

        Poly {
            monomials: normalize(adm),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lambda::binom_bool;
    use crate::mon;

    /// All admissible monomials with leading index ≤ max_lead and total
    /// internal degree (Σi_j + length) ≤ max_deg.
    fn admissibles(max_lead: i32, max_deg: i32) -> Vec<Vec<i32>> {
        fn rec(cur: &mut Vec<i32>, deg: i32, last: i32, max_deg: i32, out: &mut Vec<Vec<i32>>) {
            let hi = std::cmp::min(2 * last, max_deg - deg - 1);
            for q in 0..=hi {
                cur.push(q);
                out.push(cur.clone());
                rec(cur, deg + q + 1, q, max_deg, out);
                cur.pop();
            }
        }
        let mut out = Vec::new();
        for first in 0..=max_lead.min(max_deg - 1) {
            let mut cur = vec![first];
            out.push(cur.clone());
            rec(&mut cur, first + 1, first, max_deg, &mut out);
        }
        out
    }

    /// d² = 0: the fundamental identity, exercising Leibniz + Adem together.
    #[test]
    fn differential_squares_to_zero() {
        for mon in admissibles(12, 16) {
            let p = Poly::from_monomial(mon.clone());
            let dd = p.differential().differential();
            assert!(dd.is_empty(), "d²(λ{mon:?}) = {:?} ≠ 0", dd.monomials);
        }
    }

    /// reduce is idempotent and its output is admissible everywhere.
    #[test]
    fn reduce_is_idempotent_and_admissible() {
        use crate::lambda::is_admissible_place;
        // inadmissible products λ_a λ_b with b > 2a, times a tail
        for a in 0..8 {
            for b in (2 * a + 1)..18 {
                let mut p = Poly::from_monomial(vec![a, b, 1]);
                let r = p.reduce();
                for m in &r.monomials {
                    for i in 1..m.len() {
                        assert!(
                            is_admissible_place(i, m),
                            "inadmissible output {m:?} from λ{a}λ{b}λ1"
                        );
                    }
                }
                let again = r.clone().reduce();
                assert_eq!(r.monomials, again.monomials, "reduce not idempotent");
            }
        }
    }

    /// binom_bool agrees with Pascal's triangle mod 2.
    #[test]
    fn binom_bool_matches_pascal_mod_2() {
        let n_max = 40usize;
        let mut row = vec![0u8; n_max + 1];
        row[0] = 1;
        for n in 0..=n_max {
            for (m, &c) in row.iter().enumerate() {
                assert_eq!(
                    binom_bool(n as i32, m as i32),
                    c == 1,
                    "binom({n},{m}) mod 2"
                );
            }
            // next row (mod 2), in place from the right
            for m in (1..=n_max).rev() {
                row[m] ^= row[m - 1];
            }
        }
    }

    /// Sorted-Vec Poly invariant: toggles and merges keep it sorted, dup-free.
    #[test]
    fn poly_invariant_after_arithmetic() {
        let mut p = Poly::from_monomial(vec![2, 4]);
        p.add_poly(Poly::from_monomial(vec![1, 2]));
        p.add_poly(Poly::from_monomial(vec![2, 4])); // cancels
        assert_eq!(p.monomials, vec![mon![1, 2]]);
        p.toggle(mon![3, 5]);
        p.toggle(mon![0]);
        assert!(p.monomials.windows(2).all(|w| w[0] < w[1]), "not sorted");
    }
}
