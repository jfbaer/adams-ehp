//! The lambda-algebra kernels: the differential on generators, the Adem
//! relations, admissibility, and the Leibniz rule.
//!
//! These stay as free functions on signed scalars — the formulas involve
//! signed intermediates (`s - 2r - 1`) and bitwise Lucas tricks, and the
//! admissibility inequality `2·i_{k-1} ≥ i_k` reads best undisguised.

use std::collections::VecDeque;

use crate::mon::{Idx, Monomial};

/// d(λ_b) = Σ_c binom(b-c-1, c+1) λ_{b-c-1} λ_c  (mod 2).
pub fn differential_gen(b: i32) -> VecDeque<[i32; 2]> {
    let mut result = VecDeque::new();
    for c in 0..(((b + 1) / 2).max(1)) {
        let d = b - c - 1;
        if binom_bool(d, c + 1) {
            result.push_back([d, c]);
        }
    }
    result
}

/// Leibniz rule: d applied to a monomial, as the list of resulting monomials
/// (each position's generator replaced by a term of its differential).
pub fn leibniz(mon: &Monomial) -> Vec<Monomial> {
    let mut result = Vec::with_capacity(mon.len() * 4);
    for n in 0..mon.len() {
        for r in differential_gen(mon.get_i32(n)) {
            result.push(mon.splice(n, 1, &r));
        }
    }
    result
}

/// Binomial coefficient mod 2 via the Lucas lemma.
pub fn binom_bool(n: i32, m: i32) -> bool {
    if n < m {
        return false;
    }
    (n | !m) == -1
}

/// Is position `n` admissible, i.e. `2·mon[n-1] ≥ mon[n]`?
pub fn is_admissible_place(n: usize, mon: &[Idx]) -> bool {
    2 * (mon[n - 1] as i32) >= mon[n] as i32
}

/// Adem relation: rewrite the inadmissible pair λ_r λ_s (s > 2r) as a sum of
/// admissible-leading pairs.
pub fn reduce_pair(r: i32, s: i32) -> VecDeque<[i32; 2]> {
    let mut result = VecDeque::new();
    let b = s - 2 * r - 1;
    for c in 0..((b + 1) / 2) {
        if binom_bool(b - c - 1, c) {
            result.push_back([r + b - c, 1 + 2 * r + c]);
        }
    }
    result
}
