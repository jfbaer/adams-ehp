//! The `Monomial` type: an admissible word λ_{i1}λ_{i2}⋯ in the lambda
//! algebra, stored as a boxed slice of small unsigned indices.
//!
//! One representation serves both storage (BTreeSet/HashMap keys) and
//! manipulation. The derived `Ord` is lexicographic on the indices — the
//! ordering the completion algorithms and lead-term logic rely on (all
//! entries are non-negative, enforced at construction).
//!
//! In the Λ(C2) column (dimension key n = 0 of `E2`) the FIRST entry of a
//! monomial is a cell marker rather than a lambda index: `TOP` (1) = e_{2n},
//! `BOT` (0) = e_{2n-1}; the remainder is the Λ-coefficient.

use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

/// Storage scalar for lambda indices: 2 bytes per index.
/// Indices stay far below `u16::MAX` at any reachable degree; construction is
/// checked and panics loudly on overflow. Changing this scalar changes the
/// bincode cache format — the cache filename is versioned accordingly.
pub type Idx = u16;

/// Cell marker for the top cell e_{2n} of Λ(C2).
pub const TOP: Idx = 1;
/// Cell marker for the bottom cell e_{2n-1} of Λ(C2).
pub const BOT: Idx = 0;

/// An admissible lambda-algebra monomial (empty = the unit).
// FIELD ORDER FROZEN + serde(transparent): bincode cache compat.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Monomial(Box<[Idx]>);

impl Monomial {
    /// The empty monomial (the unit of the algebra).
    pub fn unit() -> Self {
        Monomial(Box::new([]))
    }

    /// Stem: the sum of the indices.
    pub fn stem(&self) -> i32 {
        self.0.iter().map(|&x| x as i32).sum()
    }

    /// The tail `self[1..]` (the ubiquitous "everything after the head").
    pub fn tail(&self) -> &[Idx] {
        &self.0[1..]
    }

    /// Entry as i32 (for the signed arithmetic in the Adem/differential kernels).
    pub fn get_i32(&self, i: usize) -> i32 {
        self.0[i] as i32
    }

    /// Replace `self[at .. at + take]` with `rep`, checked-converting `rep`.
    /// This is the one splice primitive behind Leibniz and Adem reduction.
    pub fn splice(&self, at: usize, take: usize, rep: &[i32]) -> Monomial {
        let mut v: Vec<Idx> = Vec::with_capacity(self.0.len() - take + rep.len());
        v.extend_from_slice(&self.0[..at]);
        v.extend(rep.iter().map(|&x| to_idx(x)));
        v.extend_from_slice(&self.0[at + take..]);
        Monomial(v.into_boxed_slice())
    }

    /// `head · self` (prepend one index).
    pub fn prepended(&self, head: Idx) -> Monomial {
        let mut v = Vec::with_capacity(self.0.len() + 1);
        v.push(head);
        v.extend_from_slice(&self.0);
        Monomial(v.into_boxed_slice())
    }

    /// A C2-column monomial `[cell, tail…]`.
    pub fn with_cell(cell: Idx, tail: &[Idx]) -> Monomial {
        let mut v = Vec::with_capacity(tail.len() + 1);
        v.push(cell);
        v.extend_from_slice(tail);
        Monomial(v.into_boxed_slice())
    }

    /// The indices as a `Vec<i32>` (for the signed scalar kernels).
    pub fn to_i32_vec(&self) -> Vec<i32> {
        self.0.iter().map(|&x| x as i32).collect()
    }
}

#[inline]
fn to_idx(x: i32) -> Idx {
    assert!(
        x >= 0 && x <= Idx::MAX as i32,
        "lambda index {x} out of range for the storage scalar"
    );
    x as Idx
}

impl Deref for Monomial {
    type Target = [Idx];
    fn deref(&self) -> &[Idx] {
        &self.0
    }
}

impl Borrow<[Idx]> for Monomial {
    fn borrow(&self) -> &[Idx] {
        &self.0
    }
}

impl From<Vec<i32>> for Monomial {
    fn from(v: Vec<i32>) -> Self {
        Monomial(v.into_iter().map(to_idx).collect())
    }
}

impl From<&[i32]> for Monomial {
    fn from(v: &[i32]) -> Self {
        Monomial(v.iter().copied().map(to_idx).collect())
    }
}

impl From<Vec<Idx>> for Monomial {
    fn from(v: Vec<Idx>) -> Self {
        Monomial(v.into_boxed_slice())
    }
}

impl fmt::Display for Monomial {
    /// "λ2λ4λ1"; the unit prints as "1".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "1");
        }
        for &i in self.0.iter() {
            write!(f, "λ{i}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Monomial {
    /// `"[2, 4, 1]"` — plain index-list form for logs.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

/// Construct a `Monomial` from index literals: `mon![2, 4, 1]`.
#[macro_export]
macro_rules! mon {
    ($($x:expr),* $(,)?) => {
        $crate::mon::Monomial::from(vec![$($x as i32),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_matches_lex_on_indices() {
        // prefix < longer word; elementwise lex otherwise
        assert!(mon![2, 4] < mon![2, 4, 1]);
        assert!(mon![2, 4, 1] < mon![2, 5]);
        assert!(mon![3] > mon![2, 100]);
        assert!(Monomial::unit() < mon![0]);
    }

    #[test]
    fn splice_replaces_a_window() {
        let m = mon![6, 12, 3];
        assert_eq!(m.splice(1, 1, &[7, 5]), mon![6, 7, 5, 3]);
        assert_eq!(m.splice(0, 2, &[]), mon![3]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn negative_indices_are_rejected() {
        let _ = Monomial::from(vec![2, -1]);
    }

    #[test]
    fn display_and_unit() {
        assert_eq!(mon![2, 4, 1].to_string(), "λ2λ4λ1");
        assert_eq!(Monomial::unit().to_string(), "1");
    }
}
