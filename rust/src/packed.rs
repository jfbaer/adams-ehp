//! Front-coded storage for polynomials at rest.
//!
//! The stored tables (`Tags` targets/tags, `Cocycles`) dominate memory at
//! high degree, and lex-sorted monomials share long prefixes (measured mean
//! lcp ≈ 8.6 of mean length ≈ 12.8 at degree 60). A `PackedPoly` stores the
//! terms in DESCENDING lex order, front-coded against the previous term, with
//! u8 indices (every lambda index is ≤ the total-degree bound, far below 256
//! at any reachable degree — checked at pack time):
//!
//! ```text
//! per term: [lcp: u8] [suffix_len: u8] [suffix: u8 × suffix_len]
//! ```
//!
//! ≈ 2 + (len − lcp) bytes per term instead of the ~90 (Box handle + u16
//! payload + allocator slack) of a boxed `Monomial` — roughly 8× smaller.
//! Descending order makes a forward decode walk exactly what the reduction
//! heap wants (largest term first).
//!
//! `PackedPoly` is storage-only: computation happens on `Poly`; consumers
//! call [`PackedPoly::unpack`] where they previously cloned, and the hot
//! reduction loop streams via [`PackedCursor`] without materializing.

use serde::{Deserialize, Serialize};

use crate::mon::{Idx, Monomial};
use crate::poly::Poly;

// FIELD ORDER FROZEN: bincode cache compat (the _v4 cache-file suffix must be bumped if this encoding changes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedPoly {
    n_terms: u32,
    data: Vec<u8>,
}

/// A borrowed view over packed bytes (same encoding as `PackedPoly`), e.g.
/// into the on-disk poly store. All decoding APIs live here; `PackedPoly`
/// derefs its owned buffer into a view.
#[derive(Clone, Copy)]
pub struct PackedView<'a> {
    pub n_terms: u32,
    pub data: &'a [u8],
}

impl<'a> PackedView<'a> {
    pub fn len(&self) -> usize {
        self.n_terms as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n_terms == 0
    }

    /// Decode back to a working `Poly` (ascending order restored).
    pub fn unpack(&self) -> Poly {
        let mut out: Vec<Monomial> = Vec::with_capacity(self.len());
        let mut cur = self.cursor();
        while let Some(term) = cur.current() {
            out.push(Monomial::from(term.to_vec()));
            cur.advance();
        }
        out.reverse(); // descending decode -> ascending storage
        Poly::from_sorted(out)
    }

    /// The lex-largest term (the lead), decoded.
    pub fn lead(&self) -> Option<Monomial> {
        let cur = self.cursor();
        cur.current().map(|t| Monomial::from(t.to_vec()))
    }

    /// Length (filtration) of the lead term without full decode.
    pub fn lead_len(&self) -> usize {
        if self.n_terms == 0 {
            0
        } else {
            self.data[1] as usize
        }
    }

    /// First index of the lead term (None if empty).
    pub fn lead_first(&self) -> Option<Idx> {
        if self.n_terms == 0 || self.data[1] == 0 {
            None
        } else {
            Some(self.data[2] as Idx)
        }
    }

    /// Descending decoding cursor (largest term first).
    pub fn cursor(&self) -> PackedCursor<'a> {
        let mut c = PackedCursor {
            data: self.data,
            pos: 0,
            remaining: self.n_terms as usize,
            buf: Vec::new(),
        };
        if c.remaining > 0 {
            c.decode_next();
        }
        c
    }
}

impl PackedPoly {
    /// Borrowed view over this poly's bytes.
    pub fn view(&self) -> PackedView<'_> {
        PackedView {
            n_terms: self.n_terms,
            data: &self.data,
        }
    }

    /// Raw encoded parts (for the on-disk store).
    pub fn raw(&self) -> (u32, &[u8]) {
        (self.n_terms, &self.data)
    }

    /// Pack a (lex-ascending) `Poly`. Panics if an index exceeds u8 range
    /// (impossible below total degree 255).
    pub fn pack(poly: &Poly) -> PackedPoly {
        let mut data = Vec::new();
        let mut prev: &[Idx] = &[];
        // descending = reverse of the ascending storage order
        for mon in poly.monomials.iter().rev() {
            let m: &[Idx] = mon;
            let lcp = prev
                .iter()
                .zip(m.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let suffix = &m[lcp..];
            debug_assert!(lcp <= u8::MAX as usize && suffix.len() <= u8::MAX as usize);
            data.push(lcp as u8);
            data.push(suffix.len() as u8);
            for &x in suffix {
                assert!(x <= u8::MAX as Idx, "lambda index {x} exceeds packed range");
                data.push(x as u8);
            }
            prev = m;
        }
        PackedPoly {
            n_terms: poly.monomials.len() as u32,
            data,
        }
    }

    pub fn len(&self) -> usize {
        self.n_terms as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n_terms == 0
    }

    /// In-RAM footprint: encoded payload bytes plus the handle struct itself
    /// (census/reporting).
    pub fn byte_size(&self) -> usize {
        self.data.len() + std::mem::size_of::<Self>()
    }

    /// Decode back to a working `Poly` (ascending order restored).
    pub fn unpack(&self) -> Poly {
        self.view().unpack()
    }

    /// The lex-largest term (the lead), decoded.
    pub fn lead(&self) -> Option<Monomial> {
        self.view().lead()
    }

    /// Length (filtration) of the lead term without full decode.
    pub fn lead_len(&self) -> usize {
        self.view().lead_len()
    }

    /// First index of the lead term (None if empty).
    pub fn lead_first(&self) -> Option<Idx> {
        self.view().lead_first()
    }

    /// Descending decoding cursor (largest term first).
    pub fn cursor(&self) -> PackedCursor<'_> {
        self.view().cursor()
    }
}

/// Streaming decoder over a `PackedPoly`, descending. `remaining` counts the
/// terms not yet consumed (current term included); `current()` exposes the
/// decoded term as a plain index slice; `advance()` consumes it.
pub struct PackedCursor<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: usize,
    buf: Vec<Idx>,
}

impl PackedCursor<'_> {
    pub fn current(&self) -> Option<&[Idx]> {
        if self.remaining == 0 {
            None
        } else {
            Some(&self.buf)
        }
    }

    pub fn remaining(&self) -> usize {
        self.remaining
    }

    pub fn advance(&mut self) {
        if self.remaining == 0 {
            return;
        }
        self.remaining -= 1;
        if self.remaining > 0 {
            self.decode_next();
        }
    }

    fn decode_next(&mut self) {
        let lcp = self.data[self.pos] as usize;
        let suffix_len = self.data[self.pos + 1] as usize;
        self.pos += 2;
        self.buf.truncate(lcp);
        for i in 0..suffix_len {
            self.buf.push(self.data[self.pos + i] as Idx);
        }
        self.pos += suffix_len;
    }
}

/// Streaming builder producing the exact `PackedPoly` byte encoding from a
/// strictly-descending term stream (as emitted by a parity merge) — the
/// merge-to-packed path that never materializes boxed monomials.
#[derive(Default)]
pub struct PackedEncoder {
    prev: Vec<Idx>,
    data: Vec<u8>,
    n_terms: u32,
}

impl PackedEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the next (strictly smaller) term.
    pub fn push(&mut self, term: &[Idx]) {
        debug_assert!(
            self.n_terms == 0 || self.prev.as_slice() > term,
            "encoder input must be strictly descending"
        );
        let lcp = self
            .prev
            .iter()
            .zip(term.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let suffix = &term[lcp..];
        debug_assert!(lcp <= u8::MAX as usize && suffix.len() <= u8::MAX as usize);
        self.data.push(lcp as u8);
        self.data.push(suffix.len() as u8);
        for &x in suffix {
            assert!(x <= u8::MAX as Idx, "lambda index {x} exceeds packed range");
            self.data.push(x as u8);
        }
        self.prev.clear();
        self.prev.extend_from_slice(term);
        self.n_terms += 1;
    }

    pub fn len(&self) -> usize {
        self.n_terms as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n_terms == 0
    }

    pub fn finish(self) -> PackedPoly {
        PackedPoly {
            n_terms: self.n_terms,
            data: self.data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mon;

    #[test]
    fn roundtrip_pack_unpack() {
        let mut p = Poly::new();
        for m in [
            mon![2, 4, 1],
            mon![2, 4, 1, 1],
            mon![2, 4, 2],
            mon![3],
            mon![0],
        ] {
            p.toggle(m);
        }
        let packed = PackedPoly::pack(&p);
        assert_eq!(packed.len(), 5);
        let back = packed.unpack();
        assert_eq!(back.monomials, p.monomials);
        assert_eq!(packed.lead(), p.lead_mon().cloned());
        assert_eq!(packed.lead_len(), 1); // lead is λ3
    }

    #[test]
    fn empty_and_unit() {
        let empty = PackedPoly::pack(&Poly::new());
        assert!(empty.is_empty());
        assert!(empty.unpack().monomials.is_empty());
        assert_eq!(empty.lead(), None);

        // the unit (empty monomial) packs as a zero-length term
        let unit = PackedPoly::pack(&Poly::from_monomial(vec![]));
        assert_eq!(unit.len(), 1);
        assert_eq!(unit.unpack().monomials.len(), 1);
    }

    #[test]
    fn cursor_walks_descending() {
        let mut p = Poly::new();
        for m in [mon![1], mon![2], mon![3], mon![2, 4]] {
            p.toggle(m);
        }
        let packed = PackedPoly::pack(&p);
        let mut cur = packed.cursor();
        let mut seen = Vec::new();
        while let Some(t) = cur.current() {
            seen.push(t.to_vec());
            cur.advance();
        }
        assert_eq!(seen, vec![vec![3], vec![2, 4], vec![2], vec![1]]);
    }
}
