//! Shared machinery for descending-sweep reductions: heap RUNS that walk
//! (prefix · stored-poly) virtually, largest term first, with mod-2 parity
//! cancellation at pop time. Used by the Curtis tag reduction and the fused
//! completion loops.

use std::collections::BinaryHeap;

use crate::mon::{Idx, Monomial};
use crate::packed::{PackedCursor, PackedEncoder, PackedView};

/// A descending cursor over `prefix · src[i]`: `src` is a lex-ascending term
/// list (usually a BORROWED stored poly), walked from the back, with a common
/// admissible `prefix` prepended virtually. Pushing a whole tag differential
/// becomes one O(1) run instead of materializing (and boxing) every term —
/// the allocator was ~half of all CPU in the deg-60 profile.
pub(crate) struct Run<'a> {
    prefix: Vec<Idx>,
    src: RunSrc<'a>,
}

pub(crate) enum RunSrc<'a> {
    /// Owned lex-ascending terms, walked from the back (descending).
    Owned { terms: Vec<Monomial>, remaining: usize },
    /// Borrowed lex-ascending terms, walked from the back (descending).
    Slice { terms: &'a [Monomial], remaining: usize },
    /// Streaming decoder over a stored front-coded poly (already descending).
    Packed(PackedCursor<'a>),
}

impl<'a> Run<'a> {
    pub(crate) fn owned(prefix: Vec<Idx>, terms: Vec<Monomial>) -> Self {
        let remaining = terms.len();
        Run {
            prefix,
            src: RunSrc::Owned { terms, remaining },
        }
    }

    pub(crate) fn packed(prefix: Vec<Idx>, poly: PackedView<'a>) -> Self {
        Run {
            prefix,
            src: RunSrc::Packed(poly.cursor()),
        }
    }

    pub(crate) fn slice(prefix: Vec<Idx>, terms: &'a [Monomial]) -> Self {
        let remaining = terms.len();
        Run {
            prefix,
            src: RunSrc::Slice { terms, remaining },
        }
    }

    /// Current head as (prefix, term) — the lex-largest remaining element.
    pub(crate) fn head(&self) -> Option<(&[Idx], &[Idx])> {
        match &self.src {
            RunSrc::Owned { terms, remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    Some((self.prefix.as_slice(), &terms[*remaining - 1]))
                }
            }
            RunSrc::Slice { terms, remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    Some((self.prefix.as_slice(), &terms[*remaining - 1]))
                }
            }
            RunSrc::Packed(cur) => cur.current().map(|t| (self.prefix.as_slice(), t)),
        }
    }

    /// Step past the current head. Precondition: `head()` is `Some` —
    /// callers guarantee this by an explicit check or by the tag contract
    /// (a resolved tag's differential leads with the lead being cancelled).
    pub(crate) fn advance(&mut self) {
        match &mut self.src {
            RunSrc::Owned { remaining, .. } => *remaining -= 1,
            RunSrc::Slice { remaining, .. } => *remaining -= 1,
            RunSrc::Packed(cur) => cur.advance(),
        }
    }

    /// Materialize the current head into `out`.
    pub(crate) fn head_into(&self, out: &mut Vec<Idx>) {
        out.clear();
        let (p, m) = self.head().expect("head_into on empty run");
        out.extend_from_slice(p);
        out.extend_from_slice(m);
    }
}

/// Lexicographic (length-completing) comparison of two virtual monomials
/// given as (prefix, term) chains — identical semantics to `Monomial`'s
/// derived `Ord` on the concatenation, without materializing it.
pub(crate) fn cmp_heads(a: (&[Idx], &[Idx]), b: (&[Idx], &[Idx])) -> std::cmp::Ordering {
    a.0.iter()
        .chain(a.1.iter())
        .cmp(b.0.iter().chain(b.1.iter()))
}

impl PartialEq for Run<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Run<'_> {}
impl PartialOrd for Run<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Run<'_> {
    /// Order runs by their current head (empty runs sort last and are never
    /// re-pushed, so they cannot linger in the heap).
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.head(), other.head()) {
            (Some(a), Some(b)) => cmp_heads(a, b),
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (None, None) => std::cmp::Ordering::Equal,
        }
    }
}

/// Does the run's current head equal the (materialized) monomial in `lead`?
pub(crate) fn head_equals(run: &Run<'_>, lead: &[Idx]) -> bool {
    match run.head() {
        Some((p, m)) => {
            p.len() + m.len() == lead.len() && lead[..p.len()] == *p && lead[p.len()..] == *m
        }
        None => false,
    }
}

/// Drain a heap of runs into a packed encoder: pop heads descending, cancel
/// equal heads mod 2 by parity, emit the survivors. Consumes the heap.
pub(crate) fn drain_to_encoder(
    heap: &mut BinaryHeap<Run<'_>>,
    enc: &mut PackedEncoder,
    scratch: &mut Vec<Idx>,
) {
    while let Some(mut top) = heap.pop() {
        if top.head().is_none() {
            continue;
        }
        top.head_into(scratch);
        top.advance();
        if top.head().is_some() {
            heap.push(top);
        }
        let mut count = 1u32;
        while let Some(peek) = heap.peek() {
            if !head_equals(peek, scratch) {
                break;
            }
            let mut run = heap.pop().expect("peeked run");
            run.advance();
            count += 1;
            if run.head().is_some() {
                heap.push(run);
            }
        }
        if count % 2 == 1 {
            enc.push(scratch);
        }
    }
}
