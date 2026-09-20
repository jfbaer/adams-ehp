//! The Curtis-table stores: `Tags` (non-surviving monomials with their
//! target/tag polynomial pair), `Cocycles` (survivor → full cocycle
//! representative), and `E2` (the page itself, basis monomials keyed by
//! (dimension n, stem s, filtration f)).
//!
//! Conventions: the dimension key n = 0 of `E2` is the Λ(C2) column, whose
//! monomials are `[cell, β…]` with cell ∈ {0 = e_{2n-1}, 1 = e_{2n}} — see
//! `crate::mon` and `crate::maps::c2`.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use std::sync::atomic::Ordering;

use crate::census;
use crate::grading::{tri, ClassId, Trigrade};
use crate::mon::{Idx, Monomial};
use crate::packed::{PackedPoly, PackedView};
use crate::poly::Poly;
use crate::store::{PolyRef, PolyStore};
use std::path::Path;

/// The tag table: non-surviving monomial -> references to its (target, tag)
/// pair in the on-disk poly store; leads + suffix lookup stay in RAM.
#[derive(Debug)]
pub struct Tags {
    pub index: FxHashMap<Monomial, (PolyRef, PolyRef)>,
    // Fast lookup: (sum, len, initial) -> [(key, tag lead)]
    lookup: FxHashMap<(i32, usize, i32), Vec<(Monomial, Monomial)>>,
    store: PolyStore,
}

impl Tags {
    /// New tag table writing its polys to `store_path` (truncates).
    pub fn create(store_path: &Path) -> crate::Result<Self> {
        Ok(Tags {
            index: FxHashMap::default(),
            lookup: FxHashMap::default(),
            store: PolyStore::create(store_path)?,
        })
    }

    /// Reopen from a sealed store + persisted index (the database).
    pub fn open(store_path: &Path, index: FxHashMap<Monomial, (PolyRef, PolyRef)>) -> crate::Result<Self> {
        let mut t = Tags {
            index,
            lookup: FxHashMap::default(),
            store: PolyStore::open(store_path)?,
        };
        t.rebuild_lookup();
        Ok(t)
    }

    /// Flush the store; afterwards reads are pure mmap (rayon-safe).
    pub fn seal(&mut self) -> crate::Result<()> {
        self.store.seal()
    }

    /// Bytes in the underlying poly store (mmap + tail); recorded by
    /// checkpoints so resume can truncate a partially-written suffix.
    pub fn store_len(&self) -> u64 {
        self.store.len()
    }

    /// Read-only snapshot for a consumer thread while this table keeps
    /// growing: clones the in-RAM index/lookup and opens a frozen view of the
    /// (append-only) store file, valid for every poly written so far. The
    /// store must be sealed first (no bytes still in the tail).
    pub fn snapshot(&self, store_path: &Path) -> crate::Result<Tags> {
        assert_eq!(
            self.store.tail_len(),
            0,
            "snapshot requires a sealed store (call seal() first)"
        );
        Ok(Tags {
            index: self.index.clone(),
            lookup: self.lookup.clone(),
            store: PolyStore::open_frozen(store_path)?,
        })
    }

    /// Open from a store + index like [`Tags::open`], but with a frozen
    /// store: appends (e.g. the evens-tag overlay) stay in a private in-RAM
    /// tail and never touch the file. For read-side consumers of a store
    /// that a producer may still own.
    pub fn open_frozen(
        store_path: &Path,
        index: FxHashMap<Monomial, (PolyRef, PolyRef)>,
    ) -> crate::Result<Self> {
        let mut t = Tags {
            index,
            lookup: FxHashMap::default(),
            store: PolyStore::open_frozen(store_path)?,
        };
        t.rebuild_lookup();
        Ok(t)
    }

    /// Reopen from a checkpointed store + index for continued writing
    /// (resume): like `open`, but appends extend the file instead of
    /// clobbering byte 0.
    pub fn reopen_append(
        store_path: &Path,
        index: FxHashMap<Monomial, (PolyRef, PolyRef)>,
    ) -> crate::Result<Self> {
        let mut t = Tags {
            index,
            lookup: FxHashMap::default(),
            store: PolyStore::open_append(store_path)?,
        };
        t.rebuild_lookup();
        Ok(t)
    }

    /// Iterate (key, target view, tag view) — census/reporting.
    pub fn iter_entries(&self) -> impl Iterator<Item = (&Monomial, PackedView<'_>, PackedView<'_>)> {
        self.index
            .iter()
            .map(move |(k, (t, g))| (k, self.store.view(*t), self.store.view(*g)))
    }

    /// (target store offset, key, first index of the tag's lead) for every
    /// entry whose key has length `len`. Keys of length ℓ+1 are created
    /// exactly by the curtis processing of length-ℓ monomials, and the store
    /// is append-only, so sorting by the offset recovers the original
    /// creation order — the raw material for the filtration-extension
    /// backfill of suppressed page rows.
    pub fn entries_by_key_len(&self, len: usize) -> Vec<(u64, Monomial, Idx)> {
        self.index
            .iter()
            .filter(|(k, _)| k.len() == len)
            .filter_map(|(k, (target, tag))| {
                let first = self.store.view(*tag).lead_first()?;
                Some((target.off, k.clone(), first))
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn lookup_key(key: &[Idx]) -> (i32, usize, i32) {
        let sum: i32 = key.iter().map(|&x| x as i32).sum();
        let initial = key.first().map(|&x| x as i32).unwrap_or(0);
        (sum, key.len(), initial)
    }

    /// Rebuild the (sum, len, initial) lookup structure from the index
    /// (used after reopening a persisted store).
    pub fn rebuild_lookup(&mut self) {
        self.lookup.clear();

        for (key, (_, tag_ref)) in &self.index {
            if let Some(tag_lead) = self.store.view(*tag_ref).lead() {
                self.lookup
                    .entry(Self::lookup_key(key))
                    .or_default()
                    .push((key.clone(), tag_lead));
            }
        }
    }

    /// Insert an already-packed pair (the merge-to-packed path — no re-pack).
    pub fn insert_packed(&mut self, tar: Monomial, target: &PackedPoly, tag: &PackedPoly) {
        if let Some(tag_lead) = tag.lead() {
            self.lookup
                .entry(Self::lookup_key(&tar))
                .or_default()
                .push((tar.clone(), tag_lead));
        }
        let target_ref = self.store.append(target);
        let tag_ref = self.store.append(tag);
        self.index.insert(tar, (target_ref, tag_ref));
    }

    pub fn insert(&mut self, tar: Monomial, polys: (Poly, Poly)) {
        // Store the leading term of the tag poly in the fast lookup
        if let Some(tag_lead) = polys.1.lead_mon() {
            self.lookup
                .entry(Self::lookup_key(&tar))
                .or_default()
                .push((tar.clone(), tag_lead.clone()));
        }
        let target_ref = self.store.append(&PackedPoly::pack(&polys.0));
        let tag_ref = self.store.append(&PackedPoly::pack(&polys.1));
        self.index.insert(tar, (target_ref, tag_ref));
    }

    /// Find the longest matching tail of `v` and return the stored
    /// (target, tag) pair. Borrows for map hits (the hot path — no clone);
    /// the synthesized `[a, 0]` special case owns its tiny tag poly (its
    /// target is computed on demand; that case always resolves exactly,
    /// never through the prefix path).
    pub fn get_full_pair(&self, v: &[Idx]) -> Option<PretagPair<'_>> {
        // Iterate from the longest possible tail length down to 1, maintaining
        // the suffix sum incrementally (O(L) total instead of O(L^2)).
        let mut sum: i32 = v.iter().map(|&x| x as i32).sum();
        for len in (1..=v.len()).rev() {
            let tail = &v[v.len() - len..];

            // Special case handling for length 2 with second element 0
            if tail.len() == 2 && tail[1] == 0 {
                census::TAG_SPECIAL.fetch_add(1, Ordering::Relaxed);
                return Some(PretagPair::Synthesized {
                    tag: Poly::from_monomial(vec![tail[0] as i32 + 1]),
                });
            }

            // Fast O(1) lookup
            if let Some(entries) = self.lookup.get(&(sum, len, tail[0] as i32)) {
                for (key, _) in entries {
                    if &**key == tail {
                        if let Some((target, tag)) = self.index.get(key) {
                            return Some(PretagPair::Stored {
                                target: self.store.view(*target),
                                tag: self.store.view(*tag),
                            });
                        }
                    }
                }
            }
            sum -= tail[0] as i32; // next tail drops its first index
        }
        None
    }

    /// Complete tag resolution: the fully materialized tag polynomial only
    /// (see [`Tags::resolve`] for the (tag, d(tag)) pair).
    pub fn get_tag(&self, v: &[Idx]) -> Option<Poly> {
        self.resolve(v).map(|(tag, _)| tag)
    }
}

/// A resolved pretag pair: borrowed from the store (the hot path — no clone)
/// or synthesized (the `[a, 0] → λ_{a+1}` special case, no map entry).
pub enum PretagPair<'a> {
    Stored {
        target: PackedView<'a>,
        tag: PackedView<'a>,
    },
    Synthesized {
        tag: Poly,
    },
}

impl PretagPair<'_> {
    /// Filtration (length of the tag's lead term) without decoding.
    pub fn tag_filtration(&self) -> i32 {
        match self {
            PretagPair::Stored { tag, .. } => tag.lead_len() as i32,
            PretagPair::Synthesized { tag } => tag.filtration(),
        }
    }

    /// First index of the tag's lead term.
    pub fn tag_lead_first(&self) -> Option<Idx> {
        match self {
            PretagPair::Stored { tag, .. } => tag.lead_first(),
            PretagPair::Synthesized { tag } => {
                tag.lead_mon().and_then(|l| l.first().copied())
            }
        }
    }
}

impl Tags {
    /// Resolve a reducible monomial to `(tag, d(tag))`, using the STORED
    /// target poly — which equals d(tag) by construction (the curtis loop
    /// inserts (reduced differential, cocycle-so-far) pairs, and `add_evens`
    /// inserts (d(λ_dim·…), λ_dim·…)) — instead of recomputing the
    /// differential of a possibly huge tag polynomial at every reduction step.
    ///
    /// Prefix case: d(prefix·tag) = d(prefix)·tag + prefix·d(tag) (Leibniz,
    /// mod 2). `prefix·d(tag)` is a pure admissible concatenation: d(tag)'s
    /// lead is the matched tail itself, so for every monomial m of d(tag),
    /// m[0] ≤ tail[0] = v[tag_len] ≤ 2·v[tag_len-1] (v is admissible).
    /// d(prefix) is tiny, so its true multiply is cheap. Both routes produce
    /// the canonical admissible normal form, hence identical sets.
    pub fn resolve(&self, v: &[Idx]) -> Option<(Poly, Poly)> {
        census::TAG_CALLS.fetch_add(1, Ordering::Relaxed);
        let f = v.len() as i32;

        let pair = match self.get_full_pair(v) {
            Some(p) => p,
            None => {
                census::TAG_MISS.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        if pair.tag_filtration() == f - 1 {
            census::TAG_EXACT.fetch_add(1, Ordering::Relaxed);
            return Some(match pair {
                PretagPair::Stored { target, tag } => (tag.unpack(), target.unpack()),
                PretagPair::Synthesized { tag } => {
                    let d = tag.differential();
                    (tag, d)
                }
            });
        }

        let tag_len = f as usize - pair.tag_filtration() as usize - 1;
        if tag_len == 0 || tag_len > v.len() {
            return None;
        }
        census::TAG_PREFIX.fetch_add(1, Ordering::Relaxed);
        let lead_first = pair.tag_lead_first()?;
        if 2 * (v[tag_len - 1] as i32) < lead_first as i32 {
            return None;
        }

        let prefix = &v[..tag_len];
        match pair {
            PretagPair::Stored { target, tag } => {
                let tag_plain = tag.unpack();
                let tag_full = prepend_admissible(prefix, &tag_plain);
                let mut d_full = prepend_admissible(prefix, &target.unpack());
                // + d(prefix)·tag (Leibniz remainder)
                let dprefix =
                    Poly::from_terms(crate::lambda::leibniz(&Monomial::from(prefix.to_vec())));
                d_full.add_poly(dprefix.multiply(&tag_plain));
                Some((tag_full, d_full))
            }
            // unreachable in practice: the synthesized case has filtration
            // f - 1 and resolves exactly above; kept for total correctness.
            PretagPair::Synthesized { tag } => {
                let tag_full = prepend_admissible(prefix, &tag);
                let d_full = tag_full.differential();
                Some((tag_full, d_full))
            }
        }
    }
}

/// Borrowed resolution for cursor-based reduction: the stored pair plus how
/// long a prefix the caller must prepend. Same guards and outcomes as
/// `resolve`, but nothing is materialized.
pub enum ResolvedParts<'a> {
    Stored {
        prefix_len: usize,
        target: PackedView<'a>,
        tag: PackedView<'a>,
    },
    Synthesized {
        tag: Poly,
    },
}

impl Tags {
    /// Non-materializing variant of [`Tags::resolve`] (see there for the
    /// mathematics). Returns the borrowed stored pair and the prefix length;
    /// the caller streams `prefix·target` / `prefix·tag` itself.
    pub fn resolve_parts(&self, v: &[Idx]) -> Option<ResolvedParts<'_>> {
        census::TAG_CALLS.fetch_add(1, Ordering::Relaxed);
        let f = v.len() as i32;

        let pair = match self.get_full_pair(v) {
            Some(p) => p,
            None => {
                census::TAG_MISS.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        if pair.tag_filtration() == f - 1 {
            census::TAG_EXACT.fetch_add(1, Ordering::Relaxed);
            return Some(match pair {
                PretagPair::Stored { target, tag } => ResolvedParts::Stored {
                    prefix_len: 0,
                    target,
                    tag,
                },
                PretagPair::Synthesized { tag } => ResolvedParts::Synthesized { tag },
            });
        }

        let tag_len = f as usize - pair.tag_filtration() as usize - 1;
        if tag_len == 0 || tag_len > v.len() {
            return None;
        }
        census::TAG_PREFIX.fetch_add(1, Ordering::Relaxed);
        let lead_first = pair.tag_lead_first()?;
        if 2 * (v[tag_len - 1] as i32) < lead_first as i32 {
            return None;
        }
        match pair {
            PretagPair::Stored { target, tag } => Some(ResolvedParts::Stored {
                prefix_len: tag_len,
                target,
                tag,
            }),
            // Unreachable in practice (the synthesized case resolves exactly);
            // fall back to a materialized owned tag for total correctness.
            PretagPair::Synthesized { tag } => Some(ResolvedParts::Synthesized {
                tag: prepend_admissible(&v[..tag_len], &tag),
            }),
        }
    }
}

impl Tags {
    /// Does `v` resolve to a tag? (No materialization.) Whenever this is
    /// true, the resolved tag's lead has length `v.len() - 1` identically
    /// (prefixed lead length = prefix_len + filtration = f − 1), so callers
    /// need no separate length check.
    pub fn resolves(&self, v: &[Idx]) -> bool {
        self.resolve_parts(v).is_some()
    }
}

/// `prefix · poly` when the junction is admissible for EVERY monomial — which
/// the caller's guard guarantees: it checks `2·prefix_last ≥ lead[0]`, and the
/// lead is the lex-max so `lead[0] ≥ mon[0]` for every monomial; the interior
/// adjacencies of `prefix` (a slice of an admissible monomial) and of each
/// stored monomial are admissible already. Hence no Adem rewriting can fire,
/// and prepending one common prefix preserves lex order — a pure
/// order-preserving concatenation instead of the general multiply + reduce.
fn prepend_admissible(prefix: &[Idx], poly: &Poly) -> Poly {
    let mut out = Vec::with_capacity(poly.monomials.len());
    for m in &poly.monomials {
        let mut w: Vec<Idx> = Vec::with_capacity(prefix.len() + m.len());
        w.extend_from_slice(prefix);
        w.extend_from_slice(m);
        out.push(Monomial::from(w));
    }
    Poly::from_sorted(out)
}

/// Survivor monomial -> reference to its cocycle representative in the
/// on-disk store. WRITE-ONLY during curtis (nothing reads a cocycle until
/// the operation passes), so the polys never need to be resident.
#[derive(Debug)]
pub struct Cocycles {
    pub index: FxHashMap<Monomial, PolyRef>,
    store: PolyStore,
}

impl Cocycles {
    pub fn create(store_path: &Path) -> crate::Result<Self> {
        Ok(Cocycles {
            index: FxHashMap::default(),
            store: PolyStore::create(store_path)?,
        })
    }

    pub fn open(store_path: &Path, index: FxHashMap<Monomial, PolyRef>) -> crate::Result<Self> {
        Ok(Cocycles {
            index,
            store: PolyStore::open(store_path)?,
        })
    }

    pub fn seal(&mut self) -> crate::Result<()> {
        self.store.seal()
    }

    /// Bytes in the underlying poly store; see [`Tags::store_len`].
    pub fn store_len(&self) -> u64 {
        self.store.len()
    }

    /// Read-only snapshot for a consumer thread; see [`Tags::snapshot`].
    pub fn snapshot(&self, store_path: &Path) -> crate::Result<Cocycles> {
        assert_eq!(
            self.store.tail_len(),
            0,
            "snapshot requires a sealed store (call seal() first)"
        );
        Ok(Cocycles {
            index: self.index.clone(),
            store: PolyStore::open_frozen(store_path)?,
        })
    }

    /// Open with a frozen store; see [`Tags::open_frozen`].
    pub fn open_frozen(
        store_path: &Path,
        index: FxHashMap<Monomial, PolyRef>,
    ) -> crate::Result<Self> {
        Ok(Cocycles {
            index,
            store: PolyStore::open_frozen(store_path)?,
        })
    }

    /// Reopen for continued writing (resume); see [`Tags::reopen_append`].
    pub fn reopen_append(
        store_path: &Path,
        index: FxHashMap<Monomial, PolyRef>,
    ) -> crate::Result<Self> {
        Ok(Cocycles {
            index,
            store: PolyStore::open_append(store_path)?,
        })
    }

    pub fn add(&mut self, key: Monomial, poly: Poly) {
        let r = self.store.append(&PackedPoly::pack(&poly));
        self.index.insert(key, r);
    }

    /// Insert an already-packed cocycle (the merge-to-packed path).
    pub fn add_packed(&mut self, key: Monomial, poly: &PackedPoly) {
        let r = self.store.append(poly);
        self.index.insert(key, r);
    }

    /// Zero-allocation lookup by index slice (`&mon` and `mon.tail()` both work).
    pub fn get(&self, key: &[Idx]) -> Option<PackedView<'_>> {
        self.index.get(key).map(|r| self.store.view(*r))
    }

    /// Iterate (key, view) — census/reporting.
    pub fn iter_entries(&self) -> impl Iterator<Item = (&Monomial, PackedView<'_>)> {
        self.index.iter().map(move |(k, r)| (k, self.store.view(*r)))
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

/// The E2 page: trigrade (n, s, f) -> basis monomials.
// FIELD ORDER FROZEN: bincode cache compat (Trigrade encodes like the tuple).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct E2(pub FxHashMap<Trigrade, Vec<Monomial>>);

impl Default for E2 {
    fn default() -> Self {
        Self::new()
    }
}

impl E2 {
    pub fn new() -> Self {
        E2(FxHashMap::default())
    }

    /// Append a basis monomial at the given trigrade.
    pub fn add(&mut self, key: Trigrade, value: Monomial) {
        self.0.entry(key).or_default().push(value);
    }

    pub fn max_dimension(&self) -> i32 {
        self.0.keys().map(|k| k.n).max().unwrap_or(0)
    }

    pub fn max_stem(&self) -> i32 {
        self.0.keys().map(|k| k.s).max().unwrap_or(0)
    }

    pub fn max_filt(&self) -> i32 {
        self.0.keys().map(|k| k.f).max().unwrap_or(0)
    }

    pub fn max_tot(&self) -> i32 {
        self.0.keys().map(|k| k.s + k.f).max().unwrap_or(0)
    }

    /// The basis at a trigrade (empty slice when the key is absent).
    pub fn basis(&self, key: Trigrade) -> &[Monomial] {
        self.0.get(&key).map_or(&[], |v| v.as_slice())
    }

    // Iterator methods
    pub fn iter(&self) -> impl Iterator<Item = (&Trigrade, &Vec<Monomial>)> {
        self.0.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &Trigrade> {
        self.0.keys()
    }

    /// Is `mon` a basis element k suspension levels up from (n, s, f)?
    pub fn suspends(&self, n: i32, s: i32, f: i32, k: i32, mon: &Monomial) -> bool {
        self.basis(tri(n, s, f).suspend(k)).contains(mon)
    }

    /// Class name `"n_s_f"` — with the `_i` basis index appended only when
    /// the trigrade has more than one basis element.
    pub fn name_from_coords(&self, id: ClassId) -> String {
        // Special case: ZERO represents "0"
        if id == ClassId::ZERO {
            return "0".to_string();
        }
        let t = id.grade;
        let basis = self.basis(t);
        if basis.len() == 1 {
            format!("{}_{}_{}", t.n, t.s, t.f)
        } else {
            format!("{}_{}_{}_{}", t.n, t.s, t.f, id.index)
        }
    }
}
