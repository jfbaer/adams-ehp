//! Optional instrumentation: data-shape census and tag-lookup counters.
//!
//! Enabled by setting `LAMBDA_CENSUS=1`; all output goes to stderr via `log`
//! and never touches the CSV outputs. The numbers drive the storage/algorithm
//! decisions for high-degree runs (see the repository optimization notes).

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::data::{Cocycles, E2, Tags};
use crate::mon::Monomial;
use crate::poly::Poly;

/// Tag-lookup counters (incremented in `data.rs`, reported here).
pub static TAG_CALLS: AtomicU64 = AtomicU64::new(0);
pub static TAG_EXACT: AtomicU64 = AtomicU64::new(0);
pub static TAG_PREFIX: AtomicU64 = AtomicU64::new(0);
pub static TAG_SPECIAL: AtomicU64 = AtomicU64::new(0);
pub static TAG_MISS: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    std::env::var("LAMBDA_CENSUS").map_or(false, |v| v == "1")
}

struct PolyStats {
    polys: u64,
    terms: u64,
    max_terms: u64,
    index_sum: u64, // total u16 indices across all monomials
    max_len: u64,
    lcp_sum: u64, // sum of longest-common-prefix between consecutive monomials
    lcp_pairs: u64,
}

impl PolyStats {
    fn new() -> Self {
        PolyStats {
            polys: 0,
            terms: 0,
            max_terms: 0,
            index_sum: 0,
            max_len: 0,
            lcp_sum: 0,
            lcp_pairs: 0,
        }
    }

    fn add(&mut self, p: &Poly, distinct: Option<&mut HashSet<Monomial>>) {
        self.polys += 1;
        self.terms += p.monomials.len() as u64;
        self.max_terms = self.max_terms.max(p.monomials.len() as u64);
        let mut prev: Option<&Monomial> = None;
        if let Some(set) = distinct {
            for m in &p.monomials {
                set.insert(m.clone());
            }
        }
        for m in &p.monomials {
            self.index_sum += m.len() as u64;
            self.max_len = self.max_len.max(m.len() as u64);
            if let Some(q) = prev {
                let lcp = q.iter().zip(m.iter()).take_while(|(a, b)| a == b).count();
                self.lcp_sum += lcp as u64;
                self.lcp_pairs += 1;
            }
            prev = Some(m);
        }
    }

    fn report(&self, label: &str) {
        let mean_terms = self.terms as f64 / self.polys.max(1) as f64;
        let mean_len = self.index_sum as f64 / self.terms.max(1) as f64;
        let mean_lcp = self.lcp_sum as f64 / self.lcp_pairs.max(1) as f64;
        // Byte models per stored occurrence:
        //   current: 16 (Box handle) + 2L (payload) + ~16 (allocator quantum)
        //   arena:   2L (flat buffer) + 4 (offset)
        //   front-coded: 2*(L - lcp) + ~3 header
        let cur = self.terms as f64 * (32.0 + 2.0 * mean_len);
        let arena = self.terms as f64 * (4.0 + 2.0 * mean_len);
        let fc = self.terms as f64 * (3.0 + 2.0 * (mean_len - mean_lcp).max(1.0));
        log::info!(
            "[census] {label}: polys={} terms={} max_terms={} mean_terms={mean_terms:.1} mean_len={mean_len:.1} max_len={} mean_lcp={mean_lcp:.1} | bytes cur≈{:.0}M arena≈{:.0}M frontcoded≈{:.0}M",
            self.polys,
            self.terms,
            self.max_terms,
            self.max_len,
            cur / 1e6,
            arena / 1e6,
            fc / 1e6
        );
    }
}

/// Full data-shape census over the stores; `distinct` counting is skipped when
/// the store is too large to mirror in a HashSet.
pub fn run(tags: &Tags, cocycles: &Cocycles, pages: &E2) {
    if !enabled() {
        return;
    }

    let mut targets = PolyStats::new();
    let mut tagpolys = PolyStats::new();
    let mut coc = PolyStats::new();
    let mut distinct: HashSet<Monomial> = HashSet::new();
    let mut occurrences: u64 = 0;

    let mut packed_bytes: u64 = 0;
    for (_k, target, tag) in tags.iter_entries() {
        let (t, g) = (target.unpack(), tag.unpack());
        packed_bytes += (target.data.len() + tag.data.len()) as u64;
        occurrences += (t.monomials.len() + g.monomials.len()) as u64;
        targets.add(&t, Some(&mut distinct));
        tagpolys.add(&g, Some(&mut distinct));
    }
    for (_k, p) in cocycles.iter_entries() {
        let u = p.unpack();
        packed_bytes += p.data.len() as u64;
        occurrences += u.monomials.len() as u64;
        coc.add(&u, Some(&mut distinct));
    }
    log::info!("[census] packed store bytes = {:.0}M", packed_bytes as f64 / 1e6);

    log::info!("[census] tags entries={} cocycles entries={}", tags.index.len(), cocycles.index.len());
    targets.report("tags.target (= d(tag))");
    tagpolys.report("tags.tag");
    coc.report("cocycles");
    log::info!(
        "[census] distinct monomials={} / occurrences={} (interning ratio {:.2}x)",
        distinct.len(),
        occurrences,
        occurrences as f64 / distinct.len().max(1) as f64
    );

    // E2 duplication: every survivor is stored once per dimension key.
    let e2_total: u64 = pages.0.values().map(|v| v.len() as u64).sum();
    let mut e2_distinct: HashSet<&Monomial> = HashSet::new();
    for v in pages.0.values() {
        for m in v {
            e2_distinct.insert(m);
        }
    }
    let e2_index_sum: u64 = pages
        .0
        .values()
        .flat_map(|v| v.iter().map(|m| m.len() as u64))
        .sum();
    log::info!(
        "[census] E2: keys={} stored monomials={} distinct={} (dup factor {:.1}x) approx bytes cur≈{:.0}M",
        pages.0.len(),
        e2_total,
        e2_distinct.len(),
        e2_total as f64 / e2_distinct.len().max(1) as f64,
        (e2_total as f64 * 32.0 + e2_index_sum as f64 * 2.0) / 1e6
    );

    log::info!(
        "[census] get_tag: calls={} exact={} prefix={} special={} miss={}",
        TAG_CALLS.load(Ordering::Relaxed),
        TAG_EXACT.load(Ordering::Relaxed),
        TAG_PREFIX.load(Ordering::Relaxed),
        TAG_SPECIAL.load(Ordering::Relaxed),
        TAG_MISS.load(Ordering::Relaxed),
    );
}
