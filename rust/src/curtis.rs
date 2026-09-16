//! Curtis algorithm core: generate the admissible monomial basis of the
//! lambda algebra degree by degree, reduce each monomial's differential
//! against the accumulated tag table, and record the outcome — a cocycle
//! representative for each survivor, or a (target, tag) pair otherwise.
//! Also owns the "evens" additions (λ0 towers and even-dimension classes)
//! and the cross-degree [`CurtisState`] stepper with its checkpoint /
//! resume / filtration-extension machinery.
//!
//! Boundary conventions: E2 rows are keyed by `tri(n, s, f)` = (sphere
//! dimension, stem, filtration); a monomial of length ℓ sits in filtration
//! ℓ, total degree stem + ℓ. Monomial order is the lexicographic `Ord` of
//! [`Monomial`]; reductions descend through leading terms mod 2.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::data::{Cocycles, Tags, E2};
use crate::grading::tri;
use crate::io::csv::vectors_to_string;
use crate::mon::{Idx, Monomial};
use crate::packed::{PackedEncoder, PackedPoly, PackedView};
use crate::sweep::{drain_to_encoder, head_equals, Run};
use crate::poly::Poly;

/// Seed table for monomial generation, keyed by (stem, length, initial).
/// Deliberately NOT a `Trigrade` — the components mean different things than
/// the E2 page's (dimension, stem, filtration). Serializable so an
/// incremental run can checkpoint it (it is the only generation state that
/// cannot be cheaply re-derived at resume).
#[derive(Serialize, Deserialize)]
struct SeedTable(FxHashMap<(i32, i32, i32), Vec<Monomial>>);

impl SeedTable {
    fn new() -> Self {
        SeedTable(FxHashMap::default())
    }

    fn add(&mut self, key: (i32, i32, i32), mon: Monomial) {
        self.0.entry(key).or_default().push(mon);
    }

    fn basis(&self, key: (i32, i32, i32)) -> &[Monomial] {
        self.0.get(&key).map_or(&[], |v| v.as_slice())
    }
}

/// The "evens" additions (the λ0 towers and the even-dimension classes with
/// their tags), split by total degree so an incremental run can emit them in
/// step with the curtis frontier. Two halves with different consumers:
///
///   - [`add_evens_rows_degree`]: the E2 rows + cocycle representatives of
///     total degree exactly `u`. Safe on the PRODUCER side (curtis never
///     reads `pages` or `cocycles`).
///   - [`add_evens_tags_through`]: the tag pairs for degrees ≤ `max_tot`.
///     CONSUMER side only — inserting these into the live table would
///     perturb suffix resolution during monomial generation; the one-shot
///     path applies them only after curtis finishes, and an incremental run
///     applies them to each read-only snapshot instead.
///
/// [`add_evens`] folds the two halves and is exactly the historical one-shot
/// behavior.
pub fn add_evens(cocycles: &mut Cocycles, pages: &mut E2, tags: &mut Tags) {
    let max_filt = pages.max_filt();
    let max_dim = pages.max_dimension();
    let max_tot = pages.max_tot();

    for u in 0..=max_tot.max(max_filt) {
        add_evens_rows_degree(u, max_dim, max_filt, max_tot, cocycles, pages);
    }
    add_evens_tags_through(max_tot, max_dim, max_filt, tags);
}

/// Rows + cocycles of the evens additions at total degree exactly `u`; see
/// [`add_evens`]. `max_tot` gates the even-dimension classes the way the
/// one-shot pass does (a per-degree caller passes `max_tot >= u`).
pub fn add_evens_rows_degree(
    u: i32,
    max_dim: i32,
    max_filt: i32,
    max_tot: i32,
    cocycles: &mut Cocycles,
    pages: &mut E2,
) {
    add_evens_rows_degree_range(u, max_dim, -1, max_filt, max_tot, cocycles, pages);
}

/// Like [`add_evens_rows_degree`] but restricted to the evens content at
/// filtration strictly above `filt_lo` (and ≤ `max_filt`): a filtration
/// extension from cap F to F' backfills with `filt_lo = F`, adding only the
/// rows the old run did not already have.
pub fn add_evens_rows_degree_range(
    u: i32,
    max_dim: i32,
    filt_lo: i32,
    max_filt: i32,
    max_tot: i32,
    cocycles: &mut Cocycles,
    pages: &mut E2,
) {
    // λ0 towers at (dim, 0, u), one per dimension ≥ 2; row filtration is u.
    if u > filt_lo && u <= max_filt {
        for dim in 2..=max_dim {
            let zero_vector = vec![0; u as usize];
            pages.add(tri(dim, 0, u), Monomial::from(zero_vector.clone()));
            let zero_poly = Poly::from_monomial(zero_vector.clone());
            cocycles.add(Monomial::from(zero_vector), zero_poly);
        }
    }

    // Even-dimension classes at (dim, dim-1, 2+n): row degree dim+1+n = u,
    // row filtration 2+n.
    if u <= max_tot {
        for dim in (2..=max_dim).step_by(2) {
            let n = u - dim - 1;
            if n < 0 || n > max_filt - 2 || 2 + n <= filt_lo {
                continue;
            }
            let mut vector = vec![dim - 1, 0];
            vector.extend(vec![0; n as usize]);
            pages.add(tri(dim, dim - 1, 2 + n), Monomial::from(vector.clone()));

            // Cocycle representative: the differential of [dim, 0, ..., 0].
            let mut differential_vector = vec![dim];
            differential_vector.extend(vec![0; n as usize]);
            let differential_poly = Poly::from_monomial(differential_vector);
            cocycles.add(Monomial::from(vector), differential_poly.differential());
        }
    }
}

/// The tag half of the evens additions, for all total degrees ≤ `max_tot`;
/// see [`add_evens`]. Must only ever be applied to a table no longer used
/// for monomial generation (the sealed one-shot table, or a consumer-side
/// snapshot).
pub fn add_evens_tags_through(max_tot: i32, max_dim: i32, max_filt: i32, tags: &mut Tags) {
    for dim in (2..=max_dim).step_by(2) {
        for n in 0..=(max_filt - 2) {
            if dim + 1 + n > max_tot {
                break;
            }
            let mut vector = vec![dim - 1, 0];
            vector.extend(vec![0; n as usize]);
            let mut differential_vector = vec![dim];
            differential_vector.extend(vec![0; n as usize]);
            let differential_poly = Poly::from_monomial(differential_vector);
            let cocycle_poly = differential_poly.differential();
            tags.insert(Monomial::from(vector), (cocycle_poly, differential_poly));
        }
    }
}

/// The packed outcome of one tag reduction.
pub struct Reduced {
    /// The remainder of the differential (empty ⇔ `seed` is a cocycle);
    /// becomes the TARGET of a new tag pair otherwise.
    pub target: PackedPoly,
    /// The accumulated cocycle-so-far (seed + tags used).
    pub cocycle: PackedPoly,
    /// Number of tag resolutions consumed.
    pub steps: i32,
    /// Wall-clock time of the reduction, in milliseconds.
    pub millis: u128,
}

/// Reduce the differential of `seed` by tags, descending through its leading
/// terms; the tags used accumulate onto `seed` to form the cocycle.
///
/// Both outputs are produced by k-way parity merges emitted DIRECTLY into the
/// packed encoding (`merge-to-packed`): the accumulated pieces are each
/// already-sorted runs (prefix·stored-tag descriptors, plus the seed), so no
/// boxed monomials are materialized and no global sort happens — the peak
/// transient is the packed output buffer itself, not ~90 bytes per term.
pub fn reduce_differential(seed: Poly, tags: &Tags) -> Reduced {
    let start_time = Instant::now();
    let mut steps = 0;

    let mut heap: BinaryHeap<Run<'_>> = BinaryHeap::new();
    let init = seed.differential();
    if !init.monomials.is_empty() {
        heap.push(Run::owned(Vec::new(), init.monomials));
    }

    // Cocycle accumulation: (prefix, stored tag) descriptors + owned terms
    // from the synthesized special case; k-way merged at the end.
    let mut acc_parts: Vec<(Vec<Idx>, PackedView<'_>)> = Vec::new();
    let mut acc_owned: Vec<Monomial> = Vec::new();
    let mut target_enc = PackedEncoder::new();
    let mut lead: Vec<Idx> = Vec::new();

    while let Some(mut top) = heap.pop() {
        if top.head().is_none() {
            continue; // exhausted run
        }
        top.head_into(&mut lead);
        top.advance();
        if top.head().is_some() {
            heap.push(top);
        }

        // Cancel equal heads mod 2 across all runs.
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

        match tags.resolve_parts(&lead) {
            Some(crate::data::ResolvedParts::Stored {
                prefix_len,
                target,
                tag,
            }) => {
                // prefix·d(tag): streaming run over the stored target. Its lex
                // max is prefix ++ target.lead = the lead itself (the tag
                // contract) — skip that one copy so it does not bounce back.
                let prefix = lead[..prefix_len].to_vec();
                let mut run = Run::packed(prefix.clone(), target);
                debug_assert!(
                    head_equals(&run, &lead),
                    "tag differential must lead with the lead"
                );
                run.advance();
                if run.head().is_some() {
                    heap.push(run);
                }
                // d(prefix)·tag (Leibniz remainder) — prefix is short, this is small.
                if prefix_len > 0 {
                    let dprefix = Poly::from_terms(crate::lambda::leibniz(&Monomial::from(
                        prefix.clone(),
                    )));
                    let extra = dprefix.multiply(&tag.unpack());
                    if !extra.monomials.is_empty() {
                        heap.push(Run::owned(Vec::new(), extra.monomials));
                    }
                }
                acc_parts.push((prefix, tag));
                steps += 1;
            }
            Some(crate::data::ResolvedParts::Synthesized { tag }) => {
                // Tiny synthesized tag: materialize as before.
                let dtag = tag.differential();
                let mut rest = Vec::with_capacity(dtag.monomials.len().saturating_sub(1));
                let mut cancelled = false;
                for m in dtag.monomials {
                    if !cancelled && *m == *lead {
                        cancelled = true;
                    } else {
                        rest.push(m);
                    }
                }
                debug_assert!(cancelled, "synthesized tag differential must contain the lead");
                if !rest.is_empty() {
                    heap.push(Run::owned(Vec::new(), rest));
                }
                acc_owned.extend(tag.monomials);
                steps += 1;
            }
            None => {
                // `lead` survives: it and everything still pending is the
                // remainder — drain the live heap straight into the encoder.
                target_enc.push(&lead);
                drain_to_encoder(&mut heap, &mut target_enc, &mut lead);
                break;
            }
        }
    }

    // Merge-to-packed: k-way parity merge of the seed, the synthesized terms,
    // and every (prefix, stored tag) descriptor, emitted descending directly
    // into the packed encoding.
    let mut acc_heap: BinaryHeap<Run<'_>> = BinaryHeap::new();
    if !seed.monomials.is_empty() {
        acc_heap.push(Run::slice(Vec::new(), &seed.monomials));
    }
    if !acc_owned.is_empty() {
        acc_owned.sort_unstable();
        acc_heap.push(Run::owned(Vec::new(), acc_owned));
    }
    for (prefix, tag) in acc_parts {
        acc_heap.push(Run::packed(prefix, tag));
    }
    let mut cocycle_enc = PackedEncoder::new();
    drain_to_encoder(&mut acc_heap, &mut cocycle_enc, &mut lead);

    let duration = start_time.elapsed().as_millis();
    Reduced {
        target: target_enc.finish(),
        cocycle: cocycle_enc.finish(),
        steps,
        millis: duration,
    }
}

/// Candidate monomials at seed key `(stem, length, initial)`: prepend
/// `initial` to each seed at `(stem − initial, length − 1, h)` for every
/// head `h` in `0..=2·initial` — exactly the heads for which
/// `λ_initial · seed` is admissible (`2·initial ≥ h`). An untagged
/// candidate is always kept; a candidate whose suffix carries a tag is kept
/// only when the tag's lead head exceeds `2·initial` (the Curtis rejection
/// rule). Returns the kept candidates sorted lexicographically, or `None`
/// if there are none.
fn generate_next_monsx(
    stem: i32,
    length: i32,
    initial: i32,
    input_table: &SeedTable,
    tags: &Tags,
) -> Option<Vec<Monomial>> {
    if length <= 1 || stem < 0 || initial < 0 || stem < initial {
        return None;
    }

    let kept_mutex = Mutex::new(Vec::new());

    (0..=(2 * initial))
        .into_par_iter()
        .for_each(|i| {
            let search_key = (stem - initial, length - 1, 2 * initial - i);

            {
                let vec_list = input_table.basis(search_key);
                vec_list.par_iter().for_each(|vec| {
                    // Keep `initial · vec` unless its tag lead rules it out.
                    let keep = match tags.get_tag(vec) {
                        Some(tag) => tag
                            .lead_mon()
                            .map(|tag_lead| 2 * initial < tag_lead[0] as i32)
                            .unwrap_or(false),
                        None => true,
                    };
                    if keep {
                        kept_mutex
                            .lock()
                            .unwrap()
                            .push(vec.prepended(initial as crate::mon::Idx));
                    }
                });
            }
        });

    let mut kept = kept_mutex.into_inner().unwrap();
    kept.par_sort(); // lexicographic (Monomial's derived Ord)

    if kept.is_empty() {
        None
    } else {
        Some(kept)
    }
}

/// Aux checkpoint payload: everything besides the database index that a
/// resumed run needs — loop position, bounds, the store lengths as of the
/// checkpoint (resume truncates any partially-written suffix back to them),
/// and the seed table (the only generation state not cheaply re-derivable).
type CheckpointAux = (i32, i32, Option<i32>, bool, u64, u64, SeedTable);

/// The checkpoint aux metadata (everything but the seed table); see
/// [`CurtisState::read_checkpoint_meta`].
pub struct CheckpointMeta {
    pub next_deg: i32,
    pub dim_bound: i32,
    pub max_filt: Option<i32>,
    pub with_evens: bool,
    pub tags_store_len: u64,
    pub coc_store_len: u64,
}

/// Cross-degree state of the Curtis algorithm, advanced one total degree at
/// a time. [`curtis`] drives it 0..=N in one shot (byte-identical to the
/// historical single-function run); an incremental driver interleaves
/// [`CurtisState::step`] with [`CurtisState::checkpoint`] and
/// [`CurtisState::snapshot`] so read-only consumers can work on completed
/// degrees while the frontier advances.
pub struct CurtisState {
    pub tags: Tags,
    pub cocycles: Cocycles,
    pub pages: E2,
    mon_table: SeedTable,
    /// The next total degree `step` will run (frontier + 1).
    next_deg: i32,
    /// Exclusive upper bound of the survivor dimension fanout (the one-shot
    /// run uses 2·N). Must be ≥ 2·(highest degree ever stepped) + small
    /// margin for the resulting page to match a one-shot run at that degree.
    dim_bound: i32,
    max_filt: Option<i32>,
    /// Emit the per-degree evens rows/cocycles as each degree completes
    /// (incremental mode). Requires a filtration cap: the evens filtration
    /// bound must be fixed up front, not read off the finished table.
    with_evens: bool,
    db_dir: PathBuf,
    stem: String,
    // Optional debug traces (opt-in via --debug-logs): LTO.txt records each
    // surviving/tagged monomial with poly sizes and timing; tags.txt the tag pairs.
    writer: Option<BufWriter<std::fs::File>>,
    writer1: Option<BufWriter<std::fs::File>>,
    started: Instant,
}

impl CurtisState {
    /// Fresh state writing its stores under `db_dir` with filename stem
    /// `stem` (the one-shot path passes `io::cache::db_stem(..)`; an
    /// open-ended run picks its own, e.g. `run_f14`).
    pub fn create(
        db_dir: &Path,
        stem: &str,
        dim_bound: i32,
        max_filt: Option<i32>,
        with_evens: bool,
        debug_dir: Option<&Path>,
    ) -> crate::Result<CurtisState> {
        if with_evens && max_filt.is_none() {
            return Err("incremental evens emission requires --max-filt (the evens \
                 filtration bound must be fixed up front)"
                .into());
        }
        std::fs::create_dir_all(db_dir)?;
        let tags = Tags::create(&db_dir.join(format!("tags_{stem}_v4.store")))?;
        let cocycles = Cocycles::create(&db_dir.join(format!("cocycles_{stem}_v4.store")))?;
        let writer = match debug_dir {
            Some(dir) => Some(BufWriter::new(std::fs::File::create(dir.join("LTO.txt"))?)),
            None => None,
        };
        let writer1 = match debug_dir {
            Some(dir) => Some(BufWriter::new(std::fs::File::create(dir.join("tags.txt"))?)),
            None => None,
        };
        Ok(CurtisState {
            tags,
            cocycles,
            pages: E2::new(),
            mon_table: SeedTable::new(),
            next_deg: 0,
            dim_bound,
            max_filt,
            with_evens,
            db_dir: db_dir.to_path_buf(),
            stem: stem.to_string(),
            writer,
            writer1,
            started: Instant::now(),
        })
    }

    /// Highest fully-completed total degree (−1 before the first step).
    pub fn frontier(&self) -> i32 {
        self.next_deg - 1
    }

    pub fn dim_bound(&self) -> i32 {
        self.dim_bound
    }

    pub fn max_filt(&self) -> Option<i32> {
        self.max_filt
    }

    /// The aux half of a checkpoint minus the seed table — what external
    /// tooling (finalize/promote) needs to interpret a checkpoint.
    pub fn read_checkpoint_meta(
        db_dir: &Path,
        stem: &str,
        deg: i32,
    ) -> crate::Result<CheckpointMeta> {
        let f = std::fs::File::open(db_dir.join(format!("curtis_{stem}_deg{deg}_v4.aux")))?;
        let (next_deg, dim_bound, max_filt, with_evens, tags_store_len, coc_store_len, _seed): CheckpointAux =
            bincode::deserialize_from(std::io::BufReader::new(f))?;
        Ok(CheckpointMeta {
            next_deg,
            dim_bound,
            max_filt,
            with_evens,
            tags_store_len,
            coc_store_len,
        })
    }

    /// Highest degree with a complete (.idx + .aux) checkpoint pair under
    /// `db_dir` for `stem`, if any.
    pub fn latest_checkpoint(db_dir: &Path, stem: &str) -> crate::Result<Option<i32>> {
        let prefix = format!("curtis_{stem}_deg");
        let suffix = "_v4.aux";
        let mut best: Option<i32> = None;
        for entry in std::fs::read_dir(db_dir)? {
            let name = entry?.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(mid) = name.strip_prefix(&prefix).and_then(|r| r.strip_suffix(suffix))
            else {
                continue;
            };
            let Ok(d) = mid.parse::<i32>() else { continue };
            let has_idx = db_dir.join(format!("curtis_{stem}_deg{d}_v4.idx")).exists();
            if has_idx && best.is_none_or(|b| d > b) {
                best = Some(d);
            }
        }
        Ok(best)
    }

    fn tags_path(&self) -> PathBuf {
        self.db_dir.join(format!("tags_{}_v4.store", self.stem))
    }

    fn coc_path(&self) -> PathBuf {
        self.db_dir.join(format!("cocycles_{}_v4.store", self.stem))
    }

    fn idx_path(&self, deg: i32) -> PathBuf {
        self.db_dir.join(format!("curtis_{}_deg{deg}_v4.idx", self.stem))
    }

    fn aux_path(&self, deg: i32) -> PathBuf {
        self.db_dir.join(format!("curtis_{}_deg{deg}_v4.aux", self.stem))
    }

    /// The (n, initial, monomial) loops of one degree, restricted to lengths
    /// >= `len_start` (1 for a normal step; old_cap+1 for a filtration
    /// extension pass).
    fn run_degree_lens(&mut self, deg: i32, len_start: i32) -> crate::Result<()> {
        // Iterate through n from 0 to floor((deg + 4) / 3).
        for n in (len_start - 1)..=((deg + 4) / 3) {
            let len = n + 1;
            // Filtration cap: len ascends with n, so nothing above the cap is
            // ever generated (and, via the SeedTable, nothing longer can be
            // built from it in later degrees either).
            if self.max_filt.is_some_and(|cap| len > cap) {
                break;
            }
            let stem = deg - len;

            for initial in 0..=(deg - 2 * len + 2) {
                let monomials = if len == 1 {
                    // Length 1: the only candidate is λ_initial itself
                    // (odd index, stem = initial).
                    if initial % 2 == 1 && stem == initial {
                        vec![Monomial::from(vec![initial])]
                    } else {
                        vec![]
                    }
                } else {
                    generate_next_monsx(stem, len, initial, &self.mon_table, &self.tags)
                        .unwrap_or_default()
                };

                for mon in monomials {
                    match self.tags.resolves(&mon) {
                        // A resolved tag's lead always has length mon.len()-1
                        // (see Tags::resolves), so resolving is the whole check.
                        true => {
                            self.mon_table.add((stem, len, initial), mon.clone());
                        }
                        false => {
                            let red = reduce_differential(Poly::from_mon(mon.clone()), &self.tags);

                            if red.target.is_empty() {
                                self.mon_table.add((stem, len, initial), mon.clone());
                                if let Some(w) = self.writer.as_mut() {
                                    writeln!(
                                        w,
                                        "(({} {}) #({}) {})    {}",
                                        stem,
                                        len,
                                        vectors_to_string(&[mon.to_i32_vec()]),
                                        red.cocycle.len() - 1,
                                        red.millis
                                    )?;
                                }
                                let coc_lead =
                                    red.cocycle.lead().expect("survivor cocycle has a lead");
                                // Checkpoint replay and the C2 column both
                                // lean on a survivor's cocycle leading with
                                // the survivor itself.
                                debug_assert_eq!(
                                    coc_lead, mon,
                                    "survivor cocycle must lead with the survivor"
                                );
                                self.cocycles.add_packed(coc_lead.clone(), &red.cocycle);
                                let min = initial + 1;
                                log::debug!("{:?} #{:?}", deg, mon);

                                for dim in (min)..(self.dim_bound) {
                                    self.pages.add(tri(dim, stem, len), coc_lead.clone())
                                }
                            } else {
                                let tag = red.cocycle.lead().expect("tag poly has a lead");
                                let tar = red.target.lead().expect("target poly has a lead");

                                self.tags.insert_packed(tar.clone(), &red.target, &red.cocycle);

                                let min = tar[0] as i32 + 1;
                                let max = tag[0] as i32 + 1;
                                // A capped page must stay exact: at len = cap this
                                // row (filtration cap+1) would be partial — the
                                // length-(cap+1) survivors are never generated —
                                // so it is suppressed. The tag itself is still
                                // recorded below (needed to reduce other
                                // length-cap monomials).
                                if self.max_filt.is_none_or(|cap| len < cap) {
                                    for dim in (min)..(max) {
                                        self.pages.add(tri(dim, stem - 1, len + 1), tar.clone())
                                    }
                                }
                                log::debug!("{:?} #{:?} #{:?}", deg, tar, tag);

                                if let Some(w) = self.writer1.as_mut() {
                                    writeln!(
                                        w,
                                        "({}) ({})",
                                        vectors_to_string(&[tar.to_i32_vec()]),
                                        vectors_to_string(&[tag.to_i32_vec()]),
                                    )?;
                                }

                                if let Some(w) = self.writer.as_mut() {
                                    writeln!(
                                        w,
                                        "(({} {}) #({}) {} #({}) {})    {}",
                                        stem - 1,
                                        len + 1,
                                        vectors_to_string(&[tar.to_i32_vec()]),
                                        red.target.len() - 1,
                                        vectors_to_string(&[tag.to_i32_vec()]),
                                        red.cocycle.len() - 1,
                                        red.millis
                                    )?;
                                }
                                self.cocycles.add_packed(tar, &red.target);
                                self.cocycles.add_packed(tag, &red.cocycle);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Run one total degree (the verbatim body of the historical outer
    /// degree loop), returning the degree just completed.
    pub fn step(&mut self) -> crate::Result<i32> {
        let deg = self.next_deg;
        self.run_degree_lens(deg, 1)?;

        // Incremental mode: this degree's evens rows/cocycles (never the
        // tags — those would perturb later generation; consumers overlay
        // them onto snapshots via `add_evens_tags_through`).
        if self.with_evens {
            let cap = self.max_filt.expect("with_evens requires a cap (checked in create)");
            add_evens_rows_degree(
                deg,
                self.dim_bound - 1,
                cap,
                deg,
                &mut self.cocycles,
                &mut self.pages,
            );
        }

        log::info!(
            "[curtis] degree {deg} done  ({:.1}s elapsed, {} tags, {} cocycles)",
            self.started.elapsed().as_secs_f32(),
            self.tags.len(),
            self.cocycles.len(),
        );
        self.next_deg += 1;
        Ok(deg)
    }

    /// Seal the stores and write the per-degree checkpoint: an `.idx` file
    /// in exactly the one-shot database-index format (so a finished run can
    /// be promoted into a standard cache) plus an `.aux` file with the loop
    /// state. Both are written to a temp name and renamed.
    pub fn checkpoint(&mut self) -> crate::Result<()> {
        self.tags.seal()?;
        self.cocycles.seal()?;
        let d = self.frontier();

        let idx_path = self.idx_path(d);
        let tmp = idx_path.with_extension("idx.tmp");
        let f = std::fs::File::create(&tmp)?;
        bincode::serialize_into(
            std::io::BufWriter::new(f),
            &(&self.tags.index, &self.cocycles.index, &self.pages),
        )?;
        std::fs::rename(&tmp, &idx_path)?;

        let aux_path = self.aux_path(d);
        let tmp = aux_path.with_extension("aux.tmp");
        let f = std::fs::File::create(&tmp)?;
        bincode::serialize_into(
            std::io::BufWriter::new(f),
            &(
                self.next_deg,
                self.dim_bound,
                self.max_filt,
                self.with_evens,
                self.tags.store_len(),
                self.cocycles.store_len(),
                &self.mon_table,
            ),
        )?;
        std::fs::rename(&tmp, &aux_path)?;
        Ok(())
    }

    /// Read-only consumer view of the current frontier. Must follow a
    /// `checkpoint()` (or at least sealed stores): the returned tables mmap
    /// the store files privately and clone the in-RAM indexes/page.
    pub fn snapshot(&self) -> crate::Result<(Tags, Cocycles, E2)> {
        Ok((
            self.tags.snapshot(&self.tags_path())?,
            self.cocycles.snapshot(&self.coc_path())?,
            self.pages.clone(),
        ))
    }

    /// Resume from the highest complete checkpoint under `db_dir`/`stem`:
    /// truncates the stores back to the checkpointed lengths (discarding any
    /// partially-written suffix from a crash) and reopens them for appending.
    pub fn resume(db_dir: &Path, stem: &str) -> crate::Result<CurtisState> {
        let d = Self::latest_checkpoint(db_dir, stem)?
            .ok_or_else(|| format!("no checkpoint for stem {stem} in {}", db_dir.display()))?;

        let aux_file = std::fs::File::open(db_dir.join(format!("curtis_{stem}_deg{d}_v4.aux")))?;
        let (next_deg, dim_bound, max_filt, with_evens, tags_len, coc_len, mon_table): CheckpointAux =
            bincode::deserialize_from(std::io::BufReader::new(aux_file))?;
        let idx_file = std::fs::File::open(db_dir.join(format!("curtis_{stem}_deg{d}_v4.idx")))?;
        let (tags_index, coc_index, pages): (
            FxHashMap<Monomial, (crate::store::PolyRef, crate::store::PolyRef)>,
            FxHashMap<Monomial, crate::store::PolyRef>,
            E2,
        ) = bincode::deserialize_from(std::io::BufReader::new(idx_file))?;

        let tags_path = db_dir.join(format!("tags_{stem}_v4.store"));
        let coc_path = db_dir.join(format!("cocycles_{stem}_v4.store"));
        crate::store::PolyStore::truncate_to(&tags_path, tags_len)?;
        crate::store::PolyStore::truncate_to(&coc_path, coc_len)?;
        let tags = Tags::reopen_append(&tags_path, tags_index)?;
        let cocycles = Cocycles::reopen_append(&coc_path, coc_index)?;
        log::info!("[curtis] resumed at frontier {d} (next degree {next_deg})");

        Ok(CurtisState {
            tags,
            cocycles,
            pages,
            mon_table,
            next_deg,
            dim_bound,
            max_filt,
            with_evens,
            db_dir: db_dir.to_path_buf(),
            stem: stem.to_string(),
            writer: None,
            writer1: None,
            started: Instant::now(),
        })
    }

    /// Extend a checkpointed cap-F run to a higher filtration cap under a
    /// NEW stem, reusing all the old work: a cap-F table is exactly the
    /// length ≤ F slice of a cap-F' table (reducing a length-ℓ monomial only
    /// ever consumes tags produced by length ≤ ℓ monomials), so this copies
    /// the old stores/indexes and, for each already-completed degree, runs
    /// only the lengths F+1..=F' — plus two backfills the old run
    /// suppressed or capped:
    ///
    ///   - the page rows at filtration F+1 (targets of tags created by
    ///     length-F monomials; replayed in original creation order via the
    ///     append-only store offsets),
    ///   - the evens rows at filtration > F (with_evens runs).
    ///
    /// The result is identical to a straight cap-F' run at the same
    /// frontier, and the returned state steps/checkpoints/resumes normally.
    pub fn extend_from(
        old_db: &Path,
        old_stem: &str,
        db_dir: &Path,
        stem: &str,
        new_cap: i32,
    ) -> crate::Result<CurtisState> {
        let d0 = Self::latest_checkpoint(old_db, old_stem)?.ok_or_else(|| {
            format!("no checkpoint for stem {old_stem} in {}", old_db.display())
        })?;
        let meta = Self::read_checkpoint_meta(old_db, old_stem, d0)?;
        let old_cap = meta
            .max_filt
            .ok_or("the old run has no filtration cap; nothing to extend")?;
        if new_cap <= old_cap {
            return Err(format!(
                "new cap {new_cap} must exceed the old run's cap {old_cap}"
            )
            .into());
        }
        if old_stem == stem && old_db == db_dir {
            return Err("extension must use a new stem (the old stores are copied)".into());
        }
        std::fs::create_dir_all(db_dir)?;

        let tags_path = db_dir.join(format!("tags_{stem}_v4.store"));
        let coc_path = db_dir.join(format!("cocycles_{stem}_v4.store"));
        std::fs::copy(old_db.join(format!("tags_{old_stem}_v4.store")), &tags_path)?;
        std::fs::copy(
            old_db.join(format!("cocycles_{old_stem}_v4.store")),
            &coc_path,
        )?;
        crate::store::PolyStore::truncate_to(&tags_path, meta.tags_store_len)?;
        crate::store::PolyStore::truncate_to(&coc_path, meta.coc_store_len)?;

        let (tags_index, coc_index, pages) = crate::io::cache::read_db_index(
            &old_db.join(format!("curtis_{old_stem}_deg{d0}_v4.idx")),
        )?;
        let aux_file =
            std::fs::File::open(old_db.join(format!("curtis_{old_stem}_deg{d0}_v4.aux")))?;
        let (_, _, _, _, _, _, mon_table): CheckpointAux =
            bincode::deserialize_from(std::io::BufReader::new(aux_file))?;

        let mut st = CurtisState {
            tags: Tags::reopen_append(&tags_path, tags_index)?,
            cocycles: Cocycles::reopen_append(&coc_path, coc_index)?,
            pages,
            mon_table,
            next_deg: 0,
            dim_bound: meta.dim_bound,
            max_filt: Some(new_cap),
            with_evens: meta.with_evens,
            db_dir: db_dir.to_path_buf(),
            stem: stem.to_string(),
            writer: None,
            writer1: None,
            started: Instant::now(),
        };

        // Backfill material: every tag keyed at length old_cap+1 was created
        // by length-old_cap processing, whose page rows at filtration
        // old_cap+1 the capped run suppressed. Group by total degree, sort
        // by store offset (= original creation order).
        let mut by_deg: FxHashMap<i32, Vec<(u64, Monomial, Idx)>> = FxHashMap::default();
        for entry in st.tags.entries_by_key_len((old_cap + 1) as usize) {
            let deg: i32 = entry.1.iter().map(|&x| x as i32).sum::<i32>() + old_cap + 1;
            by_deg.entry(deg).or_default().push(entry);
        }
        for v in by_deg.values_mut() {
            v.sort_unstable_by_key(|e| e.0);
        }

        log::info!(
            "[curtis] extending {old_stem} (cap {old_cap}, frontier {d0}) to cap {new_cap} as {stem}"
        );
        for deg in 0..=d0 {
            if let Some(entries) = by_deg.get(&deg) {
                for (_, tar, tag_first) in entries {
                    let sigma: i32 = tar.iter().map(|&x| x as i32).sum();
                    let min = tar[0] as i32 + 1;
                    let max = *tag_first as i32 + 1;
                    for dim in min..max {
                        st.pages.add(tri(dim, sigma, old_cap + 1), tar.clone());
                    }
                }
            }
            st.run_degree_lens(deg, old_cap + 1)?;
            if st.with_evens {
                add_evens_rows_degree_range(
                    deg,
                    st.dim_bound - 1,
                    old_cap,
                    new_cap,
                    deg,
                    &mut st.cocycles,
                    &mut st.pages,
                );
            }
            if deg % 10 == 0 {
                log::info!(
                    "[curtis] extension through degree {deg}/{d0}  ({:.1}s elapsed)",
                    st.started.elapsed().as_secs_f32()
                );
            }
        }
        st.next_deg = d0 + 1;
        st.checkpoint()?;
        log::info!(
            "[curtis] extension complete at frontier {d0}  ({:.1}s)",
            st.started.elapsed().as_secs_f32()
        );
        Ok(st)
    }

    /// Flush the debug writers and hand back the tables.
    pub fn into_parts(mut self) -> crate::Result<(Tags, Cocycles, E2)> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
        }
        if let Some(w) = self.writer1.as_mut() {
            w.flush()?;
        }
        Ok((self.tags, self.cocycles, self.pages))
    }
}

/// One-shot Curtis run through total degree `degree`: create a fresh state
/// under `db_dir` (filename stem from `io::cache::db_stem`) and step every
/// degree `0..=degree`, with survivor dimension fanout `2·degree`.
///
/// With `max_filt = Some(F)` only monomials of length (= Adams filtration)
/// ≤ F are generated — the resulting E2 page is exact through filtration F,
/// since reducing a length-`len` monomial only ever consumes tags produced
/// by length-≤`len` monomials. Tags/cocycles are still recorded
/// unconditionally (a length-F monomial's tag, with target length F+1, is
/// needed to reduce OTHER length-F monomials); only the partial E2 rows at
/// filtration F+1 are suppressed.
pub fn curtis(
    degree: i32,
    debug_dir: Option<&Path>,
    db_dir: &Path,
    max_filt: Option<i32>,
) -> crate::Result<(Tags, Cocycles, E2)> {
    let stem = crate::io::cache::db_stem(degree, max_filt);
    let mut state = CurtisState::create(db_dir, &stem, 2 * degree, max_filt, false, debug_dir)?;
    for _ in 0..=degree {
        state.step()?;
    }
    state.into_parts()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lambda_e2_curtis_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    type PolyMap = FxHashMap<Monomial, Vec<Monomial>>;
    type PairMap = FxHashMap<Monomial, (Vec<Monomial>, Vec<Monomial>)>;

    fn tag_contents(t: &Tags) -> PairMap {
        t.iter_entries()
            .map(|(k, tar, tag)| (k.clone(), (tar.unpack().monomials, tag.unpack().monomials)))
            .collect()
    }

    fn cocycle_contents(c: &Cocycles) -> PolyMap {
        c.iter_entries()
            .map(|(k, v)| (k.clone(), v.unpack().monomials))
            .collect()
    }

    /// Stepping with a mid-run checkpoint + resume must reproduce the
    /// one-shot run's tables exactly (indexes, poly values, page).
    #[test]
    fn stepper_checkpoint_resume_matches_one_shot() {
        const DEG: i32 = 24;
        const CAP: i32 = 6;
        let dir_a = tmp_dir("oneshot");
        let (t1, c1, p1) = curtis(DEG, None, &dir_a, Some(CAP)).unwrap();

        let dir_b = tmp_dir("stepped");
        let mut st = CurtisState::create(&dir_b, "t", 2 * DEG, Some(CAP), false, None).unwrap();
        for _ in 0..=DEG / 2 {
            st.step().unwrap();
        }
        st.checkpoint().unwrap();
        // Poison the store suffix to prove resume truncates back to the
        // checkpointed lengths.
        st.step().unwrap();
        drop(st);

        let mut st = CurtisState::resume(&dir_b, "t").unwrap();
        assert_eq!(st.frontier(), DEG / 2);
        while st.frontier() < DEG {
            st.step().unwrap();
        }
        let (t2, c2, p2) = st.into_parts().unwrap();

        assert_eq!(p1, p2);
        assert_eq!(tag_contents(&t1), tag_contents(&t2));
        assert_eq!(cocycle_contents(&c1), cocycle_contents(&c2));

        std::fs::remove_dir_all(&dir_a).ok();
        std::fs::remove_dir_all(&dir_b).ok();
    }

    /// The split evens (per-degree rows + tags-through) must reproduce the
    /// historical one-shot `add_evens` exactly. The reference impl below is
    /// the pre-split code, verbatim.
    #[test]
    fn evens_split_matches_historical_add_evens() {
        const DEG: i32 = 16;
        const CAP: i32 = 6;
        let dir_a = tmp_dir("evens_new");
        let (mut t1, mut c1, mut p1) = curtis(DEG, None, &dir_a, Some(CAP)).unwrap();
        add_evens(&mut c1, &mut p1, &mut t1);

        let dir_b = tmp_dir("evens_ref");
        let (mut t2, mut c2, mut p2) = curtis(DEG, None, &dir_b, Some(CAP)).unwrap();
        historical_add_evens(&mut c2, &mut p2, &mut t2);

        assert_eq!(p1, p2);
        assert_eq!(tag_contents(&t1), tag_contents(&t2));
        assert_eq!(cocycle_contents(&c1), cocycle_contents(&c2));

        std::fs::remove_dir_all(&dir_a).ok();
        std::fs::remove_dir_all(&dir_b).ok();
    }

    /// Incremental with_evens stepping + the consumer-side tags overlay must
    /// also land on the same tables as one-shot curtis + add_evens.
    #[test]
    fn with_evens_stepping_matches_one_shot() {
        const DEG: i32 = 16;
        const CAP: i32 = 6;
        let dir_a = tmp_dir("we_oneshot");
        let (mut t1, mut c1, mut p1) = curtis(DEG, None, &dir_a, Some(CAP)).unwrap();
        add_evens(&mut c1, &mut p1, &mut t1);

        let dir_b = tmp_dir("we_stepped");
        let mut st = CurtisState::create(&dir_b, "t", 2 * DEG, Some(CAP), true, None).unwrap();
        for _ in 0..=DEG {
            st.step().unwrap();
        }
        let (mut t2, c2, p2) = st.into_parts().unwrap();
        add_evens_tags_through(DEG, 2 * DEG - 1, CAP, &mut t2);

        assert_eq!(p1, p2);
        assert_eq!(tag_contents(&t1), tag_contents(&t2));
        assert_eq!(cocycle_contents(&c1), cocycle_contents(&c2));

        std::fs::remove_dir_all(&dir_a).ok();
        std::fs::remove_dir_all(&dir_b).ok();
    }

    /// Extending a cap-4 run to cap 6 must reproduce the straight cap-6 run
    /// exactly — indexes, poly values, and the page including the
    /// backfilled filtration-5 rows in original creation order.
    #[test]
    fn filtration_extension_matches_straight_run() {
        const DEG: i32 = 24;
        let dir = tmp_dir("extend");
        let mut straight = CurtisState::create(&dir, "s6", 2 * DEG, Some(6), false, None).unwrap();
        for _ in 0..=DEG {
            straight.step().unwrap();
        }
        let (t1, c1, p1) = straight.into_parts().unwrap();

        let mut low = CurtisState::create(&dir, "s4", 2 * DEG, Some(4), false, None).unwrap();
        for _ in 0..=DEG {
            low.step().unwrap();
        }
        low.checkpoint().unwrap();
        drop(low);

        let st = CurtisState::extend_from(&dir, "s4", &dir, "x6", 6).unwrap();
        assert_eq!(st.frontier(), DEG);
        let (t2, c2, p2) = st.into_parts().unwrap();

        assert_eq!(p1, p2);
        assert_eq!(tag_contents(&t1), tag_contents(&t2));
        assert_eq!(cocycle_contents(&c1), cocycle_contents(&c2));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Same in with_evens (pipeline) mode, and the extended state must keep
    /// stepping correctly past the old frontier.
    #[test]
    fn filtration_extension_with_evens_and_continued_stepping() {
        const DEG: i32 = 20;
        const MORE: i32 = 4;
        let dir = tmp_dir("extend_evens");
        let mut straight =
            CurtisState::create(&dir, "s6", 2 * (DEG + MORE), Some(6), true, None).unwrap();
        for _ in 0..=(DEG + MORE) {
            straight.step().unwrap();
        }
        let (t1, c1, p1) = straight.into_parts().unwrap();

        let mut low =
            CurtisState::create(&dir, "s4", 2 * (DEG + MORE), Some(4), true, None).unwrap();
        for _ in 0..=DEG {
            low.step().unwrap();
        }
        low.checkpoint().unwrap();
        drop(low);

        let mut st = CurtisState::extend_from(&dir, "s4", &dir, "x6", 6).unwrap();
        while st.frontier() < DEG + MORE {
            st.step().unwrap();
        }
        let (t2, c2, p2) = st.into_parts().unwrap();

        assert_eq!(p1, p2);
        assert_eq!(tag_contents(&t1), tag_contents(&t2));
        assert_eq!(cocycle_contents(&c1), cocycle_contents(&c2));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Verbatim pre-split `add_evens`, kept as the reference for the
    /// equivalence test above.
    fn historical_add_evens(cocycles: &mut Cocycles, pages: &mut E2, tags: &mut Tags) {
        let max_filt = pages.max_filt();
        let max_dim = pages.max_dimension();

        for dim in 2..=max_dim {
            for n in 0..=max_filt {
                let zero_vector = vec![0; n as usize];
                pages.add(tri(dim, 0, n), Monomial::from(zero_vector.clone()));
                let zero_poly = Poly::from_monomial(zero_vector.clone());
                cocycles.add(Monomial::from(zero_vector), zero_poly);
            }
        }

        for dim in (2..=max_dim).step_by(2) {
            if dim > 0 {
                for n in 0..=(max_filt - 2) {
                    if dim + 1 + n > pages.max_tot() {
                        break;
                    }
                    let mut vector = vec![dim - 1, 0];
                    vector.extend(vec![0; n as usize]);
                    pages.add(tri(dim, dim - 1, 2 + n), Monomial::from(vector.clone()));

                    let mut differential_vector = vec![dim];
                    differential_vector.extend(vec![0; n as usize]);
                    let differential_poly = Poly::from_monomial(differential_vector);
                    let cocycle_poly = differential_poly.differential();
                    cocycles.add(Monomial::from(vector.clone()), cocycle_poly.clone());

                    let target_poly = cocycle_poly;
                    let tag_poly = differential_poly;
                    tags.insert(Monomial::from(vector), (target_poly, tag_poly));
                }
            }
        }
    }
}
