//! The curtis database (v4): curtis streams its poly stores to disk as it
//! runs; this module persists/reopens the accompanying indexes, making the
//! store files + index the durable, compact artifact that the operation
//! passes (products, H, P, C2) read from.

use std::path::Path;

use rustc_hash::FxHashMap;

use crate::curtis::{add_evens, curtis};
use crate::data::{Cocycles, Tags, E2};
use crate::mon::Monomial;
use crate::store::PolyRef;

/// Persisted database index (v4): the poly payloads live in the two
/// append-only store files written by curtis; this bincode blob carries the
/// in-RAM indexes and the E2 page. The incremental checkpoints written by
/// `CurtisState::checkpoint` use exactly this layout.
pub type DbIndex = (
    FxHashMap<Monomial, (PolyRef, PolyRef)>,
    FxHashMap<Monomial, PolyRef>,
    E2,
);

/// Read a v4 database index (a one-shot `curtis_*_v4.idx` or an incremental
/// checkpoint) from `path`.
pub fn read_db_index(path: &Path) -> crate::Result<DbIndex> {
    let f = std::fs::File::open(path)?;
    Ok(bincode::deserialize_from(std::io::BufReader::new(f))?)
}

/// Filename stem shared by the index and store files of one curtis run:
/// `deg{N}` for a full run, `deg{N}_f{F}` for a filtration-capped one, so a
/// truncated database can never shadow (or be shadowed by) a full one.
pub(crate) fn db_stem(degree: i32, max_filt: Option<i32>) -> String {
    match max_filt {
        Some(f) => format!("deg{degree}_f{f}"),
        None => format!("deg{degree}"),
    }
}

/// Run the Curtis algorithm for `degree`, streaming its poly stores to disk
/// under `cache_dir` as working storage, then apply the evens additions and
/// seal the stores. With `max_filt = Some(F)` the algorithm runs only for
/// Adams filtration ≤ F. The `.idx` layout is shared with the incremental
/// checkpoint/resume path (`CurtisState::checkpoint` / `read_db_index`).
pub fn compute_curtis(
    degree: i32,
    cache_dir: &Path,
    debug_dir: Option<&Path>,
    max_filt: Option<i32>,
) -> crate::Result<(Tags, Cocycles, E2)> {
    match max_filt {
        Some(f) => log::info!("Running curtis({degree}) capped at filtration {f}..."),
        None => log::info!("Running curtis({degree})..."),
    }
    let (mut t, mut c, mut p) = curtis(degree, debug_dir, cache_dir, max_filt)?;
    log::info!("Adding evens...");
    add_evens(&mut c, &mut p, &mut t);
    t.seal()?;
    c.seal()?;
    Ok((t, c, p))
}
