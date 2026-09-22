//! Append-only on-disk store of packed polynomials — the database that
//! curtis writes and the operation passes read.
//!
//! Layout: raw concatenated `PackedPoly` payloads; a [`PolyRef`] (offset +
//! term count + byte length) addresses one poly. Writes go to an in-RAM tail
//! buffer that is flushed to the file and re-mmapped once it exceeds a
//! threshold, so a poly is always entirely in the mmap or entirely in the
//! tail — reads during the (single-writer) curtis run see everything written
//! so far, and after [`PolyStore::seal`] the store is a plain read-only mmap
//! safe to share across rayon workers.

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use crate::packed::{PackedPoly, PackedView};

/// Flush/remap threshold for the tail buffer.
const TAIL_FLUSH_BYTES: usize = 32 * 1024 * 1024;

/// Reference to one packed poly inside a [`PolyStore`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PolyRef {
    pub off: u64,
    pub n_terms: u32,
    pub data_len: u32,
}

pub struct PolyStore {
    file: File,
    mmap: Option<Mmap>,
    mapped_len: u64,
    tail: Vec<u8>,
    /// A frozen store never writes to its file: appends stay in the tail
    /// forever (no threshold flush) and `seal`/`flush_remap` are no-ops.
    /// Used for consumer-side snapshots that share a growing producer file
    /// but need small private overlays (e.g. the evens tags).
    frozen: bool,
}

impl std::fmt::Debug for PolyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolyStore")
            .field("mapped_len", &self.mapped_len)
            .field("tail_len", &self.tail.len())
            .finish()
    }
}

impl PolyStore {
    /// Create (truncating) a store for writing.
    pub fn create(path: &Path) -> crate::Result<PolyStore> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(PolyStore {
            file,
            mmap: None,
            mapped_len: 0,
            tail: Vec::new(),
            frozen: false,
        })
    }

    /// Open an existing (sealed) store for reading. (The file handle is
    /// opened read+write — `open_append` reuses this path for resume — but
    /// nothing is written unless the caller appends.)
    pub fn open(path: &Path) -> crate::Result<PolyStore> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        let mmap = if len > 0 {
            // Safety: the store is append-only and sealed stores are not
            // mutated; single-process access by construction.
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };
        Ok(PolyStore {
            file,
            mmap,
            mapped_len: len,
            tail: Vec::new(),
            frozen: false,
        })
    }

    /// Open a snapshot view of a store whose file another `PolyStore` may
    /// still be appending to. The mmap covers the file's length as of now
    /// (append-only, so every existing offset stays valid); the file itself
    /// is opened read-only and is never written through this handle. Appends
    /// are accepted but live only in this store's private in-RAM tail.
    pub fn open_frozen(path: &Path) -> crate::Result<PolyStore> {
        let file = OpenOptions::new().read(true).open(path)?;
        let len = file.metadata()?.len();
        let mmap = if len > 0 {
            // Safety: the shared file is append-only; bytes below `len` are
            // immutable for the lifetime of this map.
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };
        Ok(PolyStore {
            file,
            mmap,
            mapped_len: len,
            tail: Vec::new(),
            frozen: true,
        })
    }

    /// Reopen a store for continued appending (resume): like `open`, but the
    /// file cursor is moved to the end so the next flush appends instead of
    /// clobbering byte 0.
    pub fn open_append(path: &Path) -> crate::Result<PolyStore> {
        use std::io::{Seek, SeekFrom};
        let mut store = Self::open(path)?;
        store.file.seek(SeekFrom::End(0))?;
        Ok(store)
    }

    /// Truncate the store file at `path` to `len` bytes, discarding a partial
    /// suffix (resume-after-crash discards bytes past the last checkpoint).
    pub fn truncate_to(path: &Path, len: u64) -> crate::Result<()> {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(len)?;
        Ok(())
    }

    /// Bytes currently buffered in the in-RAM tail (0 on a sealed store).
    pub fn tail_len(&self) -> usize {
        self.tail.len()
    }

    pub fn len(&self) -> u64 {
        self.mapped_len + self.tail.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a packed poly, returning its reference.
    pub fn append(&mut self, poly: &PackedPoly) -> PolyRef {
        let (n_terms, data) = poly.raw();
        let r = PolyRef {
            off: self.len(),
            n_terms,
            data_len: data.len() as u32,
        };
        self.tail.extend_from_slice(data);
        if !self.frozen && self.tail.len() >= TAIL_FLUSH_BYTES {
            self.flush_remap().expect("poly store flush failed");
        }
        r
    }

    /// Borrowed view of a stored poly. Valid while `self` is not appended to
    /// (the borrow checker enforces this: `append` takes `&mut self`).
    pub fn view(&self, r: PolyRef) -> PackedView<'_> {
        let data: &[u8] = if r.off >= self.mapped_len {
            let start = (r.off - self.mapped_len) as usize;
            &self.tail[start..start + r.data_len as usize]
        } else {
            let m = self.mmap.as_ref().expect("mapped region must exist");
            &m[r.off as usize..(r.off + r.data_len as u64) as usize]
        };
        PackedView {
            n_terms: r.n_terms,
            data,
        }
    }

    fn flush_remap(&mut self) -> crate::Result<()> {
        // A frozen store must never write to the (shared) file; its tail is a
        // private overlay that simply stays in RAM.
        if self.frozen || self.tail.is_empty() {
            return Ok(());
        }
        self.file.write_all(&self.tail)?;
        self.file.flush()?;
        self.mapped_len += self.tail.len() as u64;
        self.tail.clear();
        // Safety: see `open`.
        self.mmap = Some(unsafe { Mmap::map(&self.file)? });
        Ok(())
    }

    /// Flush everything; afterwards all reads are pure mmap (Sync across
    /// rayon workers).
    pub fn seal(&mut self) -> crate::Result<()> {
        self.flush_remap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mon;
    use crate::poly::Poly;

    #[test]
    fn store_roundtrip_and_tail_reads() {
        let dir = std::env::temp_dir().join("lambda_e2_store_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.store");
        let mut store = PolyStore::create(&path).unwrap();

        let mut p1 = Poly::new();
        p1.toggle(mon![2, 4, 1]);
        p1.toggle(mon![3]);
        let mut p2 = Poly::new();
        p2.toggle(mon![0]);

        let r1 = store.append(&PackedPoly::pack(&p1));
        let r2 = store.append(&PackedPoly::pack(&p2));

        // reads from the tail (nothing flushed yet)
        assert_eq!(store.view(r1).unpack().monomials, p1.monomials);
        assert_eq!(store.view(r2).unpack().monomials, p2.monomials);

        // reads after seal (mmap)
        store.seal().unwrap();
        assert_eq!(store.view(r1).unpack().monomials, p1.monomials);
        assert_eq!(store.view(r2).unpack().monomials, p2.monomials);

        // reopen
        let store2 = PolyStore::open(&path).unwrap();
        assert_eq!(store2.view(r1).unpack().monomials, p1.monomials);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frozen_store_never_writes_the_file() {
        let dir = std::env::temp_dir().join("lambda_e2_store_frozen_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.store");

        let mut writer = PolyStore::create(&path).unwrap();
        let mut p1 = Poly::new();
        p1.toggle(mon![2, 4, 1]);
        let r1 = writer.append(&PackedPoly::pack(&p1));
        writer.seal().unwrap();
        let disk_len = std::fs::metadata(&path).unwrap().len();

        // A frozen view sees the sealed contents and accepts private appends
        // without ever growing the shared file, even across seal().
        let mut frozen = PolyStore::open_frozen(&path).unwrap();
        assert_eq!(frozen.view(r1).unpack().monomials, p1.monomials);
        let mut p2 = Poly::new();
        p2.toggle(mon![0]);
        let r2 = frozen.append(&PackedPoly::pack(&p2));
        assert_eq!(frozen.view(r2).unpack().monomials, p2.monomials);
        frozen.seal().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), disk_len);
        assert_eq!(frozen.tail_len() as u64, frozen.len() - disk_len);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_append_extends_instead_of_clobbering() {
        let dir = std::env::temp_dir().join("lambda_e2_store_append_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.store");

        let mut writer = PolyStore::create(&path).unwrap();
        let mut p1 = Poly::new();
        p1.toggle(mon![2, 4, 1]);
        let r1 = writer.append(&PackedPoly::pack(&p1));
        writer.seal().unwrap();
        drop(writer);

        let mut resumed = PolyStore::open_append(&path).unwrap();
        let mut p2 = Poly::new();
        p2.toggle(mon![3]);
        let r2 = resumed.append(&PackedPoly::pack(&p2));
        resumed.seal().unwrap();

        // Both the pre-existing poly and the resumed append survive on disk.
        let reread = PolyStore::open(&path).unwrap();
        assert_eq!(reread.view(r1).unpack().monomials, p1.monomials);
        assert_eq!(reread.view(r2).unpack().monomials, p2.monomials);

        // truncate_to drops the resumed suffix again.
        PolyStore::truncate_to(&path, r2.off).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), r2.off);

        std::fs::remove_dir_all(&dir).ok();
    }
}
