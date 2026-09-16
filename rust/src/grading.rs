//! Grading types: `Trigrade` (sphere dimension n, stem s, filtration f) and
//! `ClassId` (a trigrade plus the index of a basis element within it).
//!
//! `ClassId` is the "name" of an E2 class; its `Display`/`FromStr` implement
//! the CSV naming convention `"n_s_f"` / `"n_s_f_i"` (`"0"` = the zero class,
//! represented by `ClassId::ZERO`).

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// (dimension n, stem s, filtration f).
// FIELD ORDER FROZEN: bincode cache compat (encodes identically to the old
// (i32, i32, i32) tuple key of E2).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct Trigrade {
    pub n: i32,
    pub s: i32,
    pub f: i32,
}

/// Terse constructor: `tri(n, s, f)`.
pub const fn tri(n: i32, s: i32, f: i32) -> Trigrade {
    Trigrade { n, s, f }
}

impl Trigrade {
    /// Suspend `k` dimensions (same stem and filtration).
    pub const fn suspend(self, k: i32) -> Self {
        Trigrade {
            n: self.n + k,
            ..self
        }
    }
}

/// A basis element of the E2 page: its trigrade plus index within the basis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct ClassId {
    pub grade: Trigrade,
    pub index: i32,
}

impl ClassId {
    /// The zero-class sentinel. Deliberately an impossible coordinate:
    /// the tridegree (0, 0, 0) hosts a real class (the bottom-cell
    /// fundamental of the Lambda(C2) column), so the historical (0,0,0,0)
    /// encoding collided with it — silently filtering that class's rows
    /// out of every writer.
    pub const ZERO: ClassId = ClassId {
        grade: tri(-1, -1, -1),
        index: -1,
    };

    pub const fn new(n: i32, s: i32, f: i32, index: i32) -> Self {
        ClassId {
            grade: tri(n, s, f),
            index,
        }
    }
}

impl fmt::Display for ClassId {
    /// `"n_s_f_i"`. (The CSV writers drop the index when a bidegree has a
    /// single basis element — that variant needs the basis, so it lives on
    /// `E2::name_from_coords`.)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.grade;
        write!(f, "{}_{}_{}_{}", t.n, t.s, t.f, self.index)
    }
}

impl FromStr for ClassId {
    type Err = ();

    /// Parse `"n_s_f"` or `"n_s_f_i"`; `"0"` (or anything malformed) is ZERO —
    /// matching the tolerant behavior of the old `parse_element_name`.
    fn from_str(name: &str) -> Result<Self, ()> {
        if name == "0" {
            return Ok(ClassId::ZERO);
        }
        let parts: Vec<i32> = name.split('_').map(|p| p.parse().unwrap_or(0)).collect();
        match parts.len() {
            3 => Ok(ClassId::new(parts[0], parts[1], parts[2], 0)),
            4 => Ok(ClassId::new(parts[0], parts[1], parts[2], parts[3])),
            _ => Ok(ClassId::ZERO),
        }
    }
}
