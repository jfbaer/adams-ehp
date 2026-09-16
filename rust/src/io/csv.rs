//! CSV writers and naming conventions — the contract with the SageMath
//! pipeline in `python/`.
//!
//! Class names are `"n_s_f"` when a bidegree has a single basis element and
//! `"n_s_f_i"` otherwise; `"0"` denotes the zero class. Sums of basis elements
//! are joined with `" + "`.

use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::data::E2;
use crate::grading::ClassId;
use crate::poly::to_i32s;

/// Render a sum of lambda words: each word's indices are space-joined, and
/// words are joined with `" + "` (also used by curtis' debug output).
pub fn vectors_to_string(vectors: &[Vec<i32>]) -> String {
    vectors
        .iter()
        .map(|vector| {
            // Join the components of each vector with a space
            vector
                .iter()
                .map(|&num| num.to_string())
                .collect::<Vec<String>>()
                .join(" ")
        })
        // Join each vector string with a "+" sign
        .collect::<Vec<String>>()
        .join(" + ")
}

/// Parse a class name `"n_s_f[_i]"` into a [`ClassId`] for sorting;
/// unparseable names (e.g. `"0"`) sort first as `ClassId::ZERO`.
pub fn parse_element_name(name: &str) -> ClassId {
    name.parse().unwrap_or(ClassId::ZERO)
}

/// Look up a class name as `names[n][vector]`, falling back to an
/// `"unknown_..."` marker (which the writers below filter out on the
/// source/factor side) when the vector is not in the dictionary.
pub fn get_name_from_vector(n: i32, vector: &Vec<i32>, names: &crate::Names) -> String {
    // Look up in names dictionary first
    if let Some(inner_map) = names.get(&n) {
        if let Some(name) = inner_map.get(vector) {
            return name.clone();
        }
    }

    // Fallback (shouldn't happen if names dictionary is complete)
    format!("unknown_{}_{:?}", n, vector)
}

/// Write `E2_rank.csv`: one `n,s,f,dimension` row per tridegree giving the
/// F_2-dimension of the E2 group there, sorted by (n, s, f).
pub fn write_rank_csv(pages: &E2, path: &Path) -> crate::Result<()> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);

    // Write header
    wtr.write_record(["n", "s", "f", "dimension"])?;

    // Collect and sort keys
    let mut keys: Vec<_> = pages.keys().collect();
    keys.sort();

    // Write data rows
    for &key in &keys {
        if let Some(vectors) = pages.0.get(key) {
            let rank = vectors.len();
            wtr.write_record(&[
                key.n.to_string(),
                key.s.to_string(),
                key.f.to_string(),
                rank.to_string(),
            ])?;
        }
    }

    wtr.flush()?;
    Ok(())
}

/// Build the class-name dictionary (pure; no I/O): `names[n][vector] = "n_s_f[_i]"`.
pub fn build_names(pages: &E2) -> crate::Names {
    let mut names = crate::Names::new();

    // Collect and sort keys for consistent ordering
    let mut keys: Vec<_> = pages.keys().collect();
    keys.sort();

    for &key in &keys {
        if let Some(vectors) = pages.0.get(key) {
            let vector_count = vectors.len();

            // Ensure the dimension n exists in the outer HashMap
            names.entry(key.n).or_default();

            for (i, compact_vector) in vectors.iter().enumerate() {
                let vector = to_i32s(compact_vector);

                let name = if vector_count == 1 {
                    format!("{}_{}_{}", key.n, key.s, key.f)
                } else {
                    format!("{}_{}_{}_{}", key.n, key.s, key.f, i)
                };

                // Insert into names[n][vector] = name
                names.get_mut(&key.n).unwrap().insert(vector, name);
            }
        }
    }

    names
}

/// Write the class-name dictionary as JSON (`E2_names.json`), ordered by
/// (n, s, f, i): maps each class name `"n_s_f[_i]"` to its admissible-monomial
/// representative as a space-separated lambda-index word. The python pipeline
/// loads this file alongside the CSVs for human-readable chart labels.
pub fn write_names_json(pages: &E2, names: &crate::Names, path: &Path) -> crate::Result<()> {
    // Collect and sort keys for consistent ordering
    let mut keys: Vec<_> = pages.keys().collect();
    keys.sort();

    // Create sorted entries for JSON serialization
    let mut all_entries: Vec<(i32, i32, i32, i32, Vec<i32>, String)> = Vec::new();

    // Collect all entries with their (n, s, f, i) coordinates
    for &key in &keys {
        if let Some(vectors) = pages.0.get(key) {
            for (i, compact_vector) in vectors.iter().enumerate() {
                let vector = to_i32s(compact_vector);
                if let Some(name) = names.get(&key.n).and_then(|inner| inner.get(&vector)) {
                    all_entries.push((key.n, key.s, key.f, i as i32, vector, name.clone()));
                }
            }
        }
    }

    // Sort by (n, s, f, i)
    all_entries.sort_by_key(|(n, s, f, i, _, _)| (*n, *s, *f, *i));

    // Convert to ordered JSON structure: name -> space-joined lambda word
    let mut json_names = serde_json::Map::new();
    for (_n, _s, _f, _i, vector, name) in all_entries {
        let word = vector
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        json_names.insert(name, serde_json::Value::String(word));
    }

    // Save dictionary as JSON
    let json_data = serde_json::to_string_pretty(&json_names)?;
    std::fs::write(path, json_data)?;

    Ok(())
}

/// Write the E (suspension) map CSV (`E2_E.csv`): `element,image` rows with
/// target dimension n + 1. Rows with zero/unknown sources and zero images
/// are omitted; output is sorted by source (n, s, f, i).
pub fn write_e_csv(
    results: &crate::OpResults,
    names: &crate::Names,
    path: &Path,
) -> crate::Result<()> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(["element", "image"])?;

    let mut all_entries = Vec::new();

    for (dim, dim_results) in results {
        for (vector, result_vectors) in dim_results {
            let element_name = get_name_from_vector(*dim, vector, names);

            // Skip if element name is "0" or unknown
            if element_name == "0" || element_name.starts_with("unknown_") {
                continue;
            }

            // Skip if result is empty (no suspension)
            if result_vectors.is_empty() {
                continue;
            }

            let target_dim = dim + 1; // E operation: suspension to next dimension
            let result_names: Vec<String> = result_vectors
                .iter()
                .map(|result_vec| get_name_from_vector(target_dim, result_vec, names))
                .collect();
            let result_name = result_names.join(" + ");

            // Skip if result is "0"
            if result_name == "0" {
                continue;
            }

            // Parse element name to get (n, s, f, i) for sorting
            let coords = parse_element_name(&element_name);
            all_entries.push((coords, element_name, result_name));
        }
    }

    // Sort by (n, s, f, i)
    all_entries.sort_by_key(|(coords, _, _)| *coords);

    for (_, element_name, result_name) in all_entries {
        wtr.write_record(&[element_name, result_name])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Write the H (Hopf) map CSV (`E2_H.csv`): `element,image` rows with target
/// dimension 2n - 1. Rows with zero/unknown sources and zero images are
/// omitted; output is sorted by source (n, s, f, i).
pub fn write_h_csv(
    results: &crate::OpResults,
    names: &crate::Names,
    path: &Path,
) -> crate::Result<()> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(["element", "image"])?;

    let mut all_entries = Vec::new();

    for (dim, dim_results) in results {
        for (vector, result_vectors) in dim_results {
            let element_name = get_name_from_vector(*dim, vector, names);

            // Skip if element name is "0" or unknown
            if element_name == "0" || element_name.starts_with("unknown_") {
                continue;
            }

            // Skip if result is empty
            if result_vectors.is_empty() {
                continue;
            }

            let target_dim = 2 * dim - 1; // H operation: Hopf map dimension
            let result_names: Vec<String> = result_vectors
                .iter()
                .map(|result_vec| get_name_from_vector(target_dim, result_vec, names))
                .collect();
            let result_name = result_names.join(" + ");

            // Skip if result is "0"
            if result_name == "0" {
                continue;
            }

            // Parse element name to get (n, s, f, i) for sorting
            let coords = parse_element_name(&element_name);
            all_entries.push((coords, element_name, result_name));
        }
    }

    // Sort by (n, s, f, i)
    all_entries.sort_by_key(|(coords, _, _)| *coords);

    for (_, element_name, result_name) in all_entries {
        wtr.write_record(&[element_name, result_name])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Write the P (Whitehead product) map CSV (`E2_P.csv`): `element,image` rows
/// with target dimension (n - 1) / 2. Rows with zero/unknown sources and
/// zero images are omitted; output is sorted by source (n, s, f, i).
pub fn write_p_csv(
    results: &crate::OpResults,
    names: &crate::Names,
    path: &Path,
) -> crate::Result<()> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(["element", "image"])?;

    let mut all_entries = Vec::new();

    for (dim, dim_results) in results {
        for (vector, result_vectors) in dim_results {
            let element_name = get_name_from_vector(*dim, vector, names);

            // Skip if element name is "0" or unknown
            if element_name == "0" || element_name.starts_with("unknown_") {
                continue;
            }

            // Skip if result is empty
            if result_vectors.is_empty() {
                continue;
            }

            let target_dim = (dim - 1) / 2; // P operation: target dimension
            let result_names: Vec<String> = result_vectors
                .iter()
                .map(|result_vec| get_name_from_vector(target_dim, result_vec, names))
                .collect();
            let result_name = result_names.join(" + ");

            // Skip if result is "0"
            if result_name == "0" {
                continue;
            }

            // Parse element name to get (n, s, f, i) for sorting
            let coords = parse_element_name(&element_name);
            all_entries.push((coords, element_name, result_name));
        }
    }

    // Sort by (n, s, f, i)
    all_entries.sort_by_key(|(coords, _, _)| *coords);

    for (_, element_name, result_name) in all_entries {
        wtr.write_record(&[element_name, result_name])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Append one batch of product rows (`factor1,factor2,result` columns) to the
/// relations CSV. The caller (`maps/products.rs`) writes the header once and
/// then appends per-(stem, filtration) batches, so each batch is sorted by
/// (factor1, factor2) coordinates within itself. Rows with a zero/unknown
/// factor or a zero product are skipped.
pub fn write_decompositions_csv_append(
    results: &crate::ProductTable,
    names: &crate::Names,
    path: &Path,
) -> crate::Result<()> {
    let file = OpenOptions::new().append(true).open(path)?;
    let mut wtr = csv::Writer::from_writer(file);

    let mut all_entries = Vec::new();

    for (target_dim, dim_results) in results {
        for ((element_vector, source_vector), product_vectors) in dim_results {
            let element_name = get_name_from_vector(*target_dim, element_vector, names);

            // Calculate source dimension: n + s where n = target_dim and s = sum of element_vector
            let element_stem: i32 = element_vector.iter().sum();
            let source_dim = target_dim + element_stem;
            let source_name = get_name_from_vector(source_dim, source_vector, names);

            // Skip if element or source name is "0" or unknown
            if element_name == "0"
                || element_name.starts_with("unknown_")
                || source_name == "0"
                || source_name.starts_with("unknown_")
            {
                continue;
            }

            // Skip if result is empty
            if product_vectors.is_empty() {
                continue;
            }

            let result_names: Vec<String> = product_vectors
                .iter()
                .map(|result_vec| get_name_from_vector(*target_dim, result_vec, names))
                .collect();
            let result_name = result_names.join(" + ");

            // Skip if result is "0"
            if result_name == "0" {
                continue;
            }

            // Parse element name to get (n, s, f, i) for sorting
            let element_coords = parse_element_name(&element_name);
            let source_coords = parse_element_name(&source_name);
            all_entries.push((
                element_coords,
                source_coords,
                element_name,
                source_name,
                result_name,
            ));
        }
    }

    // Sort by element coordinates first, then source coordinates
    all_entries
        .sort_by_key(|(element_coords, source_coords, _, _, _)| (*element_coords, *source_coords));

    for (_, _, element_name, source_name, result_name) in all_entries {
        wtr.write_record(&[element_name, source_name, result_name])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Write a coordinate-keyed operation map (used for `E2_C2.csv`):
/// `element,image` rows sorted by source coordinates, with image summands
/// joined by `" + "`.
pub fn write_operation_csv(
    results: &crate::CoordMap,
    pages: &E2,
    path: &Path,
) -> crate::Result<()> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(["element", "image"])?;

    // Sort entries for consistent output
    let mut entries: Vec<_> = results.iter().collect();
    entries.sort_by_key(|(element_coords, _)| *element_coords);

    for (element_coords, result_coords) in entries {
        let element_name = pages.name_from_coords(*element_coords);
        let result_names: Vec<String> = result_coords
            .iter()
            .map(|coords| pages.name_from_coords(*coords))
            .collect();
        let result_name = result_names.join(" + ");

        wtr.write_record(&[element_name, result_name])?;
    }

    wtr.flush()?;
    Ok(())
}
