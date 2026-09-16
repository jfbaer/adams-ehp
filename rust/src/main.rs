//! Command-line interface for the lambda-algebra E2 pipeline.
//!
//! All mathematics lives in the `lambda_e2` library; this binary only parses
//! arguments, resolves paths, and dispatches. See `lambda_e2` crate docs for
//! the pipeline and the CSV contract with the SageMath code in `python/`.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Args, Parser, Subcommand};

use lambda_e2::grading::ClassId;
use lambda_e2::io::cache::load_or_compute_curtis;
use lambda_e2::io::csv::{
    build_names, write_e_csv, write_h_csv, write_names_json, write_operation_csv, write_p_csv,
    write_rank_csv,
};
use lambda_e2::maps::c2::{
    compute_all, compute_c2_e2, compute_c2_images, compute_c2_products, write_c2_names_json,
    write_c2_rank_csv, SourceFilter,
};
use lambda_e2::maps::hopf_p::{compute_e, compute_hopf, compute_p};
use lambda_e2::maps::products::mult_table;
use lambda_e2::verify;

/// Compute the E2 page of lambda-algebra spectral sequences (Curtis algorithm)
/// and operations on it: products, Hopf H, suspension E, Whitehead P, and
/// Mahowald's map to Lambda(C2). Writes CSVs consumed by the SageMath pipeline.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,

    #[command(flatten)]
    opts: Opts,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run only the Curtis algorithm, writing the on-disk database
    /// (tags/cocycles poly stores + index) that the other commands read
    Curtis,
    /// Full pipeline: E2_relations, E2_H, E2_E, E2_P, E2_C2, E2_C2_products, E2_rank + names JSON
    Generate,
    /// Multiplication table only: E2_relations.csv (+ names JSON, E2_rank.csv)
    Products,
    /// Hopf H map only: E2_H.csv (+ names JSON, E2_rank.csv)
    H,
    /// Whitehead P map only: E2_P.csv (+ names JSON, E2_rank.csv). Computes the
    /// H map internally (P is defined in terms of it) but writes only E2_P.csv.
    P,
    /// Map to Lambda(C2) only: E2_C2.csv + E2_rank.csv (skips the slow mult/H/P steps)
    C2,
    /// Full Lambda(C2) pipeline: the C2 map + filtration-1 products + the
    /// rank/names of the Lambda(C2) homology (E2_C2.csv, E2_C2_products.csv,
    /// E2_C2_rank.csv, E2_C2_names.json)
    C2All,
    /// Lambda(C2) filtration-1 products + rank/names only, SKIPPING the slow
    /// odd-spheres→C2 map (E2_C2_products.csv, E2_C2_rank.csv, E2_C2_names.json)
    C2Products,
    /// Lambda(C2) rank/names ONLY (E2_C2_rank.csv, E2_C2_names.json + sphere
    /// E2_names.json/E2_rank.csv): builds the n = 0 column and dumps its basis
    /// dictionary without computing the map or products. Recovers the basis of
    /// a run that died before its end-of-run names write — the column is a
    /// deterministic function of the Curtis run at a given degree, so
    /// recomputing at the same degree/--max-filt reproduces the same basis
    /// the dead run's CSVs were named against
    C2Names,
    /// Internal cross-check of the C2 map against a direct F2-homology
    /// computation; no CSV output (hidden from help; kept for development)
    #[command(hide = true)]
    VerifyC2,
}

#[derive(Args)]
struct Opts {
    /// Total-degree bound s+f [default: 70]
    #[arg(short, long, global = true)]
    degree: Option<i32>,

    /// Directory for output CSVs
    #[arg(short, long, global = true, default_value = ".")]
    out_dir: PathBuf,

    /// Directory for the Curtis working store files (tags/cocycles
    /// `.store`) that every run streams to disk [default: --out-dir]
    #[arg(long, global = true)]
    cache_dir: Option<PathBuf>,

    /// Also write the curtis debug traces (LTO.txt, tags.txt) to --out-dir
    #[arg(long, global = true)]
    debug_logs: bool,

    /// Only compute E2_relations rows with total degree s+f above this floor
    /// (for incrementally extending an existing table; 0 = full table)
    #[arg(long, global = true, default_value_t = 0)]
    mult_floor: i32,

    /// Only compute E2_relations rows with total degree s+f at or below this
    /// ceiling (use with --mult-floor to compute a band). Measured behavior
    /// (2026-08-18, deg-50/65/July cross-comparison): emitted rows are always
    /// correct and the basis is identical across runs; deeper tables only ADD
    /// rows (fanout, factor-column <= degree, h0-tower depth). Only rows
    /// landing AT the table degree (margin 0) come out genuinely partial —
    /// set the ceiling at least 1 below --degree. Default: no ceiling.
    #[arg(long, global = true)]
    mult_ceil: Option<i32>,

    /// Run the full Curtis algorithm only up to this Adams filtration (the E2
    /// page is exact through it; the C2 outputs are clamped to their
    /// provably-complete range). Accepted by curtis, c2, c2-all and
    /// c2-products; capped databases are cached under their own key
    /// (curtis_deg{N}_f{F}_v4.*)
    #[arg(long, global = true)]
    max_filt: Option<i32>,

    /// Resume an interrupted c2-all / c2-products run: classes already present
    /// in the out-dir's E2_C2_products.csv (or its .done sidecar) are skipped
    /// and rows are appended; the Mahowald-map pass likewise skips classes in
    /// E2_C2.done. Safe on a fresh out-dir (behaves like a normal run). The
    /// curtis table must be the SAME cache — resume never revalidates the basis
    #[arg(long, global = true)]
    resume: bool,

    /// Restrict the odd-spheres→C2 map to source classes on this sphere
    /// dimension only (e.g. 3 for S³). The Λ(C2) column is still built in
    /// full; only the per-class map pass is narrowed. Accepted by c2, c2-all
    /// and generate
    #[arg(long, global = true)]
    sphere: Option<i32>,

    /// Restrict the odd-spheres→C2 map to source classes with stem ≤ this
    /// bound (combines with --sphere to select a rectangle of classes)
    #[arg(long, global = true)]
    max_stem: Option<i32>,

    /// Increase log verbosity (-v: per-monomial/per-target progress)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Log warnings and errors only
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    quiet: bool,
}

/// Rewrite the pre-CLI invocation forms so existing scripts keep working:
///   `lambda_e2 --c2-only [N]`   -> `lambda_e2 c2 [-d N]`
///   `lambda_e2 --verify-c2 [N]` -> `lambda_e2 verify-c2 [-d N]`
///   `lambda_e2 N`               -> `lambda_e2 generate -d N`
fn shim_legacy_args(mut argv: Vec<String>) -> Vec<String> {
    fn rewrite_flag(argv: &mut Vec<String>, sub: &str) {
        argv[1] = sub.to_string();
        if argv.len() > 2 && argv[2].parse::<i32>().is_ok() {
            argv.insert(2, "-d".to_string());
        }
    }
    let first = argv.get(1).cloned();
    match first.as_deref() {
        Some("--c2-only") => rewrite_flag(&mut argv, "c2"),
        Some("--verify-c2") => rewrite_flag(&mut argv, "verify-c2"),
        Some(s) if s.parse::<i32>().is_ok() => {
            let deg = s.to_string();
            argv[1] = "generate".to_string();
            argv.insert(2, "-d".to_string());
            argv.insert(3, deg);
        }
        _ => {}
    }
    argv
}

fn main() -> lambda_e2::Result<()> {
    let cli = Cli::parse_from(shim_legacy_args(std::env::args().collect()));

    // Progress goes to stderr via the `log` facade; RUST_LOG overrides -v/-q.
    let level = if cli.opts.quiet {
        "warn"
    } else if cli.opts.verbose > 0 {
        "debug"
    } else {
        "info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
        .format_timestamp(None)
        .format_target(false)
        .init();

    let out_dir = cli.opts.out_dir;
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;
    let cache_dir = cli.opts.cache_dir.unwrap_or_else(|| out_dir.clone());
    let debug_dir = cli.opts.debug_logs.then_some(out_dir.as_path());
    let max_filt = cli.opts.max_filt;
    if max_filt.is_some_and(|f| f < 1) {
        return Err("--max-filt must be >= 1".into());
    }
    let src_filter = SourceFilter {
        sphere: cli.opts.sphere,
        max_stem: cli.opts.max_stem,
    };
    if cli.opts.sphere.is_some_and(|n| n % 2 == 0 || n < 1) {
        return Err("--sphere must be a positive odd dimension (the map's sources \
             are odd-sphere classes)"
            .into());
    }
    // A filtration-capped table would silently truncate the sphere-side
    // H/P/relations outputs, so only the curtis and C2 subcommands accept it.
    if max_filt.is_some()
        && matches!(cli.cmd, Cmd::Generate | Cmd::Products | Cmd::H | Cmd::P)
    {
        return Err(
            "--max-filt is only supported by the curtis, c2, c2-all and c2-products \
             subcommands"
                .into(),
        );
    }

    match cli.cmd {
        Cmd::Curtis => {
            let deg = cli.opts.degree.unwrap_or(70);
            let t0 = std::time::Instant::now();
            let (tags, cocycles, pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            log::info!("[phase] curtis+db: {:.2}s", t0.elapsed().as_secs_f64());
            lambda_e2::census::run(&tags, &cocycles, &pages);
            log::info!(
                "Database ready in {}: {} tags, {} cocycles, {} E2 keys",
                cache_dir.display(),
                tags.len(),
                cocycles.index.len(),
                pages.0.len()
            );
        }

        Cmd::VerifyC2 => {
            let deg = cli.opts.degree.unwrap_or(40);
            verify::verify_c2(deg, max_filt)?;
        }

        Cmd::Products => {
            let deg = cli.opts.degree.unwrap_or(70);
            let (tags, cocycles, pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            let names = build_names(&pages);
            write_names_json(&pages, &names, &out_dir.join("E2_names.json"))?;
            log::info!("Computing E2_relations.csv...");
            mult_table(
                &tags,
                &cocycles,
                &pages,
                &names,
                &out_dir.join("E2_relations.csv"),
                cli.opts.mult_floor,
                cli.opts.mult_ceil,
            )?;
            write_rank_csv(&pages, &out_dir.join("E2_rank.csv"))?;
            log::info!("Done: E2_relations.csv, E2_names.json, E2_rank.csv");
        }

        Cmd::H => {
            let deg = cli.opts.degree.unwrap_or(70);
            let (tags, cocycles, pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            let names = build_names(&pages);
            write_names_json(&pages, &names, &out_dir.join("E2_names.json"))?;
            log::info!("Computing E2_H.csv...");
            let (hopf_results, _hopf_image_by_bidegree) =
                compute_hopf(&tags, &cocycles, &pages)?;
            write_h_csv(&hopf_results, &names, &out_dir.join("E2_H.csv"))?;
            write_rank_csv(&pages, &out_dir.join("E2_rank.csv"))?;
            log::info!("Done: E2_H.csv, E2_names.json, E2_rank.csv");
        }

        Cmd::P => {
            let deg = cli.opts.degree.unwrap_or(70);
            let (tags, cocycles, pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            let names = build_names(&pages);
            write_names_json(&pages, &names, &out_dir.join("E2_names.json"))?;
            // P is defined via H, so compute the Hopf image first (not written).
            log::info!("Computing the Hopf H map (needed for P)...");
            let (_hopf_results, hopf_image_by_bidegree) =
                compute_hopf(&tags, &cocycles, &pages)?;
            log::info!("Computing E2_P.csv...");
            let p_results = compute_p(&tags, &cocycles, &pages, &hopf_image_by_bidegree)?;
            write_p_csv(&p_results, &names, &out_dir.join("E2_P.csv"))?;
            write_rank_csv(&pages, &out_dir.join("E2_rank.csv"))?;
            log::info!("Done: E2_P.csv, E2_names.json, E2_rank.csv");
        }

        Cmd::C2 => {
            let deg = cli.opts.degree.unwrap_or(70);
            let t0 = std::time::Instant::now();
            let (tags, cocycles, mut pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            log::info!("[phase] curtis+load: {:.2}s", t0.elapsed().as_secs_f64());
            lambda_e2::census::run(&tags, &cocycles, &pages);

            log::info!("Computing C2 map → E2_C2.csv...");
            let t1 = std::time::Instant::now();
            let c2_results =
                compute_c2_images(&tags, &cocycles, &mut pages, max_filt, src_filter)?;
            log::info!("[phase] c2 map: {:.2}s", t1.elapsed().as_secs_f64());
            // (0, 0, 0, 0) marks images that are zero or could not be completed;
            // both are omitted so the downstream solver treats them as unknown
            // rather than as an explicit zero constraint.
            let c2_nonzero: std::collections::HashMap<_, _> = c2_results
                .into_iter()
                .filter(|(_, image)| image != &vec![ClassId::ZERO])
                .collect();
            write_operation_csv(&c2_nonzero, &pages, &out_dir.join("E2_C2.csv"))?;
            log::info!("Generating E2_rank.csv...");
            write_rank_csv(&pages, &out_dir.join("E2_rank.csv"))?;
            log::info!("Done: E2_C2.csv, E2_rank.csv");
        }

        Cmd::C2All => {
            let deg = cli.opts.degree.unwrap_or(70);
            let t0 = std::time::Instant::now();
            let (tags, cocycles, mut pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            log::info!("[phase] curtis+load: {:.2}s", t0.elapsed().as_secs_f64());
            lambda_e2::census::run(&tags, &cocycles, &pages);

            let t1 = std::time::Instant::now();
            compute_all(
                &tags,
                &cocycles,
                &mut pages,
                &out_dir,
                deg,
                true,
                max_filt,
                src_filter,
                cli.opts.resume,
            )?;
            log::info!("[phase] c2 all: {:.2}s", t1.elapsed().as_secs_f64());
        }

        Cmd::C2Products => {
            let deg = cli.opts.degree.unwrap_or(70);
            let t0 = std::time::Instant::now();
            let (tags, cocycles, mut pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            log::info!("[phase] curtis+load: {:.2}s", t0.elapsed().as_secs_f64());
            lambda_e2::census::run(&tags, &cocycles, &pages);

            let t1 = std::time::Instant::now();
            compute_all(
                &tags,
                &cocycles,
                &mut pages,
                &out_dir,
                deg,
                false,
                max_filt,
                src_filter,
                cli.opts.resume,
            )?;
            log::info!("[phase] c2 products: {:.2}s", t1.elapsed().as_secs_f64());
        }

        Cmd::C2Names => {
            let deg = cli.opts.degree.unwrap_or(70);
            let t0 = std::time::Instant::now();
            let (tags, cocycles, mut pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;
            log::info!("[phase] curtis+load: {:.2}s", t0.elapsed().as_secs_f64());
            // Sphere-side basis fingerprint (for cross-run identity checks).
            let names = build_names(&pages);
            write_names_json(&pages, &names, &out_dir.join("E2_names.json"))?;
            write_rank_csv(&pages, &out_dir.join("E2_rank.csv"))?;
            let t1 = std::time::Instant::now();
            log::info!("Building the Λ(C2) column...");
            let max_dim = pages.max_dimension();
            compute_c2_e2(max_dim, &tags, &cocycles, &mut pages, max_filt);
            write_c2_rank_csv(&pages, &out_dir.join("E2_C2_rank.csv"))?;
            write_c2_names_json(&pages, &out_dir.join("E2_C2_names.json"))?;
            log::info!("[phase] c2 column+names: {:.2}s", t1.elapsed().as_secs_f64());
            log::info!("Done: E2_C2_rank.csv, E2_C2_names.json, E2_names.json, E2_rank.csv");
        }

        Cmd::Generate => {
            let deg = cli.opts.degree.unwrap_or(70);
            // 1–2. Curtis + evens (cached per degree)
            let (tags, cocycles, mut pages) =
                load_or_compute_curtis(deg, &cache_dir, debug_dir, max_filt)?;

            // 3. Build vector name dictionary
            log::info!("Building vector names dictionary...");
            let names = build_names(&pages);
            write_names_json(&pages, &names, &out_dir.join("E2_names.json"))?;

            // 4. Compute multiplication table → E2_relations.csv
            log::info!("Computing E2_relations.csv...");
            mult_table(
                &tags,
                &cocycles,
                &pages,
                &names,
                &out_dir.join("E2_relations.csv"),
                cli.opts.mult_floor,
                cli.opts.mult_ceil,
            )?;

            // 5. Compute Hopf map → E2_H.csv
            log::info!("Computing E2_H.csv...");
            let (hopf_results, hopf_image_by_bidegree) = compute_hopf(&tags, &cocycles, &pages)?;
            write_h_csv(&hopf_results, &names, &out_dir.join("E2_H.csv"))?;

            // 6. Compute E operation → E2_E.csv
            log::info!("Computing E2_E.csv...");
            let e_results = compute_e(&pages)?;
            write_e_csv(&e_results, &names, &out_dir.join("E2_E.csv"))?;

            // 7. Compute P operation → E2_P.csv
            log::info!("Computing E2_P.csv...");
            let p_results = compute_p(&tags, &cocycles, &pages, &hopf_image_by_bidegree)?;
            write_p_csv(&p_results, &names, &out_dir.join("E2_P.csv"))?;

            // 8. Compute C2 map → E2_C2.csv
            // Builds the n=0 Lambda(C2) column into `pages`, then computes the image
            // of each odd-n element under Mahowald's map. Must run after the E/H/P
            // outputs (which expect no n=0 column) and before the rank output (which
            // must include it).
            log::info!("Computing E2_C2.csv...");
            let c2_results =
                compute_c2_images(&tags, &cocycles, &mut pages, max_filt, src_filter)?;
            // (0, 0, 0, 0) marks images that are zero or could not be completed;
            // both are omitted so the downstream solver treats them as unknown
            // rather than as an explicit zero constraint.
            let c2_nonzero: std::collections::HashMap<_, _> = c2_results
                .into_iter()
                .filter(|(_, image)| image != &vec![ClassId::ZERO])
                .collect();
            write_operation_csv(&c2_nonzero, &pages, &out_dir.join("E2_C2.csv"))?;

            // 9. Filtration-1 products on the Lambda(C2) column, read by the
            // propagator's h0-h3 product maps. The column was built in step 8.
            log::info!("Computing E2_C2_products.csv...");
            compute_c2_products(
                &tags,
                &cocycles,
                &pages,
                deg,
                &out_dir.join("E2_C2_products.csv"),
                max_filt,
                cli.opts.resume,
            )?;

            // 10. Generate E2_rank.csv (including the n=0 Lambda(C2) column)
            log::info!("Generating E2_rank.csv...");
            write_rank_csv(&pages, &out_dir.join("E2_rank.csv"))?;

            log::info!("All computations complete!");
        }
    }

    Ok(())
}
