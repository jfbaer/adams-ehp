#!/usr/bin/env python3
"""
Generate a complete interactive chart dataset from EHP spectral-sequence CSVs.

Reads one per-page chart CSV for each E-page and produces, for every sphere n,
a JSON chart plus a light- (and optionally dark-) mode interactive HTML page,
rendered through seqsee/jsonmaker.py and seqsee/main.py. An index.html
linking to all of them is written alongside.

Input CSVs are the per-page files E{r}_{N}.csv written by python/run.py's
write_spheres() (e.g. python/charts/E2_76.csv).

Run this script from the `charts/` directory so it can find its bundled
`seqsee/` package:

    poetry install                    # once, to create the environment
    poetry run python generate_charts.py

By default it reads the shipped chart CSVs in ./data and writes
./interactive_charts. Override with --charts-dir (e.g. ../python/charts for a
fresh propagator run) / --output-dir.
"""

import csv
import glob
import os
import re
import subprocess
import argparse
from pathlib import Path


def find_page_csvs(charts_dir="data"):
    """
    Scan charts_dir for the per-page chart CSVs E{r}_{N}.csv (write_spheres()
    output). Suffixed variants like E{r}_{N}_unknown.csv carry an extra
    underscore-word before ".csv" and are ignored.

    Returns a dict mapping r (int) -> csv path.
    """
    pages = {}
    pattern = os.path.join(charts_dir, "E*_*.csv")
    for csv_path in sorted(glob.glob(pattern)):
        basename = os.path.basename(csv_path)
        m = re.match(r"E(\d+)_(\d+)\.csv$", basename)
        if not m:
            continue
        pages[int(m.group(1))] = csv_path
    return pages


def get_max_stem(csv_path):
    """Read a CSV and return the maximum value in the 'stem' column."""
    max_stem = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_stem = max(max_stem, int(row["stem"]))
            except (ValueError, KeyError):
                continue
    return max_stem


def get_max_filt(csv_path):
    """Read a CSV and return the maximum value in the 'Adams filtration'
    column (0 if none). Used to set a uniform y-axis height across spheres
    without dropping any computed class."""
    max_filt = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_filt = max(max_filt, int(row["Adams filtration"]))
            except (ValueError, KeyError):
                continue
    return max_filt


def get_n_values_from_csv(csv_path):
    """Read a CSV and return the sorted set of distinct int values in the 'n' column,
    excluding any n greater than the maximum stem value."""
    max_stem = get_max_stem(csv_path)
    n_values = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row["n"])
                if n <= max_stem:
                    n_values.add(n)
            except (ValueError, KeyError):
                continue
    return sorted(n_values)


# Map registry for the side-by-side "map image" views. `col` is the CSV column
# holding each source class's image; `target_n` maps a source sphere n to its
# target chart's n (C2 targets n=0 = S/2). `domain` further restricts source n.
MAPS = {
    "E":  {"col": "E",  "target_n": lambda n: n + 1,        "domain": lambda n: n >= 2},
    "H":  {"col": "H",  "target_n": lambda n: 2 * n - 1,    "domain": lambda n: n >= 2},
    "P":  {"col": "P",  "target_n": lambda n: (n - 1) // 2, "domain": lambda n: n % 2 == 1 and n >= 5},
    "C2": {"col": "C2", "target_n": lambda n: 0,            "domain": lambda n: n % 2 == 1 and n >= 3},
}


def sphere_title_latex(n, r):
    """LaTeX panel title for the E_r page of S^n (n=0 is the S/2 Moore spectrum)."""
    if n == 0:
        return f"$\\mathrm{{E}}_{{{r}}}(\\mathbb{{S}}/2)$"
    return f"$\\mathrm{{E}}_{{{r}}}(S^{{{n}}})$"


def get_map_source_ns(csv_path, columns):
    """For each column name, the set of source n (int) with a nonempty value.

    Single pass over the CSV so we only generate a map side-by-side where the
    map actually has image data. Returns {column: set(n)}.
    """
    result = {c: set() for c in columns}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row["n"])
            except (ValueError, KeyError):
                continue
            for c in columns:
                v = (row.get(c) or "").strip()
                if v:
                    result[c].add(n)
    return result


def run_poetry_command(cmd, description):
    """Run a command via poetry in the seqsee directory."""
    print(f"  {description}")

    full_cmd = ["poetry", "run"] + cmd
    result = subprocess.run(full_cmd, cwd="seqsee", capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        return False

    print("    OK")
    return True


def seqsee_path(path):
    """Rebase a path for a command running inside seqsee/ (absolute paths
    pass through; relative ones gain the ../ hop back to this directory)."""
    return path if os.path.isabs(path) else f"../{path}"


def expected_chart_files(output_dir, n, r, generate_light, generate_dark):
    """Paths of the JSON plus each requested HTML output for one (n, r) chart."""
    paths = [f"{output_dir}/S{n}_E{r}.json"]
    if generate_light:
        paths.append(f"{output_dir}/S{n}_E{r}.html")
    if generate_dark:
        paths.append(f"{output_dir}/S{n}_E{r}_dark.html")
    return paths


def generate_chart(csv_file, n, r, output_dir, generate_light, generate_dark, max_stem=None, max_filt=None):
    """
    Generate JSON and HTML files for the sphere chart S^n on page E_r.

    max_stem, when set, clamps the chart's x-axis (stem) at that value and draws
    differentials from the off-chart stem max_stem+1 as incoming half-lines.
    Returns the number of files successfully produced (0..3).
    """
    json_file = f"{output_dir}/S{n}_E{r}.json"
    html_light_file = f"{output_dir}/S{n}_E{r}.html"
    html_dark_file = f"{output_dir}/S{n}_E{r}_dark.html"

    count = 0
    label = f"S{n} E{r}"
    stem_args = ["--max-stem", str(max_stem)] if max_stem else []
    if max_filt:
        stem_args = stem_args + ["--max-filt", str(max_filt)]

    # Generate JSON
    success = run_poetry_command(
        ["python", "jsonmaker.py", seqsee_path(csv_file), seqsee_path(json_file), "sphere", str(n)] + stem_args,
        f"{label} JSON",
    )
    if not success:
        return 0
    count += 1

    # Generate Light Mode HTML
    if generate_light:
        success = run_poetry_command(
            ["python", "main.py", seqsee_path(json_file), seqsee_path(html_light_file), "light"] + stem_args,
            f"{label} Light HTML",
        )
        if success:
            count += 1

    # Generate Dark Mode HTML
    if generate_dark:
        success = run_poetry_command(
            ["python", "main.py", seqsee_path(json_file), seqsee_path(html_dark_file), "dark"] + stem_args,
            f"{label} Dark HTML",
        )
        if success:
            count += 1

    return count


def generate_map_sidebyside(map_key, n, r, output_dir, generate_light, generate_dark, csv_file, max_filt=None):
    """
    Generate the side-by-side "map image" HTML for one map at source sphere n and
    page r, reusing the already-generated per-sphere JSONs. The left panel is the
    source chart S^n (carrying data-map links), the right panel is the target
    chart. Silently skips when out of domain or the target chart wasn't generated.

    Returns the number of HTML files produced (0..2).
    """
    spec = MAPS[map_key]
    target_n = spec["target_n"](n)

    if not spec["domain"](n) or target_n == n or target_n < 0:
        return 0

    source_json = f"{output_dir}/S{n}_E{r}.json"
    target_json = f"{output_dir}/S{target_n}_E{r}.json"
    # "Only where target exists": skip boundary cases whose target chart is
    # outside the generated range rather than emit a broken/empty panel.
    if not (os.path.exists(source_json) and os.path.exists(target_json)):
        return 0

    # Source JSON variant carrying data-map (the map's image links).
    map_json = f"{output_dir}/S{n}_E{r}_map{map_key}.json"
    if not os.path.exists(map_json):
        ok = run_poetry_command(
            ["python", "jsonmaker.py", seqsee_path(csv_file), seqsee_path(map_json),
             "sphere", str(n), "--map", spec["col"]]
            + (["--max-filt", str(max_filt)] if max_filt else []),
            f"{map_key}-map S{n} E{r} source JSON",
        )
        if not ok:
            return 0

    source_title = sphere_title_latex(n, r)
    target_title = sphere_title_latex(target_n, r)

    variants = []
    if generate_light:
        variants.append(("light", f"{output_dir}/S{n}_E{r}_{map_key}sbs.html", f"S{n}_E{r}.html"))
    if generate_dark:
        variants.append(("dark", f"{output_dir}/S{n}_E{r}_{map_key}sbs_dark.html", f"S{n}_E{r}_dark.html"))

    count = 0
    for theme, out_html, back_url in variants:
        if os.path.exists(out_html):
            count += 1
            continue
        ok = run_poetry_command(
            ["python", "main.py", "--sidebyside", seqsee_path(map_json), seqsee_path(target_json),
             seqsee_path(out_html), theme, back_url, map_key, source_title, target_title],
            f"{map_key}-map S{n}->S{target_n} E{r} {theme} sidebyside",
        )
        if ok:
            count += 1
    return count


def create_index_html(
    charts_dir, r_values, sphere_n_values, generate_light, generate_dark
):
    """
    Create index.html with one grid of sphere charts per E-page.

    sphere_n_values: dict  r -> sorted list of n values
    """
    index_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Spectral Sequence Navigator</title>
    <style>
        body {
            font-family: 'Computer Modern Serif', serif;
            margin: 0;
            padding: 20px;
            background: #eff1f5;
            color: #4c4f69;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            font-size: 2.2em;
            margin-bottom: 30px;
            color: #179299;
        }

        .csv-section {
            margin: 30px 0;
            background: #e6e9ef;
            border-radius: 12px;
            padding: 20px;
        }

        .csv-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            color: #179299;
        }

        .sphere-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(85px, 1fr));
            gap: 8px;
            max-height: 500px;
            overflow-y: auto;
            padding: 15px;
            background: #dce0e8;
            border-radius: 8px;
        }

        .sphere-card {
            background: #ccd0da;
            border: 2px solid #bcc0cc;
            border-radius: 6px;
            padding: 12px 8px;
            text-decoration: none;
            color: #4c4f69;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .sphere-card:hover {
            background: #179299;
            color: #eff1f5;
            border-color: #179299;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .sphere-number {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 3px;
        }

        .sphere-label {
            font-size: 0.75em;
            opacity: 0.8;
        }

        .instructions {
            background: #dce0e8;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
        }

        .instructions h3 {
            margin-top: 0;
            color: #179299;
            font-size: 1.3em;
        }

        .key-bindings {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 10px 20px;
            margin: 15px 0;
        }

        .key {
            font-family: monospace;
            background: #179299;
            color: #eff1f5;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
        }

        .stats {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: #dce0e8;
            border-radius: 12px;
            color: #5c5f77;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Spectral Sequence Navigator</h1>

        <div class="instructions">
            <h3>Navigation Controls</h3>
            <div class="key-bindings">
                <span class="key">s</span> <span>Move to n+1</span>
                <span class="key">w</span> <span>Move to n-1</span>
                <span class="key">d</span> <span>Move to next page E{r+1} (same space)</span>
                <span class="key">a</span> <span>Move to previous page E{r-1} (same space)</span>
                <span class="key">e</span> <span>E-map split-screen: S<sup>n</sup> beside S<sup>n+1</sup></span>
                <span class="key">h</span> <span>H-map split-screen: S<sup>n</sup> beside S<sup>2n-1</sup></span>
                <span class="key">p</span> <span>P-map split-screen (odd n&ge;5): S<sup>n</sup> beside S<sup>(n-1)/2</sup></span>
                <span class="key">c</span> <span>C2-map split-screen: S<sup>n</sup> beside &#120138;/2</span>
                <span class="key">E/H/P/C</span> <span>Same split-screen, with the map image highlighted</span>
            </div>
        </div>
'''

    total_charts = 0

    for r in sorted(r_values):
        s_nvals = sphere_n_values.get(r, [])

        index_content += f'''
        <div class="csv-section">
            <div class="csv-title">E{r} Page (r = {r})</div>
'''

        if s_nvals:
            index_content += '            <div class="sphere-grid">\n'
            for n in s_nvals:
                html_light = f"S{n}_E{r}.html"
                html_dark = f"S{n}_E{r}_dark.html"
                light_exists = os.path.exists(f"{charts_dir}/{html_light}")
                dark_exists = os.path.exists(f"{charts_dir}/{html_dark}")
                if not (light_exists or dark_exists):
                    continue
                total_charts += 1
                default_file = html_light if light_exists else html_dark
                if generate_light and generate_dark and light_exists and dark_exists:
                    index_content += (
                        f'                <a href="{default_file}" class="sphere-card"'
                        f' data-light="{html_light}" data-dark="{html_dark}">\n'
                    )
                else:
                    index_content += f'                <a href="{default_file}" class="sphere-card">\n'
                # n=0 is the mod-2 Moore spectrum S/2, not the 0-sphere.
                sphere_label = "&#120138;/2" if n == 0 else f"S<sup>{n}</sup>"
                index_content += f'                    <div class="sphere-number">n = {n}</div>\n'
                index_content += f'                    <div class="sphere-label">{sphere_label}</div>\n'
                index_content += '                </a>\n'
            index_content += '            </div>\n'

        index_content += '        </div>\n'

    # Theme info
    if generate_light and generate_dark:
        theme_info = "Catppuccin theming (Latte &amp; Mocha)"
    elif generate_light:
        theme_info = "Light mode only"
    else:
        theme_info = "Dark mode only"

    index_content += f'''
        <div class="stats">
            Generated {total_charts} interactive charts &bull; {theme_info}
        </div>
    </div>

    <script>
        // Theme-aware navigation for index page (only if both themes available)
        document.addEventListener('DOMContentLoaded', function() {{
            const cardsWithThemes = document.querySelectorAll('.sphere-card[data-light][data-dark]');
            if (cardsWithThemes.length > 0) {{
                const currentTheme = sessionStorage.getItem('seqsee-theme') || 'light';

                cardsWithThemes.forEach(card => {{
                    const lightFile = card.getAttribute('data-light');
                    const darkFile = card.getAttribute('data-dark');

                    if (currentTheme === 'dark' && darkFile) {{
                        card.href = darkFile;
                    }} else {{
                        card.href = lightFile;
                    }}
                }});
            }}
        }});
    </script>
</body>
</html>
'''

    index_path = f"{charts_dir}/index.html"
    with open(index_path, "w") as f:
        f.write(index_content)

    print(f"Created index.html at {index_path}")
    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive charts from per-page EHP spectral-sequence CSVs"
    )
    parser.add_argument(
        "--mode",
        choices=["light", "dark", "both"],
        default="light",
        help="Which theme modes to generate (default: light)",
    )
    parser.add_argument(
        "--charts-dir",
        default="data",
        help="Directory containing the CSV files (default: data — the shipped "
        "chart CSVs; use ../python/charts for a fresh propagator run)",
    )
    parser.add_argument(
        "--output-dir",
        default="interactive_charts",
        help="Output directory for generated files (default: interactive_charts)",
    )
    parser.add_argument(
        "--max-stem",
        type=int,
        default=50,
        help="Cut the per-sphere charts off at this stem (default: 50); "
             "differentials from the off-chart stem max-stem+1 are drawn as "
             "incoming half-lines. Pass 0 to disable (full-width charts).",
    )
    parser.add_argument(
        "--sidebyside", action="store_true",
        help="also generate the map side-by-side comparison views (E/H/P/C2) "
             "(off by default; these are the slow, bulky part of a run)",
    )
    args = parser.parse_args()

    generate_light = args.mode in ("light", "both")
    generate_dark = args.mode in ("dark", "both")

    # Auto-detect the per-page CSVs
    page_csvs = find_page_csvs(args.charts_dir)
    if not page_csvs:
        print(f"No chart CSVs found in {args.charts_dir}/.")
        print("Expected files matching E{r}_{N}.csv.")
        return

    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)

    # Uniform y-axis height = the largest Adams filtration actually present in
    # the data. Every column ends at its own s+f <= tot boundary naturally (the
    # CSV simply has no rows beyond it), so this sets a shared axis height
    # WITHOUT dropping any computed class.
    filt_cap = max((get_max_filt(p) for p in page_csvs.values()), default=0) or None

    # Determine n-values per r-value from CSV data
    sphere_n_values = {r: get_n_values_from_csv(path)
                       for r, path in sorted(page_csvs.items())}

    r_values = sorted(page_csvs.keys())

    # Print summary
    mode_desc = {"light": "light mode only", "dark": "dark mode only", "both": "both light and dark modes"}[args.mode]
    print("=== Generating Complete Interactive Charts Dataset ===")
    print(f"Source: {args.charts_dir}/")
    print(f"Mode: {mode_desc}")
    print(f"Detected E-pages: {', '.join(f'E{r}' for r in r_values)}")
    for r in r_values:
        s_count = len(sphere_n_values.get(r, []))
        print(f"  E{r}: {s_count} sphere n-values")
    print(f"Output: {output_dir}/")
    print("Note: Files that already exist will be skipped automatically")

    successful = 0
    total_attempts = 0
    skipped = 0
    files_per = 1 + int(generate_light) + int(generate_dark)

    # --- Generate sphere charts ---
    for r in r_values:
        csv_file = page_csvs[r]
        for n in sphere_n_values.get(r, []):
            total_attempts += files_per
            expected = expected_chart_files(output_dir, n, r, generate_light, generate_dark)
            if all(os.path.exists(p) for p in expected):
                successful += files_per
                skipped += files_per
                continue

            count = generate_chart(csv_file, n, r, output_dir, generate_light, generate_dark, args.max_stem, filt_cap)
            successful += count

    # --- Generate map-image side-by-side views (E/H/P/C2) ---
    # For each page, one pass over the CSV finds which source n carry each map's
    # data; we only emit a view where the map has image data AND the target
    # chart was generated (see generate_map_sidebyside).
    sbs_count = 0
    if args.sidebyside:
        map_cols = [spec["col"] for spec in MAPS.values()]
        for r in r_values:
            csv_file = page_csvs[r]
            src_ns_by_col = get_map_source_ns(csv_file, map_cols)
            for map_key, spec in MAPS.items():
                src_ns = src_ns_by_col[spec["col"]]
                for n in sphere_n_values.get(r, []):
                    if n not in src_ns:
                        continue
                    sbs_count += generate_map_sidebyside(
                        map_key, n, r, output_dir, generate_light, generate_dark, csv_file, filt_cap
                    )

    print(f"\n=== Generation Complete ===")
    print(f"Successful operations: {successful}/{total_attempts}")
    if skipped > 0:
        print(f"Skipped {skipped} files (already exist)")
        print(f"Newly generated: {successful - skipped} files")
    print(f"Map side-by-side views (E/H/P/C2): {sbs_count} HTML files")

    # Create index
    if successful > 0:
        create_index_html(
            output_dir, r_values, sphere_n_values,
            generate_light, generate_dark,
        )
        print(f"\nStart exploring:")
        print(f"open {output_dir}/index.html")
        print(f"\nOr jump directly to any chart:")
        print(f"open {output_dir}/S3_E2.html")
        print(f"\nNavigation controls:")
        print(f"  s/w: move between spheres (n+/-1)")
        print(f"  a/d: move between pages (r+/-1)")
        print(f"  e/h/p/c: open the E/H/P/C2-map split-screen view")
        print(f"  E/H/P/C: same, with the map image highlighted")


if __name__ == "__main__":
    main()
