#!/usr/bin/env sage-python
"""Batch-generate why graphs for every nonzero differential on a page.

Three modes, run in sequence:

1. Generate DOT files + per-page manifest (needs sage, one page per run):
       sage -python make_why_graphs.py --page 2
   Writes why_graphs/d2/dot/why_{n}_{s}_{f}.dot and why_graphs/manifest_d2.json
   for every tridegree of data/E2/d2 whose differential value is nonzero.

2. Render DOT -> SVG in parallel (plain python, no sage):
       python3 make_why_graphs.py --render-svg
   Renders why_graphs/d{r}/why_{n}_{s}_{f}.svg for every dot file that has no
   up-to-date SVG yet.

3. Merge the per-page manifests into the viewer's manifest:
       python3 make_why_graphs.py --merge-manifest
   Writes why_graphs/manifest.js (a `const MANIFEST = [...]` script, so the
   viewer works from file:// where fetch() of JSON is blocked).

The viewer itself is the hand-written why_graphs/index.html.
"""

import argparse
import ast
import json
import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path

OUT_DEFAULT = "why_graphs"


def generate_page(r, out_dir, limit=None):
    """Build proof trees for every nonzero differential of d_r and write the
    DOT files and the page manifest. Requires sage (imports why -> d -> lib)."""
    import why  # sage import; deferred so --render-svg/--merge-manifest stay sage-free
    from differentials import DifferentialsPage

    d_file = Path(f"data/E{r}/d{r}")
    print(f"Loading {d_file} ...", flush=True)
    with open(d_file) as fp:
        obj = json.load(fp)

    nonzero = sorted(
        ast.literal_eval(key)
        for key, entry in obj.items()
        if key != "r" and any(entry["v"])
    )
    if limit is not None:
        nonzero = nonzero[:limit]
    print(f"d{r}: {len(nonzero)} nonzero differentials", flush=True)

    dims = why.dimensions_from_json(obj, int(obj["r"]))
    print("Reconstructing DifferentialsPage (this is the slow part) ...", flush=True)
    d_page = DifferentialsPage.from_json(obj, dimension_dict=dims)
    del obj

    dot_dir = Path(out_dir) / f"d{r}" / "dot"
    dot_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, td in enumerate(nonzero):
        n, s, f = td
        tree = why.build_proof_tree(d_page, td)
        node_count = len({id(node) for _, _, node in why.walk(tree)})
        (dot_dir / f"why_{n}_{s}_{f}.dot").write_text(why.render_dot(tree))
        # "constraint with no recorded source" is why.py's known act-of-God
        # category (drawn as a childless node); it appears on ~74% of graphs,
        # so it gets its own count instead of drowning out real anomalies.
        real = [w for w in tree.warnings if "no recorded source" not in w]
        manifest.append({
            "r": r, "n": n, "s": s, "f": f,
            "file": f"d{r}/why_{n}_{s}_{f}.svg",
            "len": tree.proof_length,
            "nodes": node_count,
            "warn": real,
            "nosrc": len(tree.warnings) - len(real),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(nonzero)}", flush=True)

    manifest_file = Path(out_dir) / f"manifest_d{r}.json"
    manifest_file.write_text(json.dumps(manifest))
    warned = sum(1 for m in manifest if m["warn"])
    print(f"Wrote {len(manifest)} dot files and {manifest_file} "
          f"({warned} graphs with warnings)", flush=True)


def _render_one(pair):
    dot_file, svg_file = pair
    try:
        subprocess.run(["dot", "-Tsvg", str(dot_file), "-o", str(svg_file)],
                       check=True, capture_output=True, text=True, timeout=120)
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        return f"{dot_file}: {stderr.strip() or exc}"


def render_svgs(out_dir, jobs):
    """Render every dot file lacking an up-to-date SVG sibling."""
    if not shutil.which("dot"):
        sys.exit("graphviz `dot` not found on PATH")
    todo = []
    for dot_file in sorted(Path(out_dir).glob("d*/dot/why_*.dot")):
        svg_file = dot_file.parent.parent / (dot_file.stem + ".svg")
        if not svg_file.exists() or svg_file.stat().st_mtime < dot_file.stat().st_mtime:
            todo.append((dot_file, svg_file))
    print(f"Rendering {len(todo)} SVGs with {jobs} workers ...", flush=True)
    failures = []
    with multiprocessing.Pool(jobs) as pool:
        for i, err in enumerate(pool.imap_unordered(_render_one, todo, chunksize=16)):
            if err:
                failures.append(err)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    print(f"Rendered {len(todo) - len(failures)}/{len(todo)}")
    for err in failures:
        print(f"FAILED {err}")
    if failures:
        sys.exit(1)


def merge_manifest(out_dir):
    """Concatenate the per-page manifests into manifest.js for the viewer."""
    records = []
    for mf in sorted(Path(out_dir).glob("manifest_d*.json")):
        records.extend(json.loads(mf.read_text()))
    records.sort(key=lambda m: (m["r"], m["s"], m["n"], m["f"]))
    out = Path(out_dir) / "manifest.js"
    out.write_text("const MANIFEST = " + json.dumps(records) + ";\n")
    warned = sum(1 for m in records if m["warn"])
    print(f"Wrote {out}: {len(records)} graphs, {warned} with warnings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Pages 6-8 carry no nonzero differentials in the shipped range, so
    # there are no proof graphs to draw for them.
    parser.add_argument("--page", type=int, choices=(2, 3, 4, 5),
                        help="generate dot files for this page (needs sage)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N tridegrees (smoke tests)")
    parser.add_argument("--render-svg", action="store_true",
                        help="render missing/stale SVGs from the dot files")
    parser.add_argument("--merge-manifest", action="store_true",
                        help="merge per-page manifests into manifest.js")
    parser.add_argument("--out", default=OUT_DEFAULT, help="output directory")
    parser.add_argument("--jobs", type=int, default=8, help="parallel dot renders")
    args = parser.parse_args()

    if not (args.page or args.render_svg or args.merge_manifest):
        parser.error("nothing to do: pass --page R, --render-svg, or --merge-manifest")
    if args.page:
        generate_page(args.page, args.out, limit=args.limit)
    if args.render_svg:
        render_svgs(args.out, args.jobs)
    if args.merge_manifest:
        merge_manifest(args.out)


if __name__ == "__main__":
    main()
