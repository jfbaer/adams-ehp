"""CSV -> chart-JSON converter for the EHP chart pipeline.

Reads one of the per-page E_r CSVs (one row per basis class, with edge/target
columns such as h0target, drtarget, nulldif, and the map columns E/H/P/C2/J)
and emits a SeqSee chart JSON document that main.py renders to HTML.

CLI:
    jsonmaker.py <input.csv> <output.json> [view_mode] [filter_value]
                 [highlight_mode] [source_csv] [--map COL] [--max-stem N]
"""

import re
import sys
from pathlib import Path

import pandas as pd
from compact_json import Formatter
from jsonschema import validate

# The renderer module that lives next to this script; we reuse its schema loader.
from main import load_schema

# Regular expressions for substitutions
substitutions = [
    # Matches any string starting with an underscore, and surrounds it in \overline{}
    (re.compile(r"^_(.*)$"), r"\\overline{\1}"),
    # Matches the dot operator, which is rendered as a centered dot in LaTeX
    (re.compile(r"\."), r"\\cdot"),
    # Matches the string "DD" and replaces it with "{D}". The braces are necessary if we want to
    # handle both "DD" -> "D" and "D" -> "\Delta".
    (re.compile(r"DD"), r"{D}"),
    # Matches the string "D" and replaces it with "\Delta", but only if it is not surrounded by
    # curly braces. The `(?<!...)` and `(?!...)` expressions are negative lookbehind and negative
    # lookahead, respectively. They ensure that whatever we are matching is in the right "context",
    # i.e. in this case not being in braces, but without including that context in the match. This
    # is necessary to avoid replacing the "D" in "{D}".
    (re.compile(r"(?<!{)D(?!})"), r"\\Delta"),
    # Matches the word "t" and replaces it with "\tau". This 't' character needs to be surrounded by
    # either the beginning or end of the line, or a non-word character (something that matches
    # `\W`).
    (re.compile(r"\bt\b"), r"\\tau"),
    # Matches word expressions starting with a non-empty string of latin letters and curly braces
    # (group 1) and ending with a non-empty string of numbers or commas (,) (group 2). We insert an
    # underscore between the two groups and wrap the second one in curly braces. This ensures that
    # subscripts are correctly rendered in LaTeX. NB: Caret (^) is a word separator.
    (re.compile(r"\b([a-zA-Z{}]+)([\d,]+)\b"), r"\1_{\2}"),
    # Matches terms of the form '(letters or curly braces)^(numbers)', and wraps the second group in
    # curly braces. This ensures that superscripts are correctly rendered in LaTeX.
    (re.compile(r"\b([a-zA-Z{}]+)\^(\d+)\b"), r"\1^{\2}"),
]

# Regular expression for detecting "again" suffixes. We strip anything that is whitespace followed
# by any number of non-word characters, then the word "again", and then anything else until the end
# of the line.
detect_again = re.compile(r"\s\W*again.*")

arrow_length = 0.7


def try_get_key(row, key, default=None):
    """
    Get a key from a row.

    Return a default value if the key is not present or the value is `nan`.
    """

    try:
        if pd.isna(row[key]):
            return default
        return row[key]
    except KeyError:
        return default


def edge_offset(edge_type, arrow_length=1):
    """Chart-coordinate offset for an arrow (edge with no on-chart target)."""
    if edge_type == "h0":
        offset = {"x": 0, "y": 1}
    elif edge_type == "h1":
        offset = {"x": 1, "y": 1}
    elif edge_type == "h2":
        offset = {"x": 3, "y": 1}
    elif edge_type == "dr":
        offset = {"x": -1, "y": 2}
    elif edge_type == "nulldif":
        offset = {"x": -1, "y": 2}
    elif edge_type == "E":
        offset = {"x": 1, "y": 0}  # E-type edges (stems mode only)
    else:
        raise ValueError
    for key in offset:
        offset[key] *= arrow_length
    return offset


def parse_node_coordinates(node_name):
    """Parse node name to extract s, stem, f, index values.

    Handles both bare names like '4_28_5_1' and prefixed names like
    'S4_28_5_1'.  The prefix (everything before the first digit in the
    first segment) is stripped before parsing.

    Names follow the n_s_f[_i] convention (sphere n, stem s, filtration f,
    basis index i). The returned keys are historical: "s" holds the sphere
    number n and "stem" holds the stem s.
    """
    if not node_name or node_name == "loc":
        return None
    parts = node_name.split("_")
    if len(parts) >= 3:
        try:
            # Strip any non-digit prefix from the first part (e.g. "S2" -> "2",
            # "SO10" -> "10").  If the first part is already numeric this is a
            # no-op.
            first = re.sub(r"^[A-Za-z]+", "", parts[0])
            s = int(first)
            stem = int(parts[1])
            f = int(parts[2])
            index = int(parts[3]) if len(parts) > 3 else 0
            return {"s": s, "stem": stem, "f": f, "index": index}
        except ValueError:
            return None
    return None


def build_hit_map(df):
    """Map node name -> d_r height, for every node listed in some drtarget.

    Precomputed once so stem-mode node coloring is O(N) instead of scanning
    the whole frame per node.
    """
    hit = {}
    for _, row in df.iterrows():
        drt = row.get("drtarget")
        if not drt or str(drt) == "nan" or drt == "":
            continue
        src_name, _ = deduplicate_name(row["name"])
        sc = parse_node_coordinates(src_name)
        if not sc:
            continue
        for t in str(drt).split(";"):
            tc = parse_node_coordinates(t)
            if tc:
                hit[t] = tc["f"] - sc["f"]
    return hit


def extract_node_attributes(row, view_mode="sphere", hit_map=None):
    ret = []
    if row.get("tautorsion", []):
        torsion = int(row["tautorsion"])
        if torsion >= 4:
            ret.append("tau4plus")
        elif torsion > 0:
            ret.append(f"tau{torsion}")

    # Add differential coloring for stem mode. Supporting a differential wins
    # over being hit by one (STEM_VIEW_SPEC.md).
    if view_mode == "stem":
        node_name, _ = deduplicate_name(row["name"])
        drt = row.get("drtarget")
        if drt and str(drt) != "nan" and drt != "":
            # Node supports a differential — open circle in the d_r color
            sc = parse_node_coordinates(node_name)
            tc = parse_node_coordinates(str(drt).split(";")[0])
            if sc and tc:
                ret.append(f"diff_d{tc['f'] - sc['f']}_open")
        elif hit_map and node_name in hit_map:
            # Node is hit by a differential — filled circle in the d_r color
            ret.append(f"diff_d{hit_map[node_name]}_filled")

    return ret


def extract_edge_attributes(row, edge_type, nodes):
    """Compute the attribute/alias list for one edge of type `edge_type` from `row`."""
    ret = []
    # Handle nulldif special case - column is named "nulldif" not "nulldiftarget"
    if edge_type == "nulldif":
        target_node = row["nulldif"]
        target_info = try_get_key(row, f"{edge_type}info")
    elif edge_type == "E":
        # EHP CSVs store suspension targets in the "E" column
        target_node = row["E"] if "E" in row else try_get_key(row, "Etarget", "")
        target_info = try_get_key(row, "Einfo")
    else:
        target_node = row[f"{edge_type}target"]
        # Info fields are sometimes missing, so we default to an empty string
        target_info = try_get_key(row, f"{edge_type}info")
    if edge_type == "dr":
        # Compute height for dr edges and assign specific differential type (d2, d3, d4, etc.)
        source_name, _ = deduplicate_name(row["name"])
        source_coords = parse_node_coordinates(source_name)
        # Handle semicolon-separated targets by using the first target for height calculation
        first_target = target_node.split(";")[0] if ";" in target_node else target_node
        target_coords = parse_node_coordinates(first_target)
        if source_coords and target_coords:
            height = target_coords["f"] - source_coords["f"]
            # Add specific differential type based on height
            if height > 0:
                ret.append(f"d{height}")  # e.g., "d2", "d3", "d4"
    elif edge_type == "E":
        ret.append("E")  # E-type edges for stem mode
        # An E edge is colored by the class it hits: inherit the d_r color of
        # the target node's differential state (STEM_VIEW_SPEC.md).
        first_target = str(target_node).split(";")[0]
        for attr in try_get_key(nodes.get(first_target, {}), "attributes", default=[]):
            if isinstance(attr, str) and attr.startswith("diff_d"):
                ret.append(attr.split("_")[1])  # "diff_d3_open" -> "d3"
    elif edge_type == "nulldif":
        # Compute length for nulldif based on Adams filtration difference
        source_name, _ = deduplicate_name(row["name"])
        source_coords = parse_node_coordinates(source_name)
        # Handle semicolon-separated targets by using the first target for length calculation
        first_target = target_node.split(";")[0] if ";" in target_node else target_node
        target_coords = parse_node_coordinates(first_target)
        if source_coords and target_coords:
            length = abs(target_coords["f"] - source_coords["f"])
            # Add length-based type (n2, n3, n4, etc.)
            if length > 0:
                ret.append(f"n{length}")  # e.g., "n2", "n3", "n4"
    elif target_node in nodes and edge_type not in ["nulldif"]:
        # Edges into a node inherit the attributes of their target, as long as they aren't
        # nulldiff edges (which get their own coloring logic)
        target_node_attributes = try_get_key(
            nodes.get(target_node, {}), "attributes", default=[]
        )
        ret.extend(target_node_attributes)
    if target_node == "loc" or target_info == "loc":
        # This edge is an arrow
        if edge_type == "h1":
            # We treat h1 towers differently because they are also always red
            ret.append("h1tower")
        else:
            ret.append({"arrowTip": "simple"})
    # Only process info fields for non-differential edges (h0, h1, h2, E)
    # For dr and nulldif edges, we compute their types directly above
    if target_info and edge_type not in ["dr", "nulldif"]:
        # The info field has other instructions for the edge. We treat them as aliases and let
        # SeqSee handle them. The only exception is "h", which we need to tag with "edge_type" so
        # the correct alias is applied.
        if isinstance(target_info, float):
            target_info = str(int(target_info))
        extra_attributes = target_info.split(" ")
        if "h" in extra_attributes:
            extra_attributes.remove("h")
            extra_attributes.append(f"h{edge_type}")
        ret.extend(extra_attributes)

    return ret



def label_from_node_name(node_name):
    """Apply substitutions to a node name to generate a label, wrapped in dollar signs for Latex."""
    label = node_name
    for pattern, replacement in substitutions:
        label = pattern.sub(replacement, label)
    if label:
        label = f"${label}$"
    return label


def deduplicate_name(name):
    """Deduplicate names by removing all variations of 'again'. We also return whether the name was
    a duplicate."""
    # The legacy CSV format marks repeated rows for the same class with an "again"
    # suffix rather than listing multiple edge targets on one row; the input format
    # is fixed, so we normalize the suffix away here.

    # First strip outer opening and closing parentheses if both are present. This makes it possible
    # for the regex to only match suffixes, while not affecting the rest of the name.
    if name.startswith("(") and name.endswith(")"):
        name = name[1:-1]
    if detect_again.search(name):
        return (detect_again.sub("", name), True)
    return (name, False)


def nodes_to_json(df, view_mode="sphere", filter_value=None, highlight_mode=None, highlight_targets=None, map_column=None, max_stem=None, max_filt=None):
    """Build the "nodes" JSON object (node name -> chart node) from the CSV frame.

    Sphere mode plots (stem, Adams filtration); stem mode plots (n, Adams
    filtration). `filter_value` selects the single n (sphere) or stem (stem)
    slice being drawn.
    """
    nodes = {}
    hit_map = build_hit_map(df) if view_mode == "stem" else None
    for _, row in df.iterrows():
        # Process node information
        node_name, is_duplicate = deduplicate_name(row["name"])
        if is_duplicate:
            continue

        # Filter by view mode if filter_value is provided
        if filter_value is not None:
            if view_mode == "sphere" and "n" in row and int(row["n"]) != filter_value:
                continue
            elif view_mode == "stem" and int(row["stem"]) != filter_value:
                continue

        # Stem charts stop at the stable edge n = k+2 — stable copies beyond
        # it would just duplicate the last column (STEM_VIEW_SPEC.md).
        if view_mode == "stem" and "n" in row and int(row["n"]) > int(row["stem"]) + 2:
            continue

        # Set coordinates based on view mode
        if view_mode == "sphere":
            x = int(row["stem"])  # s value on x-axis
            y = int(row["Adams filtration"])  # f value on y-axis
        elif view_mode == "stem":
            x = int(row["n"]) if "n" in row else 0  # n value on x-axis
            y = int(row["Adams filtration"])  # f value on y-axis
        else:
            # Default to sphere mode
            x = int(row["stem"])
            y = int(row["Adams filtration"])

        # Stem cutoff for the final charts: drop nodes past the boundary column
        # (sphere charts plot stem on x). Nodes at exactly max_stem are kept;
        # differentials landing on that column from the off-chart stem max_stem+1
        # are drawn as incoming half-lines (see edges_to_json).
        if max_stem is not None and max_stem > 0 and view_mode == "sphere" and x > max_stem:
            continue

        # Filtration cutoff: the total-degree window supports filtration only
        # up to `max_filt` at the boundary stem, so cap every column there for
        # a uniform top edge (dangling edges to dropped nodes are skipped).
        if max_filt is not None and view_mode == "sphere" and y > max_filt:
            continue

        label = try_get_key(row, "label", "")
        if not label:
            label = ""

        node_data = {
            "x": x,
            "y": y,
            "label": label,
        }
        if try_get_key(row, "weight", None):
            node_data["label"] += (
                f"    ({row['weight']})" if node_data["label"] else f"({row['weight']})"
            )
        if try_get_key(row, "shift", None):
            node_data["position"] = int(row["shift"])
        
        if attributes := extract_node_attributes(row, view_mode, hit_map):
            # Only add an attributes key if there are attributes to add
            node_data["attributes"] = attributes

        # Highlight mode: fade every node outside the computed target set so
        # the map image stands out (the .faded class is styled in main.py).
        if highlight_targets is not None and node_name not in highlight_targets:
            node_data.setdefault("attributes", []).append("faded")

        # Read the source-node map targets for the requested map column (E, H,
        # P, C2, or the legacy J), used by the side-by-side "map image" viewer.
        # Targets are the class node-IDs in the destination chart, ';'-separated.
        if map_column:
            map_value = try_get_key(row, map_column, "")
            if map_value and str(map_value).strip():
                targets = [t.strip() for t in str(map_value).strip().split(";") if t.strip()]
                if len(targets) == 1:
                    node_data["map"] = targets[0]
                elif len(targets) > 1:
                    node_data["map"] = targets

        nodes[node_name] = node_data
    return nodes


def _row_stem(row):
    try:
        return int(row["stem"])
    except (KeyError, ValueError, TypeError):
        return None


def _add_incoming_halfedges(row, nodes, edges, max_stem):
    """Draw a d_r whose source is one stem past the cutoff (stem = max_stem+1)
    and whose target lands on the boundary column (stem = max_stem) as an
    incoming half-line: anchored at the on-chart target with offset (+1, -r)
    toward the off-chart source, so the segment hits the target then runs off
    the right edge. r is the differential length = target_f - source_f."""
    target_value = try_get_key(row, "drtarget", "")
    if not target_value or str(target_value) == "nan":
        return
    try:
        src_f = int(row["Adams filtration"])
    except (KeyError, ValueError, TypeError):
        return
    for target_node in str(target_value).split(";"):
        if target_node not in nodes:
            continue
        tgt = nodes[target_node]
        if tgt.get("x") != max_stem:
            continue  # only half-lines that terminate on the boundary column
        r = tgt.get("y", 0) - src_f
        if r <= 0:
            continue
        edges.append({
            "source": target_node,
            "offset": {"x": 1, "y": -r},
            "attributes": [f"d{r}", "incoming"],
        })


def edges_to_json(df, nodes, view_mode="sphere", max_stem=None):
    """Build the "edges" JSON list from the CSV frame, restricted to `nodes`."""
    edges = []
    # Define which edge types to process based on view mode
    if view_mode == "sphere":
        edge_types = ["h0", "h1", "h2", "dr", "nulldif"]
    elif view_mode == "stem":
        edge_types = ["h0", "E"]
    else:
        edge_types = ["h0", "h1", "h2", "dr", "nulldif"]  # default to sphere
        
    for _, row in df.iterrows():
        node_name, _ = deduplicate_name(row["name"])
        if node_name not in nodes:
            # Row is outside the current slice; without this, out-of-slice
            # sources would emit dangling edges (e.g. stem-mode arrows).
            # EXCEPTION: an off-chart source one stem past the cutoff whose d_r
            # lands on the boundary column is drawn as an incoming half-line.
            if (
                max_stem is not None and max_stem > 0 and view_mode == "sphere"
                and _row_stem(row) == max_stem + 1
            ):
                _add_incoming_halfedges(row, nodes, edges, max_stem)
            continue
        for edge_type in edge_types:
            # Handle nulldif special case - column is named "nulldif" not "nulldiftarget"
            if edge_type == "nulldif":
                target_col = "nulldif"
                info_col = f"{edge_type}info"
            elif edge_type == "E":
                # EHP CSVs store suspension targets in the "E" column
                target_col = "E" if "E" in row else "Etarget"
                info_col = "Einfo"
            else:
                target_col = f"{edge_type}target"
                info_col = f"{edge_type}info"
            if target_col not in row:
                # In some CSVs, the target column is missing completely
                continue
            target_value = row[target_col]
            if target_value == "":
                continue  # Skip processing if the value is NaN or empty
            if str(target_value) == "nan":
                continue

            # Removed verbose print for batch processing - uncomment for debugging
            # print(f"{node_name} targeting {str(target_value)}")
            for target_node in str(target_value).split(";"):
                target_info = try_get_key(row, info_col, "")
                edge_data = {"source": node_name}
                if pd.notna(target_node):
                    if target_node in nodes:
                        # This is a structline
                        edge_data["target"] = target_node
                    elif target_node == "loc" or target_info == "loc":
                        # This is an arrow
                        edge_data["offset"] = edge_offset(edge_type, arrow_length)
                    elif edge_type == "E" and view_mode == "stem":
                        # Suspension continues beyond the displayed range
                        # (past the stable edge): short right-arrow instead.
                        edge_data["offset"] = edge_offset(edge_type, arrow_length)
                    else:
                        # A target filtered out by the stem cutoff (off-chart)
                        # is expected, not an error — skip it quietly. Targets
                        # that are genuinely missing (not merely off-chart) still
                        # report so real data problems stay visible.
                        tcoords = parse_node_coordinates(target_node)
                        if not (max_stem and tcoords and tcoords["stem"] > max_stem):
                            print(
                                f"Invalid target node: ({target_node}) for {edge_type} on ({node_name})"
                            )
                        continue
                elif target_info == "free" or target_info == "loc":
                    # This is also an arrow, but with a different notation
                    edge_data["offset"] = edge_offset(edge_type, arrow_length)
                else:
                    # no edge to be drawn
                    continue

                if attributes := extract_edge_attributes(row, edge_type, nodes=nodes):
                    # Only add an attributes key if there are attributes to add
                    edge_data["attributes"] = attributes

                edges.append(edge_data)
        # Check for `tauextn` if we're printing an E_infinity page
        if target_node := try_get_key(row, "tauextn"):
            if target_node in nodes:
                edges.append(
                    {
                        "source": node_name,
                        "target": target_node,
                        "attributes": ["tauextn"],
                    }
                )
    return edges


def extract_highlight_targets(df, highlight_mode, source_n):
    """Extract the target nodes for highlighting based on the mode."""
    targets = set()
    
    if highlight_mode == "E-forward":
        # E column contains targets in n+1 sphere (forward: where current sphere maps TO)
        for _, row in df.iterrows():
            if int(row["n"]) == source_n:
                e_targets = try_get_key(row, "E", "")
                if e_targets and not pd.isna(e_targets):
                    # Split on ';' (multi-image delimiter, matching the map column format)
                    for target in str(e_targets).split(';'):
                        target = target.strip()
                        if target:
                            targets.add(target)
    
    elif highlight_mode == "E-backward":
        # Find elements that map TO current sphere via E-map (backward: what maps INTO current sphere)
        for _, row in df.iterrows():
            e_targets = try_get_key(row, "E", "")
            if e_targets and not pd.isna(e_targets):
                # Split on ';' and check if any target is in our target sphere
                for target in str(e_targets).split(';'):
                    target = target.strip()
                    if target:
                        # Parse target to check if it's in our sphere
                        target_parts = target.split('_')
                        if len(target_parts) >= 2 and target_parts[0].isdigit():
                            target_n = int(target_parts[0])
                            if target_n == source_n:  # This element maps into our sphere
                                source_name = row["name"]
                                if pd.notna(source_name):
                                    targets.add(str(source_name).strip())
    
    elif highlight_mode == "H-forward":
        # H column contains targets in 2*n-1 sphere (forward: where current sphere maps TO)
        for _, row in df.iterrows():
            if int(row["n"]) == source_n:
                h_targets = try_get_key(row, "H", "")
                if h_targets and not pd.isna(h_targets):
                    for target in str(h_targets).split(';'):
                        target = target.strip()
                        if target:
                            targets.add(target)
    
    elif highlight_mode == "H-backward":
        # Find elements that map TO current sphere via H-map (backward: what maps INTO current sphere)
        for _, row in df.iterrows():
            h_targets = try_get_key(row, "H", "")
            if h_targets and not pd.isna(h_targets):
                for target in str(h_targets).split(';'):
                    target = target.strip()
                    if target:
                        # Parse target to check if it's in our sphere
                        target_parts = target.split('_')
                        if len(target_parts) >= 2 and target_parts[0].isdigit():
                            target_n = int(target_parts[0])
                            if target_n == source_n:  # This element maps into our sphere
                                source_name = row["name"]
                                if pd.notna(source_name):
                                    targets.add(str(source_name).strip())
    
    elif highlight_mode == "P-forward":
        # P column contains targets: odd n ≥ 5 → (n-1)/2 sphere
        if source_n % 2 == 1 and source_n >= 5:  # Only for odd n ≥ 5
            for _, row in df.iterrows():
                if int(row["n"]) == source_n:
                    p_targets = try_get_key(row, "P", "")
                    if p_targets and not pd.isna(p_targets):
                        for target in str(p_targets).split(';'):
                            target = target.strip()
                            if target:
                                targets.add(target)
    
    elif highlight_mode == "P-backward":
        # Find elements that map TO current sphere via P-map (backward: what maps INTO current sphere)
        # P-map: odd n → (n-1)/2, so to find what maps to even source_n, look for odd n = 2*source_n + 1
        target_odd_n = 2 * source_n + 1
        for _, row in df.iterrows():
            if int(row["n"]) == target_odd_n:  # Look in the odd sphere that maps to us
                p_targets = try_get_key(row, "P", "")
                if p_targets and not pd.isna(p_targets):
                    for target in str(p_targets).split(';'):
                        target = target.strip()
                        if target:
                            # Parse target to check if it's in our sphere
                            target_parts = target.split('_')
                            if len(target_parts) >= 2 and target_parts[0].isdigit():
                                target_n = int(target_parts[0])
                                if target_n == source_n:  # This element maps into our sphere
                                    source_name = row["name"]
                                    if pd.notna(source_name):
                                        targets.add(str(source_name).strip())

    elif highlight_mode == "J-forward":
        # J column contains targets in S^n sphere (forward: where SO(n) maps TO)
        for _, row in df.iterrows():
            if int(row["n"]) == source_n:
                j_targets = try_get_key(row, "J", "")
                if j_targets and not pd.isna(j_targets):
                    for target in str(j_targets).split(';'):
                        target = target.strip()
                        if target:
                            targets.add(target)

    return targets



def _compute_highlight_targets(highlight_mode, source_csv, filter_value):
    """
    Load the highlight-source CSV and collect the node names to highlight.

    `filter_value` is the n of the chart being drawn; the source sphere n is
    derived from it according to the map (E: n±1, H: n <-> 2n-1, P: odd
    n <-> (n-1)/2, J: same n) and the direction encoded in `highlight_mode`.
    Returns None when highlighting is not requested or no source n applies.
    """
    if not (highlight_mode and source_csv):
        return None

    # Load source CSV to get the mapping data
    source_df = pd.read_csv(source_csv)
    # Determine source_n based on the highlight mode and direction
    if highlight_mode == "E-forward" and filter_value:
        source_n = filter_value - 1  # Forward: coming from n-1 to n
    elif highlight_mode == "E-backward" and filter_value:
        source_n = filter_value + 1  # Backward: coming from n+1 to n
    elif highlight_mode == "H-forward" and filter_value:
        # Forward: coming from n to 2n-1, so source_n = (filter_value + 1) // 2
        source_n = (filter_value + 1) // 2
    elif highlight_mode == "H-backward" and filter_value:
        # Backward: coming from (n+1)/2 to n, so source_n = 2*filter_value - 1
        source_n = 2 * filter_value - 1
    elif highlight_mode == "P-forward" and filter_value:
        # Forward: coming from odd n ≥ 5 to (n-1)/2, so source_n = 2*filter_value + 1
        source_n = 2 * filter_value + 1
    elif highlight_mode == "P-backward" and filter_value:
        # Backward: coming from even n to 2n+1, so source_n = (filter_value - 1) / 2
        source_n = (filter_value - 1) // 2
    elif highlight_mode == "J-forward" and filter_value:
        # Forward: J-map from SO(n) to S^n, so source_n = filter_value (same n)
        source_n = filter_value
    else:
        source_n = filter_value

    if source_n:
        return extract_highlight_targets(source_df, highlight_mode, source_n)
    return None


def _build_chart_json(input_file, view_mode, filter_value, highlight_mode, source_csv, map_column, max_stem, max_filt=None):
    """Read a chart CSV and assemble the (not yet validated) chart JSON document."""
    df = pd.read_csv(input_file)

    # Build a minimal header - let main.py handle theming
    header = {}

    highlight_targets = _compute_highlight_targets(highlight_mode, source_csv, filter_value)

    # Process nodes first
    nodes = nodes_to_json(df, view_mode, filter_value, highlight_mode, highlight_targets, map_column, max_stem, max_filt)

    # Process edges after, since they depend on nodes
    edges = edges_to_json(df, nodes, view_mode, max_stem)

    # Combine the data into a single JSON object
    return {
        "$schema": "https://raw.githubusercontent.com/JoeyBF/SeqSee/refs/heads/master/seqsee/input_schema.json",
        "header": header,
        "nodes": nodes,
        "edges": edges,
    }


def process_csv(input_file, output_file, view_mode="sphere", filter_value=None, highlight_mode=None, source_csv=None, quiet=False, map_column=None, max_stem=None, max_filt=None):
    """Convert a chart CSV to a validated chart JSON file at `output_file`."""
    schema = load_schema()
    json_data = _build_chart_json(
        input_file, view_mode, filter_value, highlight_mode, source_csv, map_column, max_stem, max_filt
    )

    # Validation and output
    try:
        validate(instance=json_data, schema=schema)
        formatter = Formatter()
        formatter.indent_spaces = 2
        formatter.dump(json_data, output_file)
        if not quiet:
            print("JSON data successfully generated and validated against the schema.")
        return json_data
    except Exception as e:
        if not quiet:
            print("Validation error:", e)
        raise e


def process_csv_to_dict(input_file, view_mode="sphere", filter_value=None, highlight_mode=None, source_csv=None, map_column=None, max_stem=None, max_filt=None):
    """
    Process CSV to JSON data structure without writing to file.
    Useful for batch processing and multi-chart generation.
    """
    schema = load_schema()
    json_data = _build_chart_json(
        input_file, view_mode, filter_value, highlight_mode, source_csv, map_column, max_stem, max_filt
    )

    # Validation
    try:
        validate(instance=json_data, schema=schema)
        return json_data
    except Exception as e:
        print(f"Validation error: {e}")
        raise e


def batch_process_csv(input_file, output_dir, view_modes=None, filter_ranges=None, quiet=False):
    """
    Batch process a single CSV file into multiple JSON outputs.
    
    Args:
        input_file: Path to CSV file
        output_dir: Directory to write JSON files 
        view_modes: List of view modes ["sphere", "stem"] (default: ["sphere"])
        filter_ranges: Dict with ranges for each view mode (default: sphere=2-72, stem=0-10)
        quiet: If True, suppress progress output
    
    Returns:
        List of generated file paths
    """
    if view_modes is None:
        view_modes = ["sphere"]
    
    if filter_ranges is None:
        filter_ranges = {
            "sphere": range(2, 73),  # S2 through S72
            "stem": range(0, 11)     # stem 0 through 10
        }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(input_file)
    # Extract E number from filename (e.g., E2.csv -> 2)
    er_num = int(input_path.stem[1:]) if input_path.stem.startswith('E') and input_path.stem[1:].isdigit() else 2
    
    generated_files = []
    
    for view_mode in view_modes:
        if view_mode not in filter_ranges:
            continue
            
        for filter_value in filter_ranges[view_mode]:
            if view_mode == "sphere":
                output_file = output_dir / f"S{filter_value}_E{er_num}.json"
            else:
                output_file = output_dir / f"S{filter_value}_E{er_num}_{view_mode}.json"
            
            try:
                process_csv(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    view_mode=view_mode,
                    filter_value=filter_value,
                    quiet=quiet
                )
                generated_files.append(str(output_file))
                if not quiet:
                    print(f"  Generated {output_file.name}")
            except Exception as e:
                if not quiet:
                    print(f"  Failed to generate {output_file.name}: {e}")
    
    return generated_files


def main():
    # Pull the optional `--map <COL>` flag out first so it composes with the
    # existing positional call style used by generate_charts.py. When set,
    # source-node targets from that CSV column are embedded as node "map" data
    # for the side-by-side "map image" viewer (COL is E, H, P, C2, or J).
    argv = sys.argv[1:]
    map_column = None
    if "--map" in argv:
        i = argv.index("--map")
        map_column = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]

    # Optional stem cutoff for the final charts: drop nodes past stem max_stem
    # and draw incoming differentials from stem max_stem+1 as half-lines.
    max_stem = None
    if "--max-stem" in argv:
        i = argv.index("--max-stem")
        max_stem = int(argv[i + 1]) if i + 1 < len(argv) and argv[i + 1] != "" else None
        del argv[i:i + 2]
        if max_stem is not None and max_stem <= 0:
            max_stem = None  # 0/negative disables the cutoff (full-width charts)

    max_filt = None
    if "--max-filt" in argv:
        i = argv.index("--max-filt")
        max_filt = int(argv[i + 1]) if i + 1 < len(argv) and argv[i + 1] != "" else None
        del argv[i:i + 2]
        if max_filt is not None and max_filt <= 0:
            max_filt = None

    if len(argv) < 2 or len(argv) > 6:
        print("Usage: jsonmaker <input.csv> <output.json> [view_mode] [filter_value] [highlight_mode] [source_csv] [--map COL]")
        print("  view_mode: 'sphere' or 'stem' (default: sphere)")
        print("  filter_value: integer to filter by (n for sphere, s for stem)")
        print("  highlight_mode: '<MAP>-forward' or '<MAP>-backward' with MAP one of E/H/P (or 'J-forward') (optional)")
        print("  source_csv: CSV file containing the source data for highlighting (optional)")
        print("  --map COL: embed source-node targets from column COL (E/H/P/C2/J)")
        sys.exit(1)

    input_file = argv[0]
    output_file = argv[1]
    view_mode = argv[2] if len(argv) > 2 else "sphere"
    filter_value = int(argv[3]) if len(argv) > 3 and argv[3] != "" else None
    highlight_mode = argv[4] if len(argv) > 4 and argv[4] != "" else None
    source_csv = argv[5] if len(argv) > 5 and argv[5] != "" else None

    process_csv(input_file, output_file, view_mode, filter_value, highlight_mode, source_csv, map_column=map_column, max_stem=max_stem, max_filt=max_filt)


if __name__ == "__main__":
    main()
