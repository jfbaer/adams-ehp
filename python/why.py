#!/usr/bin/env sage-python
"""
Generate the flow chart ("why graph") that proves a differential d_r at
tridegree (n, s, f).

Every constraint the propagator applies to a differential records a proof
reason (see AffineMatrixSubspace.add_reason in lib.py).  This module turns
those reasons into the flow charts described in the paper (see the paper's
flow-chart legend) in three phases:

1. build_proof_tree -- backward-chain through the recorded reasons, producing
   an explicit tree of ProofNodes.  Chains terminate at the assumed inputs:
   the stable Adams charts for the sphere (S(s, f) nodes) and for S/2
   (C2(s, f) nodes, the whole n = 0 column).
2. classification -- each node's state (unconstrained / partial / forced /
   forced zero / trivial target / stable input) is derived from the snapshot
   stored in its proof reason, matching the legend's color semantics.
3. render_dot / render_tikz -- serialize the tree, either as a Graphviz DOT
   file (default; graphs are often large) or as paper-ready tikz.

Run from this directory with:  sage -python why.py <n> <s> <f> [-r PAGE] [-o OUT]

Example:
  sage -python why.py 9 37 7 -r 3          # DOT proof of d3 at (9, 37, 7)
  sage -python why.py 9 37 7 -r 3 --tikz   # same proof as tikz for the paper
"""

import argparse
import ast
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field

from differentials import DifferentialsPage
from lib import STANDARD_MAPS

# Catppuccin Latte palette (single source of truth for the DOT renderer),
# matching the light color scheme the interactive charts use.
PALETTE = {
    "base": "#eff1f5",
    "text": "#4c4f69",
    "peach": "#fe640b",
    "sky": "#04a5e5",
    "green": "#40a02b",
    "overlay2": "#7c7f93",
    "red": "#d20f39",
}

# Node state -> (DOT fill color, tikz fill color).  The states are the ones in
# the paper's flow-chart legend:
#   unconstrained   grey   D = 0 + Hom (leaf never constrained)
#   partial         sky    {0} < I_r < Hom
#   forced          green  D = d_r + {0}
#   forced_zero     red    D = 0 + {0}
#   trivial_target  red    target group is zero
#   stable_input    peach  assumed input (stable S or C2 Adams chart)
STATE_FILL = {
    "unconstrained": ("overlay2", "CtpOverlay2"),
    "partial": ("sky", "CtpSky"),
    "forced": ("green", "CtpGreen"),
    "forced_zero": ("red", "CtpRed"),
    "trivial_target": ("red", "CtpRed"),
    "stable_input": ("peach", "CtpPeach"),
}

# The maps whose naturality steps draw a directed, labeled arrow in the chart.
# E and H point toward the higher sphere, P toward the lower one.
MAP_ARROW_TOWARD_HIGHER_N = ("E", "H")
MAP_ARROW_TOWARD_LOWER_N = ("P",)


@dataclass
class ProofEdge:
    """An edge from a node to one of the earlier deductions supporting it."""

    child: "ProofNode"
    label: str = ""  # "" (unlabeled) or a map name: "E", "H", "P"
    arrow: str = "none"  # "to_child" | "to_parent" | "none"


@dataclass
class ProofNode:
    """One deduction step: a tridegree resolved at a particular proof counter.

    `counter` is the reason-counter that produced this node (its identity, so
    the same deduction reached by two paths is one node); terminals and leaves
    inherit their parent's / -1.

    kind:
      interior        a recorded deduction with supporting children
      leaf            no reason recorded before its lookup bound
      stable_terminal assumed input from the stable Adams chart for the sphere
      c2_terminal     assumed input reached through the C2 map (odd n source)
      c2_input        assumed input: a tridegree on the C2 column itself (n = 0)
    """

    tridegree: tuple
    counter: int
    kind: str
    state: str = "partial"
    edges: list = field(default_factory=list)
    # Label of the reason this node expands ("" = Leibniz; None for leaves
    # and terminals, which have no reason of their own).
    reason_label: str = None
    # A constraint imposed with no supporting deduction (x = y = (0, 0, 0)):
    # an assumed input (hand list / table) when source_label names it, a true
    # act of God when it is empty. Drawn dashed, colored from its snapshot.
    unsourced: bool = False
    source_label: str = ""


@dataclass
class ProofTree:
    root: ProofNode
    proof_length: int
    bidegrees_used: set
    warnings: list


# --- Phase 1: build the proof tree by backward chaining -----------------------


def _latest_reason(d_page, tridegree, counter):
    """The most recent proof reason recorded for `tridegree` strictly before
    `counter`, or None"""
    reasons = [w for w in d_page[tridegree].why if w["counter"] < counter]
    return reasons[-1] if reasons else None


def _classify_from_snapshot(reason, warnings, tridegree):
    """Legend state of an interior node, from the offset/subspace snapshot its
    reason recorded (rules I1-I3)"""
    if "v" not in reason or "subspace" not in reason:
        warnings.append(
            f"{tridegree} <#{reason['counter']}: reason has no state snapshot; "
            "shown as partial"
        )
        return "partial"
    if len(reason["subspace"]) > 0:
        return "partial"
    if reason["v"].is_zero():
        return "forced_zero"
    return "forced"


def _classify_leaf(d_page, tridegree):
    """Legend state of a node with no further deductions (rules L1-L2): red if
    the target group is zero, grey (never constrained) otherwise"""
    if d_page[tridegree].ncols == 0:
        return "trivial_target"
    return "unconstrained"


def _route_label(label, tridegree):
    """The child tridegree a map label belongs on: the map's target degree.
    Returns None for unlabeled reasons or unknown labels."""
    if not label:
        return None
    for map_name, map_obj in STANDARD_MAPS.items():
        if label.startswith(map_name):
            return map_obj.target_degree(*tridegree)
    return None


def _arrow(parent_td, child_td, num_children, label):
    """Arrow direction for an edge, per the legend: E/H arrows point toward the
    higher-n sphere and P toward the lower one; a 2-child step draws an arrow
    to the child unless it is the parent's own earlier state"""
    if label in MAP_ARROW_TOWARD_HIGHER_N:
        return "to_child" if child_td[0] > parent_td[0] else "to_parent"
    if label in MAP_ARROW_TOWARD_LOWER_N:
        return "to_child" if child_td[0] < parent_td[0] else "to_parent"
    if num_children == 2 and child_td != parent_td:
        return "to_child"
    return "none"


def _sorted_for_worklist(children, parent_td):
    """Order children for the LIFO worklist so they pop (= display left to
    right) as x, y, xy for Leibniz steps, and with the parent's own earlier
    state on the right for 2-child steps"""
    if len(children) == 2:
        # same-tridegree child sorts first -> popped last -> rightmost
        return sorted(children, key=lambda c: 0 if c[0] == parent_td else 1)
    role_priority = {"x": 2, "y": 1, "xy": 0}
    return sorted(children, key=lambda c: role_priority[c[2]])


def build_proof_tree(d_page, tridegree):
    """Backward-chain from `tridegree` through the recorded proof reasons and
    return the ProofTree of deductions.

    Each node is identified by the reason that produced it: the pair
    (tridegree, reason-counter).  Its lookup bound (the counter it is reached
    at) selects the latest reason strictly before that bound; the resulting
    reason-counter is the node's identity, so the same deduction reached by two
    different proof paths is drawn once (the tree is a DAG).  A map reason
    (label E/H/P) contributes the map's other endpoint and the node's own
    earlier state as children; a Leibniz reason (empty label) contributes the
    two factors and their product.

    Chains stop at the assumed inputs -- "stable" reasons (stable sphere
    chart), "C2" reasons (chart for S/2), and any tridegree on the n = 0 column
    -- but a single tridegree can carry several such inputs (e.g. a sphere and
    a C2 differential at different counters).  We attach the terminal for the
    latest one and continue chaining the node's own earlier state, so every
    stable/C2 input on a tridegree surfaces (two-source proofs).
    """
    drawn = {}  # (tridegree, reason_counter) -> ProofNode, the DAG's nodes
    proof_length = 0
    bidegrees_used = set()
    warnings = []
    root = None

    # Worklist entries: (tridegree, lookup_bound, parent, edge_label, edge_arrow)
    worklist = [(tuple(tridegree), d_page.counter + 1, None, "", "none")]
    while worklist:
        td, bound, parent, edge_label, edge_arrow = worklist.pop()

        # Resolve the reason that identifies this node before creating it, so
        # its (tridegree, reason-counter) key can merge repeats across paths.
        if td[0] == 0:
            # The n = 0 column is the E_2-page of S/2: assumed input, terminal.
            reason, rc, kind = None, -1, "c2_input"
        else:
            reason = _latest_reason(d_page, td, bound)
            rc = reason["counter"] if reason is not None else -1
            kind = "interior" if reason is not None else "leaf"

        key = (td, rc)
        existing = drawn.get(key)
        if existing is not None:
            # Same deduction already in the chart: draw the edge to it, but do
            # not re-expand its subtree (keeps the DAG finite and de-duplicated).
            if parent is not None:
                parent.edges.append(ProofEdge(existing, edge_label, edge_arrow))
            continue

        node = ProofNode(td, rc, kind=kind)
        drawn[key] = node
        bidegrees_used.add(td)
        if root is None:
            root = node
        else:
            parent.edges.append(ProofEdge(node, edge_label, edge_arrow))

        if kind == "c2_input":
            node.state = "stable_input"
            continue

        if reason is None:
            node.state = _classify_leaf(d_page, td)
            continue

        if reason["label"] in ("stable", "C2"):
            # Assumed input(s): a single tridegree can be pinned by both a
            # stable sphere differential and a C2 differential (a two-source
            # proof, e.g. (9,17,4) in the paper's fig:flowchart2). Attach a
            # terminal for every distinct stable/C2 input recorded at or before
            # this node's bound, then stop -- the value is proved by these
            # assumed inputs. Each source yields one terminal (S(s,f)/C2(s,f)
            # depend only on the tridegree), so dedup by terminal kind.
            proof_length += 1
            node.state = "partial"
            seen_kinds = set()
            for w in d_page[td].why:
                if w["counter"] > bound or w["label"] not in ("stable", "C2"):
                    continue
                kind_t = ("stable_terminal" if w["label"] == "stable"
                          else "c2_terminal")
                if kind_t in seen_kinds:
                    continue
                seen_kinds.add(kind_t)
                node.edges.append(ProofEdge(
                    ProofNode(td, rc, kind=kind_t, state="stable_input"),
                    "", "to_child"))
            continue

        proof_length += 1
        node.state = _classify_from_snapshot(reason, warnings, td)
        node.reason_label = reason["label"]

        x = (reason["x_n"], reason["x_s"], reason["x_f"])
        y = (reason["y_n"], reason["y_s"], reason["y_f"])
        xy = (reason["x_n"], reason["x_s"] + reason["y_s"],
              reason["x_f"] + reason["y_f"])

        # A map reason contributes x and y (the two ends of the map); a
        # Leibniz reason (empty label) also contributes the product xy.
        # (0, 0, 0) marks an unused slot.
        label_target = _route_label(reason["label"], td)
        children = []
        label_assigned = False
        for child_td, role in ((x, "x"), (y, "y"), (xy, "xy")):
            if child_td == (0, 0, 0):
                continue
            if role == "xy" and reason["label"] != "":
                continue
            child_label = ""
            if child_td == label_target and not label_assigned:
                child_label = reason["label"]
                label_assigned = True
            children.append((child_td, child_label, role))
        if reason["label"] and not label_assigned and children:
            # No child matches the map's target degree (a map reason names
            # its source, whose target lies elsewhere); label the first child
            children[0] = (children[0][0], reason["label"], children[0][2])

        if not children:
            # A constraint with no supporting deduction: a named assumed input
            # (label "unstable"/"spurious"/...) or, when unlabeled, a true act
            # of God. Either way the node keeps its snapshot state (a forced
            # value must not display as unconstrained) and is drawn dashed.
            if x == (0, 0, 0) and y == (0, 0, 0) and not reason["label"]:
                warnings.append(
                    f"{td} <#{reason['counter']}: constraint with no recorded "
                    "source"
                )
            node.unsourced = True
            node.source_label = reason["label"]
            continue

        for child_td, child_label, _ in _sorted_for_worklist(children, td):
            arrow = _arrow(td, child_td, len(children), child_label)
            worklist.append((child_td, reason["counter"], node, child_label, arrow))

    # An interior node that produced no edges of its own is shown with the
    # leaf semantics the legend defines -- except unsourced constraints, whose
    # snapshot state is authoritative.
    for node in drawn.values():
        if node.kind == "interior" and not node.edges and not node.unsourced:
            node.state = _classify_leaf(d_page, node.tridegree)

    # Prune uninformative self-references. A non-Leibniz reason (map /
    # stable / d^2 / ...) always cites the node's own earlier state, but
    # when that state had recorded no deduction yet it is just the freshly
    # initialized full space and adds nothing to the proof. Keep the
    # same-tridegree child only for a Leibniz step (label "", where it is
    # the factor x) or when the earlier state genuinely restricted the
    # differential (kind "interior"). Edge-level, because a (td, -1) leaf
    # can also be a legitimate map endpoint reached by a labeled edge.
    for node in drawn.values():
        node.edges = [
            e for e in node.edges
            if not (e.label == ""
                    and e.child.tridegree == node.tridegree
                    and e.child.kind == "leaf"
                    and node.reason_label)
        ]

    # A same-tridegree connector that merely re-reaches an earlier state
    # already derived elsewhere in the proof adds nothing (the shared
    # tridegree is visible from the node labels), so drop it when the
    # predecessor stays reachable without it. Connectors that are the only
    # path into their sub-proof stay, as do Leibniz factor edges.
    def _reachable_without(skip_parent, skip_edge):
        seen, stack = set(), [root]
        while stack:
            nd = stack.pop()
            if id(nd) in seen:
                continue
            seen.add(id(nd))
            for e in nd.edges:
                if nd is skip_parent and e is skip_edge:
                    continue
                stack.append(e.child)
        return seen

    if root is not None:
        for node in list(drawn.values()):
            for e in list(node.edges):
                if (e.label == ""
                        and e.child.tridegree == node.tridegree
                        and e.child.kind == "interior"
                        and node.reason_label
                        and id(e.child) in _reachable_without(node, e)):
                    node.edges.remove(e)

    return ProofTree(root, proof_length, bidegrees_used, warnings)


# --- Phase 2: node presentation shared by both renderers ----------------------


def display_label(node):
    """The chart label of a node: (n, s, f) for spheres, C2(s, f) on the C2
    column, S(s, f) / C2(s, f) for the assumed-input terminals"""
    n, s, f = node.tridegree
    if node.kind == "stable_terminal":
        return f"S({s}, {f})"
    if node.kind == "c2_terminal":
        _, ts, tf = STANDARD_MAPS["C2"].target_degree(n, s, f)
        return f"C2({ts}, {tf})"
    if n == 0:
        return f"C2({s}, {f})"
    return str(node.tridegree)


def _dot_id(node):
    if node.kind == "stable_terminal":
        return f"Stable_{node.tridegree}_{node.counter}"
    if node.kind == "c2_terminal":
        return f"C2_{node.tridegree}_{node.counter}"
    return f"{node.tridegree}_{node.counter}"


def walk(tree):
    """Yield (parent, edge, node) in DFS pre-order, children left to right.

    The proof structure is a DAG (a node may be reached by several edges): every
    edge is yielded so all arrows are drawn, but each node's children are pushed
    only the first time it is seen, so shared subtrees are not re-expanded.
    Used by render_dot below and by make_why_graphs.py to count nodes."""
    seen = set()
    stack = [(None, None, tree.root)]
    while stack:
        parent, edge, node = stack.pop()
        yield parent, edge, node
        if id(node) in seen:
            continue
        seen.add(id(node))
        for e in reversed(node.edges):
            stack.append((node, e, e.child))


# --- Phase 3a: DOT renderer ---------------------------------------------------


def render_dot(tree):
    """Serialize a ProofTree as a Graphviz DOT digraph"""
    lines = [
        "digraph why_graph {",
        "    rankdir=TB;",
        f'    bgcolor="{PALETTE["base"]}";',
        "    fontname=Arial;",
        "    fontsize=12;",
        "",
        "    // Node styling defaults",
        f'    node [shape=box, style="filled,rounded", fontname=Arial, '
        f'fontsize=10, fontcolor="{PALETTE["base"]}"];',
        f'    edge [fontname=Arial, fontsize=10, fontcolor="{PALETTE["text"]}", '
        f'color="{PALETTE["text"]}"];',
        "",
    ]
    text = PALETTE["text"]

    # build_proof_tree already merges repeats into a single node per
    # (tridegree, reason-counter), so each id needs defining exactly once.
    defined = set()
    for parent, edge, node in walk(tree):
        node_id = _dot_id(node)
        if node_id not in defined:
            defined.add(node_id)
            fill = PALETTE[STATE_FILL[node.state][0]]
            label = display_label(node)
            style_attr = ""
            if node.unsourced:
                # Assumed input / act of God: dashed border, named by its label
                if node.source_label:
                    label += f"\\n[{node.source_label}]"
                style_attr = ', style="filled,rounded,dashed"'
            lines.append(
                f'    "{node_id}" [label="{label}", '
                f'fillcolor="{fill}"{style_attr}];'
            )
        if parent is not None:
            if edge.label:
                attrs = "" if edge.arrow == "to_child" else ", dir=back"
                lines.append(
                    f'    "{_dot_id(parent)}" -> "{node_id}" '
                    f'[label="{edge.label}", color="{text}", '
                    f'fontname="Arial Bold", fontsize=14{attrs}];'
                )
            else:
                attrs = "" if edge.arrow == "to_child" else ", arrowhead=none"
                lines.append(
                    f'    "{_dot_id(parent)}" -> "{node_id}" '
                    f'[color="{text}"{attrs}];'
                )
    lines.append("}")
    return "\n".join(lines) + "\n"


# --- Phase 3b: tikz renderer --------------------------------------------------

TIKZ_COLUMN_WIDTH = 2.3  # cm between sibling columns (wide enough for tridegree labels)
TIKZ_ROW_HEIGHT = 1.2  # cm between proof levels


def _tikz_label(node):
    n, s, f = node.tridegree
    if node.kind == "stable_terminal":
        return f"\\mathbb{{S}}({s}, {f})"
    if node.kind == "c2_terminal":
        _, ts, tf = STANDARD_MAPS["C2"].target_degree(n, s, f)
        return f"C2({ts}, {tf})"
    if n == 0:
        return f"C2({s}, {f})"
    return str(node.tridegree)


def _tikz_id(node):
    n, s, f = node.tridegree
    if node.kind == "stable_terminal":
        return f"n_Stable_{n}_{s}_{f}_{node.counter}"
    if node.kind == "c2_terminal":
        return f"n_C2_{n}_{s}_{f}_{node.counter}"
    return f"n_{n}_{s}_{f}_{node.counter}"


def render_tikz(tree):
    """Serialize a ProofTree in the paper's tikz style (ctpnode / CtpText).
    The proof is a DAG: a node reached by several edges is placed once and the
    later edges draw a connector to it.  Beware: charts much larger than the
    paper's examples are usually too big for tikz."""

    # Subtree width in columns, so siblings never overlap.  A shared node is
    # measured once (memoized on id) and its width attributed to its first
    # placement; the placeholder guards against cycles from back-references.
    width = {}

    def measure(node):
        if id(node) in width:
            return width[id(node)]
        width[id(node)] = 1  # placeholder while recursing (breaks cycles)
        if node.edges:
            width[id(node)] = max(1, sum(measure(e.child) for e in node.edges))
        return width[id(node)]

    measure(tree.root)

    lines = ["\\begin{tikzpicture}[>=latex,line join=bevel]"]
    node_lines = []
    edge_lines = []
    placed = set()

    def place(node, left, depth):
        """Assign positions with `node` centered over its children, which
        occupy columns starting at `left`.  A node already placed via another
        edge keeps its first position; only the connector is drawn."""
        if id(node) in placed:
            # Reached again via another edge; the connector to it is drawn by
            # whichever parent owns that edge, so nothing to do here.
            return
        placed.add(id(node))
        x = (left + width[id(node)] / 2 - 0.5) * TIKZ_COLUMN_WIDTH
        y = -depth * TIKZ_ROW_HEIGHT
        fill = STATE_FILL[node.state][1]
        dashed = ",dashed" if node.unsourced else ""
        node_lines.append(
            f"      \\node ({_tikz_id(node)}) at ({x:g},{y:g}) "
            f"[ctpnode,fill={fill}{dashed}] {{$\\mathbf{{{_tikz_label(node)}}}$}};"
        )
        child_left = left
        for e in node.edges:
            place(e.child, child_left, depth + 1)
            child_left += width[id(e.child)]
            if e.arrow == "to_child":
                arrow = ", ->"
            elif e.arrow == "to_parent":
                arrow = ", <-"
            else:
                arrow = ""
            if e.label:
                edge_lines.append(
                    f"      \\draw [CtpText{arrow}, very thick] "
                    f"({_tikz_id(node)}) to[out=270,in=90] "
                    f"node [midway,left,CtpText,inner sep=2pt]\n"
                    f"    {{$\\mathbf{{{e.label}}}$}} ({_tikz_id(e.child)});"
                )
            else:
                edge_lines.append(
                    f"      \\draw [CtpText{arrow}, very thick] "
                    f"({_tikz_id(node)}) to[out=270,in=90] "
                    f"({_tikz_id(e.child)});"
                )

    place(tree.root, 0, 0)
    lines.extend(node_lines)
    lines.append("")
    lines.extend(edge_lines)
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


# --- Public API and CLI -------------------------------------------------------


def write_why_graph(d_page, tridegree, filename=None, fmt="dot"):
    """Build the proof flow chart for the differential at `tridegree` and
    write it to `filename` in the given format ("dot" or "tikz"); returns the
    root ProofNode and the number of deduction steps"""
    tree = build_proof_tree(d_page, tuple(tridegree))
    for warning in tree.warnings:
        print(f"note: {warning}")
    print(f"Bidegrees used: {sorted(tree.bidegrees_used)}")

    if filename is None:
        n, s, f = tridegree
        filename = f"why_{n}_{s}_{f}." + ("tex" if fmt == "tikz" else "dot")
    content = render_tikz(tree) if fmt == "tikz" else render_dot(tree)
    with open(filename, "w") as fp:
        fp.write(content)
    print(f"Graph saved as {filename}")
    return tree.root, tree.proof_length


def dimensions_from_json(obj, r):
    """Rebuild {(n, s, f): dimension} from a saved differential JSON.

    Each entry stores its differential matrix as nrows (source dimension) x
    ncols (target dimension), so the dimensions the run actually used can be
    recovered without the page data.
    """
    dims = defaultdict(int)
    for key, entry in obj.items():
        if key == "r":
            continue
        n, s, f = ast.literal_eval(key)
        dims[n, s, f] = entry["nrows"]
        dims[n, s - 1, f + r] = entry["ncols"]
    return dims


def main():
    parser = argparse.ArgumentParser(
        description="Generate a flow chart proving the differential d_r(n, s, f)."
    )
    parser.add_argument("n", type=int, help="grade n")
    parser.add_argument("s", type=int, help="stem s")
    parser.add_argument("f", type=int, help="Adams filtration f")
    parser.add_argument("-r", "--page", type=int, default=2,
                        help="page number r (loads data/E{r}/d{r}, default 2)")
    parser.add_argument("-d", "--d-file", default=None,
                        help="differential JSON file (default: data/E{page}/d{page})")
    parser.add_argument("-o", "--output", default=None,
                        help="output file (default: why_{n}_{s}_{f}.dot or .tex)")
    parser.add_argument("--tikz", action="store_true",
                        help="emit paper-ready tikz instead of Graphviz DOT")
    args = parser.parse_args()

    d_file = args.d_file or f"data/E{args.page}/d{args.page}"
    print(f"Loading differentials from {d_file}...")
    with open(d_file) as fp:
        obj = json.load(fp)
    dims = dimensions_from_json(obj, int(obj["r"]))
    d = DifferentialsPage.from_json(obj, dimension_dict=dims)

    tridegree = (args.n, args.s, args.f)
    fmt = "tikz" if args.tikz else "dot"
    out = args.output or (f"why_{args.n}_{args.s}_{args.f}."
                          + ("tex" if args.tikz else "dot"))
    _, proof_length = write_why_graph(d, tridegree, filename=out, fmt=fmt)
    print(f"Wrote {out} (proof length: {proof_length})")

    if fmt == "dot":
        if shutil.which("dot"):
            pdf = out.rsplit(".", 1)[0] + ".pdf"
            subprocess.run(["dot", "-Tpdf", out, "-o", pdf], check=True)
            print(f"Rendered {pdf}")
        else:
            print("graphviz not found; render the DOT file with `dot -Tpdf`, or paste")
            print("its contents at https://dreampuf.github.io/GraphvizOnline/")


if __name__ == "__main__":
    main()
