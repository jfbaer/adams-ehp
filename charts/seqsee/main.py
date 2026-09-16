"""HTML chart renderer for the EHP spectral-sequence charts (adapted from SeqSee).

Reads a chart JSON document (validated against input_schema.json), lays out
nodes and edges, and renders a standalone interactive HTML page through the
Jinja templates in this directory. Three CLI modes:

    main.py <input.json> <output.html> [theme] [view_mode] [filter_value]
    main.py --multi <output.html> <chart1.json> [chart2.json] ... [theme]
    main.py --sidebyside <source.json> <target.json> <output.html>
            [theme] [back_url] [map_key] [source_title] [target_title]

Theming is pure CSS: every theme in themes.json becomes a `:root[data-theme]`
CSS-variable block (see build_theme_css), so switching themes at runtime is a
single attribute flip with no regeneration.
"""

import copy
import json
import math
import os
import re
import sys
from collections import defaultdict

import jsonschema
from jinja2 import Environment, FileSystemLoader
from jsonschema.exceptions import ValidationError

# The distance between successive x or y coordinates. Units are in pixels. This will be fixed
# throughout the html file, but zooming is implemented through a transformation matrix applied to
# the <g> element that contains the nodes, edges, and background grid.
scale = None


# Lifted/adapted from MIT-licensed https://github.com/slacy/pyssed/
class CssStyle:
    """A list of CSS styles, but stored as a dict.
    Can contain nested styles."""

    def __init__(self, *args, **kwargs):
        self._styles = {}
        for a in args:
            self.append(a)

        for name, value in kwargs.items():
            self._styles[name] = value

    def __getitem__(self, key):
        return self._styles[key]

    def keys(self):
        """Return keys of the style dict."""
        return self._styles.keys()

    def items(self):
        """Return iterable contents."""
        return self._styles.items()

    def append(self, other):
        """Append style 'other' to self."""
        self._styles = self.__add__(other)._styles

    def __add__(self, other):
        """Add self and other, and return a new style instance."""
        summed = copy.deepcopy(self)
        if isinstance(other, str):
            single = other.split(":")
            summed._styles[single[0]] = single[1]
        elif isinstance(other, dict):
            summed._styles.update(other)
        elif isinstance(other, CssStyle):
            summed._styles.update(other._styles)
        else:
            raise TypeError(f"Bad type for style: {type(other)!r}")
        return summed

    def __repr__(self):
        return str(self._styles)

    def generate(self, parent="", indent=4):
        """Given a dict mapping CSS selectors to a dict of styles, generate a
        list of lines of CSS output."""
        subnodes = []
        stylenodes = []
        result = []

        for name, value in self.items():
            # If the sub node is a sub-style...
            if isinstance(value, dict):
                subnodes.append((name, CssStyle(value)))
            elif isinstance(value, CssStyle):
                subnodes.append((name, value))
            # Else, it's a string, and thus, a single style element
            elif (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
            ):
                stylenodes.append((name, value))
            else:
                raise TypeError(f"Bad type for style value: {type(value)!r}")

        if stylenodes:
            result.append(parent.strip() + " {")
            for stylenode in stylenodes:
                attribute = stylenode[0].strip(" ;:")
                if isinstance(stylenode[1], str):
                    # string
                    value = stylenode[1].strip(" ;:")
                else:
                    # everything else (int or float, likely)
                    value = str(stylenode[1]) + "px"

                result.append(" " * indent + "%s: %s;" % (attribute, value))

            result.append("}")
            result.append("")  # a newline

        for subnode in subnodes:
            result += subnode[1].generate(
                parent=(parent.strip() + " " + subnode[0]).strip()
            )

        if parent == "":
            ret = "\n".join(result)
        else:
            ret = result

        return ret


# Theme palettes live in themes.json — the single source of truth shared with
# the EHP REPL (ehp_chart.rs reads the same file for its differential overlay
# palettes and index page). Add new themes there, not here.
def _load_theme_data():
    themes_path = os.path.join(os.path.dirname(__file__), "themes.json")
    with open(themes_path, "r") as f:
        return json.load(f)


_THEME_DATA = _load_theme_data()

# Cycling order shown in the chart UI.
THEME_ORDER = [t for t in _THEME_DATA["order"] if t in _THEME_DATA["themes"]]

THEME_PALETTES = {
    name: entry["palette"] for name, entry in _THEME_DATA["themes"].items()
}

# Which themes are "dark" (light text on dark background).
DARK_THEMES = {
    name for name, entry in _THEME_DATA["themes"].items() if entry.get("dark")
}

# Human-readable names shown on the chart's theme button.
THEME_LABELS = {
    name: entry.get("label", name) for name, entry in _THEME_DATA["themes"].items()
}

global_css = CssStyle()
# Optional x-axis (stem) cutoff for the final charts; set from the --max-stem
# CLI flag. When set, compute_chart_dimensions clamps the chart width so the
# plot ends cleanly at this stem and incoming half-lines run off the edge.
MAX_STEM = None


def get_theme_colors(theme="light"):
    """Get the color palette for `theme`, falling back to the light palette."""
    return THEME_PALETTES.get(theme, THEME_PALETTES["light"])


def get_themed_color_aliases(theme="light"):
    """Map the legacy color-alias names to concrete colors from `theme`'s palette."""
    colors = get_theme_colors(theme)
    return {
        # Legacy color names mapped to Catppuccin
        "darkcyan": colors["teal"],      # Use teal instead of sapphire for darkcyan
        "darkgreen": colors["green"],
        "gray": colors["text"],          # Use theme text color for default nodes/edges
        "red": colors["red"],
        "blue": colors["blue"],
        "purple": colors["mauve"],
        "magenta": colors["pink"],
        "orange": colors["peach"],
        
        # Theme-specific colors
        "background": colors["base"],
        "text": colors["text"],
        "surface": colors["surface0"],
        "grid": colors["surface1"],
        
        # Differential edge types (d-types, solid lines)
        "d2": colors["teal"],            # d2 differential edges - teal
        "d3": colors["red"],             # d3 differential edges - red
        "d4": colors["green"],           # d4 differential edges - green
        "d5": colors["blue"],            # d5 differential edges - blue
        "d6": colors["yellow"],          # d6 differential edges - yellow
        "d7": colors["peach"],           # d7 differential edges - peach
        "d8": colors["mauve"],           # d8 differential edges - mauve
        
        # Nulldif edge types (n-types, dashed lines)
        "n2": colors["teal"],            # n2 nulldif edges - teal dashed
        "n3": colors["red"],             # n3 nulldif edges - red dashed
        "n4": colors["green"],           # n4 nulldif edges - green dashed
        "n5": colors["blue"],            # n5 nulldif edges - blue dashed
        "n6": colors["yellow"],          # n6 nulldif edges - yellow dashed
        "n7": colors["peach"],           # n7 nulldif edges - peach dashed
        "n8": colors["mauve"],           # n8 nulldif edges - mauve dashed
        
        # Note: nulldifs now use n2, n3, n4, etc. which are already defined above
    }


def build_theme_css(initial_theme="light"):
    """
    Emit one `:root[data-theme="X"]` CSS-variable block per theme, covering both
    the chrome variables (--bg-color etc.) and every themed color alias as
    --cc-<alias>. The generated chart classes reference these variables, so
    switching theme at runtime is a single data-theme attribute flip on <html>
    — no per-element restyling and no regeneration.

    The block for `initial_theme` also matches a bare `:root` so the chart
    renders correctly before any JS runs (flash prevention).
    """
    if initial_theme not in THEME_PALETTES:
        initial_theme = "light"
    blocks = []
    for name in THEME_ORDER:
        pal = THEME_PALETTES[name]
        theme_vars = {
            "--bg-color": pal["base"],
            "--text-color": pal["text"],
            "--surface-color": pal["surface0"],
            "--grid-color": pal["surface1"],
            "--button-bg": pal["surface1"],
            "--button-hover": pal["surface2"],
            "--button-text": pal["text"],
            "--highlight-color": pal["teal"],
            "--accent-color": pal["blue"],
            "--muted-color": pal["subtext0"],
            "--panel-bg": pal["mantle"],
        }
        for alias, color in get_themed_color_aliases(name).items():
            theme_vars[f"--cc-{alias}"] = color
        body = "\n".join(f"      {k}: {v};" for k, v in theme_vars.items())
        # Tell the browser this theme's own base canvas (shown between full-page
        # navigations, before the new page paints) is dark/light — kills the
        # white strobe when stepping through dark charts with w/a/s/d.
        body += f"\n      color-scheme: {'dark' if name in DARK_THEMES else 'light'};"
        selector = f':root[data-theme="{name}"]'
        if name == initial_theme:
            selector = f":root, {selector}"
        blocks.append(f"    {selector} {{\n{body}\n    }}")
    return "\n".join(blocks)


def load_schema():
    schema_path = os.path.join(os.path.dirname(__file__), "input_schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    return schema


schema = load_schema()


def _load_template_file(filename):
    """Load a Jinja template that lives next to this script."""
    env = Environment(loader=FileSystemLoader(searchpath=os.path.dirname(__file__)))
    return env.get_template(filename)


def load_template():
    """Load the single-chart page template."""
    return _load_template_file("template.html.jinja")


def get_schema_default(data, path):
    """
    Get the default value from the schema at the given path.

    This is useful for when we need to know the default value of a field in the schema, but the field
    is not present in the data.
    """

    default_value = schema
    for key in path:
        default_value = default_value["properties"][key]
    return default_value["default"]


def get_value_or_schema_default(data, path):
    """
    Attempt to get a value from `data` at the given path.

    If it is not specified, get the default value from the schema. The schema is always assumed to
    contain a default value for the given path.
    """

    try:
        current_value = data
        for key in path:
            current_value = current_value[key]
        return current_value
    except KeyError:
        return get_schema_default(data, path)


def cssify_name(name):
    """Get a CSS class selector from a name by adding a dot prefix."""
    return "." + name


def style_and_aliases_from_attributes(attributes):
    """
    Given a list of attributes, return a `CssStyle` object that contains the union of all raw
    attribute objects, and a list of aliases.

    We return the aliases separately because we may want to specify them in a `class` attribute
    instead of a `style` attribute.
    """

    new_style = CssStyle()
    aliases = []
    for attr in attributes:
        if isinstance(attr, dict):
            # This is a raw attribute object
            for key, value in attr.items():
                if key == "color":
                    if (value_key := cssify_name(value)) in global_css.keys():
                        # This is a color alias
                        new_style += global_css[value_key]
                    else:
                        # This is a CSS color value
                        new_style += {"fill": value, "stroke": value}
                elif key == "size":
                    new_style += {"r": scale * float(value)}
                elif key == "thickness":
                    new_style += {"stroke-width": scale * float(value)}
                elif key == "fill":
                    if value == "none":
                        new_style += {"fill": "none"}
                    else:
                        new_style += {"fill": value}
                elif key == "arrowTip":
                    if value == "none":
                        new_style += {"marker-end": "none"}
                    else:
                        # We only support a few hardcoded arrow tips. To define a new arrow tip
                        # `foo`, you need to define a `<marker>` element with id `arrow-foo` in the
                        # template file. See the `arrow-simple` marker for an example.
                        new_style += {"marker-end": f"url(#arrow-{value})"}
                elif key == "pattern":
                    # We only support a few hardcoded patterns
                    if value == "solid":
                        new_style += {"stroke-dasharray": "none"}
                    elif value == "dashed":
                        new_style += {"stroke-dasharray": "5, 5"}
                    elif value == "dotted":
                        new_style += {
                            "stroke-dasharray": "0, 2",
                            "stroke-linecap": "round",
                        }
                    # Other values impossible due to schema
                elif key in ["shape", "width", "height", "visibleText"]:
                    # Skip TikZ-specific attributes - these are handled separately in generate_nodes_svg
                    pass
                else:
                    # Just treat the key-value pair as raw CSS
                    new_style += {key: value}
        elif isinstance(attr, str):
            # This is a style alias
            aliases.append(cssify_name(attr).removeprefix("."))
    return (new_style, aliases)


def generate_style(style, aliases):
    """Collapse a list of styles and aliases into a single `CssStyle` object."""
    style = copy.deepcopy(style)
    for alias in aliases:
        style.append(global_css[cssify_name(alias)])
    return style


def ensure_json_path_is_defined(data, path):
    """
    Ensure that the path exists in the JSON data, creating it if necessary.

    This modifies `data` in-place. If the path doesn't already exist, we create a JSON object, which
    is equivalent to a Python `dict`.
    """

    current_value = data
    for key in path:
        if key not in current_value:
            current_value[key] = {}
        current_value = current_value[key]


def compute_chart_dimensions(data):
    """
    This modifies `data` in-place to set up the `header.chart.width` and `header.chart.height`
    objects. Namely, it replaces the `null` values by autodetected boundaries.

    The bounds on the width and height are calculated based on the positions of the nodes in the
    chart. For maximum values, we give the smallest even size that makes the last column/row empty.
    We do the opposite for minimum values. Defaults to a 2x2 first quadrant grid if there are no
    nodes.
    """

    nodes = data.get("nodes", {})

    def compute_dimension_bounds(dim_name, coord_name, default):
        ensure_json_path_is_defined(data, ["header", "chart", dim_name])
        if data["header"]["chart"][dim_name].get("min") is None:
            # Greatest even number strictly smaller than the minimum coordinate of any node
            dimension = 2 * (
                min((node[coord_name] for node in nodes.values()), default=default) // 2
                - 1
            )
            data["header"]["chart"][dim_name]["min"] = dimension
        if data["header"]["chart"][dim_name].get("max") is None:
            # Smallest even number strictly greater than the maximum coordinate of any node
            dimension = 2 * (
                max((node[coord_name] for node in nodes.values()), default=default) // 2
                + 1
            )
            data["header"]["chart"][dim_name]["max"] = dimension

    # Arbitrary default values. These are only used if there are no nodes.
    compute_dimension_bounds("width", "x", 0)
    compute_dimension_bounds("height", "y", 0)

    # Hard stem cutoff for the final charts: end the plot exactly at MAX_STEM so
    # the boundary column sits on the right edge and incoming half-lines (drawn
    # to stem MAX_STEM+1) run off the chart instead of into an empty column.
    if MAX_STEM is not None and MAX_STEM > 0:
        data["header"]["chart"]["width"]["max"] = MAX_STEM


def calculate_absolute_positions(data):
    """
    Compute the final positions of the nodes in the chart.

    This modifies `data` in-place to add attributes `absoluteX` and `absoluteY`. They will be used
    by the SVG generation code to place the nodes at the correct positions and to draw the edges.
    """

    nodes_by_bidegree = defaultdict(list)

    # Group nodes by bidegree
    for node_id, node in data.get("nodes", {}).items():
        x, y = node["x"], node["y"]
        nodes_by_bidegree[x, y].append(node_id)

    # Sort bidegrees by the `position` attribute of the nodes
    default_position = schema["properties"]["nodes"]["additionalProperties"][
        "properties"
    ]["position"]["default"]
    for bidegree, nodes in nodes_by_bidegree.items():
        nodes_by_bidegree[bidegree] = sorted(
            nodes,
            key=lambda node_id: data["nodes"][node_id].get(
                "position", default_position
            ),
        )

    # Get defaults and compute constants
    node_size = get_value_or_schema_default(data, ["header", "chart", "nodeSize"])
    node_spacing = get_value_or_schema_default(data, ["header", "chart", "nodeSpacing"])
    node_slope = get_value_or_schema_default(data, ["header", "chart", "nodeSlope"])

    distance_between_centers = node_spacing + 2 * node_size

    # Calculate the angle of the line that the nodes will be placed on
    if node_slope is not None:
        theta = math.atan(node_slope)
    else:
        # null means vertical
        theta = math.pi / 2

    # Calculate absolute positions
    for (x, y), nodes in nodes_by_bidegree.items():
        bidegree_rank = len(nodes)
        first_center_to_last_center = (bidegree_rank - 1) * distance_between_centers
        for i, node_id in enumerate(nodes):
            node = data["nodes"][node_id]
            # Check if absolute coordinates are already set (for TikZ compatibility)
            if "absoluteX" not in node or "absoluteY" not in node:
                offset = -first_center_to_last_center / 2 + i * distance_between_centers
                data["nodes"][node_id]["absoluteX"] = x + offset * math.cos(theta)
                data["nodes"][node_id]["absoluteY"] = y + offset * math.sin(theta)


def generate_nodes_svg(data, id_suffix=""):
    """Generate an SVG <g> element containing all nodes. `id_suffix`
    disambiguates the group id when two charts share one document
    (the side-by-side view), keeping ids unique per the HTML spec."""

    nodes_svg = f'<g id="nodes-group{id_suffix}">\n'

    for node_id, node in data.get("nodes", {}).items():
        cx = node["absoluteX"] * scale
        cy = node["absoluteY"] * scale

        attributes = node.get("attributes", [])
        style, aliases = style_and_aliases_from_attributes(attributes)
        
        # Extract shape information from attributes
        node_shape = "circle"  # default
        node_width = None
        node_height = None
        visible_text = ""
        
        for attr in attributes:
            if isinstance(attr, dict):
                if "shape" in attr:
                    node_shape = attr["shape"]
                if "width" in attr:
                    node_width = attr["width"] * scale
                if "height" in attr:
                    node_height = attr["height"] * scale
                if "visibleText" in attr:
                    visible_text = attr["visibleText"]

        style = style.generate(indent=0).replace("\n", " ").strip(" {}")
        if style:
            style = f'style="{style}"'
        aliases = " ".join(aliases)

        label = node.get("label", "")

        # Build data-map attribute if this node has map targets (E/H/P/C2/J).
        # Used by the side-by-side viewer to link source classes to their image.
        map_attr = ""
        map_value = node.get("map")
        if map_value:
            if isinstance(map_value, list):
                map_str = ";".join(map_value)
            else:
                map_str = str(map_value)
            map_attr = f' data-map="{map_str}"'

        # Generate appropriate SVG element based on shape
        if node_shape in ["square", "rectangle"]:
            # Use provided dimensions or defaults
            width = node_width or (scale * 0.09)  # default square size
            height = node_height or width

            # Position rectangle using the same Y coordinate reference as text (center-based)
            # Both rect and text will use y="{cy}" so they get identical coordinate transformations
            nodes_svg += f'<rect id="{node_id}" class="defaultNode {aliases}" x="{cx-width/2}" y="{cy-height/2}" width="{width}" height="{height}" {style} data-label="{label}"{map_attr}></rect>\n'

            # Add visible text if present
            if visible_text:
                nodes_svg += f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" fill="white" font-size="8" class="node-text">{visible_text}</text>\n'
        else:
            # Default circle behavior (backward compatible)
            nodes_svg += f'<circle id="{node_id}" class="defaultNode {aliases}" cx="{cx}" cy="{cy}" {style} data-label="{label}"{map_attr}></circle>\n'

    nodes_svg += "</g>\n"
    return nodes_svg


def generate_edges_svg(data, id_suffix=""):
    """Generate an SVG <g> element containing all edges. `id_suffix`: see
    generate_nodes_svg."""

    edges_svg = f'<g id="edges-group{id_suffix}">\n'

    for edge in data.get("edges", []):
        source = data["nodes"][edge["source"]]
        if "target" in edge:
            target = data["nodes"][edge["target"]]
            target_x = target["absoluteX"] * scale
            target_y = target["absoluteY"] * scale
        elif "offset" in edge:
            target_x = (source["absoluteX"] + edge["offset"]["x"]) * scale
            target_y = (source["absoluteY"] + edge["offset"]["y"]) * scale
        else:
            # Impossible due to schema
            raise NotImplementedError

        x1 = source["absoluteX"] * scale
        y1 = source["absoluteY"] * scale

        attributes = edge.get("attributes", [])
        style, aliases = style_and_aliases_from_attributes(attributes)
        style = style.generate(indent=0).replace("\n", " ").strip(" {}")
        aliases = " ".join(aliases)

        # Add data-source/data-target for JS-based edge highlighting
        source_id = edge["source"]
        target_id = edge.get("target", "")
        data_attrs = f' data-source="{source_id}"'
        if target_id:
            data_attrs += f' data-target="{target_id}"'

        if edge.get("bezier"):
            control_points = edge["bezier"]
            if len(control_points) == 1:
                control_x = control_points[0]["x"] * scale + x1
                control_y = control_points[0]["y"] * scale + y1
                curve_d = f"Q {control_x} {control_y} {target_x} {target_y}"
            elif len(control_points) == 2:
                control0_x = control_points[0]["x"] * scale + x1
                control0_y = control_points[0]["y"] * scale + y1
                control1_x = control_points[1]["x"] * scale + target_x
                control1_y = control_points[1]["y"] * scale + target_y
                curve_d = f"C {control0_x} {control0_y} {control1_x} {control1_y} {target_x} {target_y}"
            else:
                # Impossible due to schema
                raise NotImplementedError
            # For paths, we only want stroke styling, not fill
            # Remove all fill properties and keep only stroke properties
            path_style = re.sub(r'fill:\s*[^;]+;?\s*', '', style)
            if not path_style.strip():
                # If no stroke properties, convert the fill color to stroke
                fill_match = re.search(r'fill:\s*([^;]+)', style)
                if fill_match:
                    path_style = f"stroke: {fill_match.group(1)};"
            edge_svg = f'<path d="M {x1} {y1} {curve_d}" class="defaultEdge {aliases}" style="fill: none;{path_style}"{data_attrs}></path>\n'
        else:
            edge_svg = f'<line x1="{x1}" y1="{y1}" x2="{target_x}" y2="{target_y}" class="defaultEdge {aliases}" style="{style}"{data_attrs}></line>\n'

        # Remove empty style attributes; purely cosmetic cleanup of the output.
        edges_svg += edge_svg.replace(' style=""', "")

    edges_svg += "</g>\n"
    return edges_svg


def generate_svg(data, id_suffix=""):
    """Generate the chart body SVG: edges first, then nodes (drawn on top).
    `id_suffix` keeps group ids unique when two charts share a document."""
    calculate_absolute_positions(data)
    return generate_edges_svg(data, id_suffix) + generate_nodes_svg(data, id_suffix)


def generate_html(data, theme="light"):
    """Render a full single-chart HTML page from validated chart data."""
    # Generate CSS styles to be placed in <head>
    generate_css_styles(data, theme)
    # Calculate chart dimensions
    compute_chart_dimensions(data)
    # Generate SVG content
    static_svg_content = generate_svg(data)

    # Check if this is an Adams-Novikov chart to double the grid spacing and axis intervals
    is_adams_novikov = get_value_or_schema_default(data, ["header", "metadata", "adamsNovikov"])
    grid_spacing = scale * 2 if is_adams_novikov else scale
    axis_interval = 4 if is_adams_novikov else 2
    
    # Get theme colors for template
    theme_colors = get_theme_colors(theme)
    
    template = load_template()
    html_output = template.render(
        data=data,
        spacing=grid_spacing,
        axis_interval=axis_interval,
        css_styles=global_css.generate(),
        static_svg_content=static_svg_content,
        max_n=MAX_SPHERE_N,
        theme=theme if theme in THEME_PALETTES else "light",
        theme_colors=theme_colors,
        theme_css=build_theme_css(theme),
        theme_names_json=json.dumps(THEME_ORDER),
        theme_labels_json=json.dumps(THEME_LABELS),
        theme_label=THEME_LABELS.get(theme, theme),
    )
    return html_output


def load_sidebyside_template():
    """Load the two-panel (map source/target) page template."""
    return _load_template_file("template_sidebyside.html.jinja")


# Per-map key/navigation config for the side-by-side "map image" viewer.
# `key` opens/returns; `shift_key` toggles image mode; sibling navigation
# (w/s in the viewer) steps the source n by `nav_step`, bounded below by `min_n`.
SIDEBYSIDE_MAP_SPEC = {
    "E":  {"key": "e", "shift_key": "E", "nav_step": 1, "min_n": 2},
    "H":  {"key": "h", "shift_key": "H", "nav_step": 1, "min_n": 2},
    "P":  {"key": "p", "shift_key": "P", "nav_step": 2, "min_n": 5},
    "C2": {"key": "c", "shift_key": "C", "nav_step": 2, "min_n": 3},
    "J":  {"key": "j", "shift_key": "J", "nav_step": 1, "min_n": 2},
}
# Upper bound for w/s sibling navigation in the side-by-side viewer; matches
# the S2..S72 range of generated per-sphere charts.
MAX_SPHERE_N = 72


def _render_chart_panel(data, theme, id_suffix):
    """
    Run the standard single-chart pipeline (styles, dimensions, SVG) on `data`
    for one panel of the side-by-side view, resetting the global CssStyle/scale
    state first so the two panels don't bleed into each other.

    Returns (svg_content, css, spacing, axis_interval).
    """
    global global_css, scale
    global_css = CssStyle()
    scale = get_value_or_schema_default(data, ["header", "chart", "scale"])
    generate_css_styles(data, theme)
    compute_chart_dimensions(data)
    svg_content = generate_svg(data, id_suffix)
    is_adams_novikov = get_value_or_schema_default(
        data, ["header", "metadata", "adamsNovikov"]
    )
    spacing = scale * 2 if is_adams_novikov else scale
    axis_interval = 4 if is_adams_novikov else 2
    css = global_css.generate()
    return svg_content, css, spacing, axis_interval


def generate_sidebyside_html(source_json_file, target_json_file, output_file, theme="light",
                             back_url="", map_key="", source_title=None, target_title=None):
    """
    Generate a side-by-side HTML file: the map's SOURCE chart on the left and its
    TARGET chart on the right, wired for "map image" highlighting.

    Args:
        source_json_file: JSON for the left (source) chart; its nodes carry data-map.
        target_json_file: JSON for the right (target) chart.
        output_file: Path for the output HTML file.
        theme: Theme to use ("light", "dark", ...).
        back_url: URL for the back button / map-key navigation.
        map_key: which map this view represents (E/H/P/C2/J); drives keys + filenames.
        source_title / target_title: LaTeX panel titles; derived from filenames if omitted.
    """
    global global_css

    # Load source data (left panel)
    with open(source_json_file, "r") as f:
        source_data = json.load(f)
    jsonschema.validate(instance=source_data, schema=schema)

    # Load target data (right panel)
    with open(target_json_file, "r") as f:
        target_data = json.load(f)
    jsonschema.validate(instance=target_data, schema=schema)

    theme_colors = get_theme_colors(theme)

    # Process source chart (left), then target chart (right)
    source_svg_content, source_css, source_spacing, source_axis_interval = (
        _render_chart_panel(source_data, theme, "-left")
    )
    target_svg_content, target_css, target_spacing, target_axis_interval = (
        _render_chart_panel(target_data, theme, "-right")
    )

    # Fallback titles derived from the source filename (S{n}_E{r}...), only used
    # when the caller doesn't supply explicit LaTeX titles.
    if source_title is None or target_title is None:
        m = re.search(r'S(\d+)_E(\d+)', os.path.basename(source_json_file))
        chart_n = int(m.group(1)) if m else 0
        chart_r = int(m.group(2)) if m else 2
        if source_title is None:
            source_title = f"$\\mathrm{{E}}_{{{chart_r}}}(S^{{{chart_n}}})$"
        if target_title is None:
            target_title = f"$\\mathrm{{E}}_{{{chart_r}}}(?)$"

    # Unknown map keys fall back to the J spec (step 1, min n = 2).
    spec = SIDEBYSIDE_MAP_SPEC.get(map_key, SIDEBYSIDE_MAP_SPEC["J"])

    template = load_sidebyside_template()
    html_output = template.render(
        # Left panel = source, right panel = target. The template's so_*/sphere_*
        # variable slots feed the left/right panels respectively.
        so_data=source_data,
        sphere_data=target_data,
        so_svg_content=source_svg_content,
        sphere_svg_content=target_svg_content,
        so_spacing=source_spacing,
        sphere_spacing=target_spacing,
        so_axis_interval=source_axis_interval,
        sphere_axis_interval=target_axis_interval,
        so_css_styles=source_css,
        sphere_css_styles=target_css,
        so_title=source_title,
        sphere_title=target_title,
        theme=theme if theme in THEME_PALETTES else "light",
        theme_colors=theme_colors,
        theme_css=build_theme_css(theme),
        theme_names_json=json.dumps(THEME_ORDER),
        theme_labels_json=json.dumps(THEME_LABELS),
        theme_label=THEME_LABELS.get(theme, theme),
        back_url=back_url,
        map_key=map_key,
        map_open_key=spec["key"],
        shift_key=spec["shift_key"],
        nav_step=spec["nav_step"],
        min_n=spec["min_n"],
        max_n=MAX_SPHERE_N,
    )

    with open(output_file, "w") as f:
        f.write(html_output)
    print(f"Generated side-by-side {output_file} successfully.")

    # Reset
    global_css = CssStyle()


def generate_css_styles(data, theme="light"):
    """Populate the global_css variable with CSS classes for color and attribute aliases."""
    global global_css

    # Themed aliases point at the per-theme CSS custom properties emitted by
    # build_theme_css(), so the generated classes are theme-independent and
    # runtime theme switching needs no restyling. User-defined aliases from
    # the data header keep their literal values (they override).
    themed_colors = {
        alias: f"var(--cc-{alias})" for alias in get_themed_color_aliases(theme)
    }
    color_aliases = get_value_or_schema_default(data, ["header", "aliases", "colors"])
    final_color_aliases = {**themed_colors, **color_aliases}
    
    attribute_aliases = {
        "grid": get_schema_default(data, ["header", "aliases", "attributes", "grid"]),
        "defaultNode": get_schema_default(
            data, ["header", "aliases", "attributes", "defaultNode"]
        ),
        "defaultEdge": get_schema_default(
            data, ["header", "aliases", "attributes", "defaultEdge"]
        ),
    }
    user_attribute_aliases = get_value_or_schema_default(
        data, ["header", "aliases", "attributes"]
    )

    # Merge user-defined attribute aliases with the defaults
    for alias_name, attributes_list in user_attribute_aliases.items():
        current_attributes = attribute_aliases.get(alias_name, [])
        # This creates a new list instead of modifying the existing one, which would be bad. This is
        # because it could mutate a default value, which would ultimately corrupt `schema`.
        attribute_aliases[alias_name] = current_attributes + attributes_list

    # Generate CSS classes for color aliases. We do it first because we may need to reference them
    # in the attribute aliases.
    for color_name, color_value in final_color_aliases.items():
        css_styles = {"fill": color_value, "stroke": color_value}
        
        # Add dashing for n-type classes (nulldif edges)
        if color_name.startswith("n"):
            css_styles["stroke-dasharray"] = "5, 5"
            
        global_css += {
            cssify_name(color_name): css_styles
        }

    # Generate CSS class for nodes to set the appropriate size
    node_size = get_value_or_schema_default(data, ["header", "chart", "nodeSize"])
    global_css += {"circle": {"stroke-width": 0, "r": scale * node_size}}

    # Generate CSS classes for attribute aliases
    for alias_name, attributes_list in attribute_aliases.items():
        style, aliases = style_and_aliases_from_attributes(attributes_list)
        global_css += {cssify_name(alias_name): generate_style(style, aliases)}

    # Elements carry both an attribute alias and a color alias (e.g.
    # class="defaultEdge d3"). The color must win, but .defaultEdge/.defaultNode
    # are emitted later at equal specificity, so re-emit each color alias at
    # higher specificity. (This mis-cascade used to be masked by the runtime JS
    # restyle pass, which is gone now that theming is pure CSS.)
    for color_name in final_color_aliases:
        color_class = cssify_name(color_name)
        boosted = f".defaultNode{color_class}, .defaultEdge{color_class}"
        global_css += {boosted: dict(global_css[color_class])}

    # Stem-mode differential node states (see STEM_VIEW_SPEC.md): a class hit
    # by a d_r is a filled circle in the d_r color; a class supporting a d_r
    # is an "open" circle — filled with the background so it still catches
    # hover and occludes grid lines, with a thick d_r stroke (which must
    # override the default node `stroke-width: 0`). Defined after the
    # attribute aliases so they win against `defaultNode` at equal
    # specificity.
    for r_num in range(2, 9):
        c = f"var(--cc-d{r_num})"
        global_css += {
            cssify_name(f"diff_d{r_num}_filled"): {"fill": c, "stroke": c}
        }
        global_css += {
            cssify_name(f"diff_d{r_num}_open"): {
                "fill": "var(--bg-color)",
                "stroke": c,
                "stroke-width": scale * node_size * 0.4,
            }
        }

    # Add faded class for highlighting mode
    global_css += {
        ".faded": {"opacity": 0.3}
    }

    # Add dashed class for nulldif edges
    global_css += {
        ".dashed": {"stroke-dasharray": "5, 5"}
    }


def process_json(input_file, output_file, theme="light", view_mode="sphere", filter_value=None):
    """Validate a chart JSON file and render it to a standalone HTML page."""
    global global_css

    # Load input JSON
    with open(input_file, "r") as f:
        data = json.load(f)

    # validate against schema
    try:
        jsonschema.validate(instance=data, schema=schema)
    except ValidationError as e:
        print("Input JSON validation error:")
        print(e)
        sys.exit(1)

    global scale
    scale = get_value_or_schema_default(data, ["header", "chart", "scale"])

    # Add view mode and filter info to data for template access
    if "header" not in data:
        data["header"] = {}
    if "metadata" not in data["header"]:
        data["header"]["metadata"] = {}
    
    data["header"]["metadata"]["viewMode"] = view_mode
    data["header"]["metadata"]["filterValue"] = filter_value

    # Generate HTML
    html_content = generate_html(data, theme)

    # Write to output file
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"Generated {output_file} successfully.")

    # Reset global_css for the next file
    global_css = CssStyle()


def load_multiple_charts(chart_files, theme="light"):
    """
    Load and process multiple JSON chart files for multi-chart HTML generation.
    
    Args:
        chart_files: List of JSON file paths
        theme: Theme to use for processing ("light" or "dark")
    
    Returns:
        Dict mapping chart_id -> processed chart data with SVG content
    """
    charts = {}
    
    for file_path in chart_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Validate against schema
        try:
            jsonschema.validate(instance=data, schema=schema)
        except ValidationError as e:
            print(f"Validation error in {file_path}: {e}")
            continue
            
        # Extract chart identifier from filename (e.g., S3_E2.json -> S3_E2)
        chart_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Process the chart data
        global scale
        scale = get_value_or_schema_default(data, ["header", "chart", "scale"])
        
        # Add view mode and filter info to data for template access
        if "header" not in data:
            data["header"] = {}
        if "metadata" not in data["header"]:
            data["header"]["metadata"] = {}
        
        # Generate CSS styles and SVG content
        generate_css_styles(data, theme)
        compute_chart_dimensions(data)
        svg_content = generate_svg(data)
        
        # Calculate bounds for layout
        nodes = data.get("nodes", {})
        if nodes:
            x_coords = [node["x"] for node in nodes.values()]
            y_coords = [node["y"] for node in nodes.values()]
            bounds = {
                'minX': min(x_coords) if x_coords else 0,
                'maxX': max(x_coords) if x_coords else 10,
                'minY': min(y_coords) if y_coords else 0,
                'maxY': max(y_coords) if y_coords else 10
            }
        else:
            bounds = {'minX': 0, 'maxX': 10, 'minY': 0, 'maxY': 10}
        
        # Check if this is an Adams-Novikov chart
        is_adams_novikov = get_value_or_schema_default(data, ["header", "metadata", "adamsNovikov"])
        grid_spacing = scale * 2 if is_adams_novikov else scale
        axis_interval = 4 if is_adams_novikov else 2
        
        # Store processed chart data
        charts[chart_id] = {
            'data': data,
            'svg': svg_content,
            'spacing': grid_spacing,
            'axis_interval': axis_interval,
            'bounds': bounds,
            'is_adams_novikov': is_adams_novikov
        }
        
    return charts


def create_multi_chart_template():
    """
    Create enhanced template for multi-chart navigation.
    Based on template.html.jinja but with embedded chart switching.
    """
    template_content = '''<!-- Multi-chart template for SeqSee -->
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <title>{{ title or "SeqSee Multi-Chart Navigator" }}</title>
  
  <!-- Immediate theme colors to prevent any flashing -->
  <style>
    body { 
      background-color: {{ theme_colors.base }} !important; 
      color: {{ theme_colors.text }} !important; 
    }
    svg, svg * { 
      background-color: {{ theme_colors.base }} !important; 
    }
    .defaultNode { 
      fill: {{ theme_colors.text }} !important; 
      stroke: {{ theme_colors.text }} !important; 
    }
    .defaultEdge { 
      stroke: {{ theme_colors.text }} !important; 
    }
    .grid-line { 
      stroke: {{ theme_colors.surface1 }} !important; 
    }
  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.2/dist/katex.min.css" crossorigin="anonymous" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/fonts.css" />
  <!-- KaTeX for math rendering -->
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.2/dist/katex.min.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.2/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <!-- svg-pan-zoom for obvious reasons -->
  <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
  <!-- Hammer.js for touch controls -->
  <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
  <!-- Path data polyfill for SVG path manipulation -->
  <script src="https://cdn.jsdelivr.net/npm/path-data-polyfill@1.0.6/path-data-polyfill.min.js"></script>
  <style>
    :root {
      --bg-color: {{ theme_colors.base }};
      --text-color: {{ theme_colors.text }};
      --surface-color: {{ theme_colors.surface0 }};
      --grid-color: {{ theme_colors.surface1 }};
      --button-bg: {{ theme_colors.surface1 }};
      --button-hover: {{ theme_colors.surface2 }};
      --button-text: {{ theme_colors.text }};
    }

    body {
      font-family: "Computer Modern Serif", serif;
      font-size: 20pt;
      margin: 0;
      overflow: hidden;
      background-color: var(--bg-color);
      color: var(--text-color);
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* KaTeX font size fix */
    .katex {
      font-size: 1em !important;
    }

    #canvas-container {
      width: 100vw;
      height: 100vh;
      position: absolute;
    }

/* Dynamically generated CSS styles */
{{ css_styles }}

    #tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      background-color: var(--surface-color);
      border: 1px solid var(--text-color);
      color: var(--text-color);
      padding: 5px;
      z-index: 10;
      border-radius: 4px;
    }

    #controls-container {
      position: absolute;
      top: 20px;
      right: 20px;
      z-index: 10;
      display: flex;
      gap: 10px;
      pointer-events: auto;
    }

    .control-button {
      background-color: var(--button-bg);
      color: var(--button-text);
      border: 1px solid var(--text-color);
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      transition: background-color 0.3s ease, border-color 0.3s ease;
      user-select: none;
    }

    .control-button:hover {
      background-color: var(--button-hover);
    }

    .control-button.active {
      background-color: var(--text-color);
      color: var(--bg-color);
    }

    #title-container {
      position: absolute;
      top: 20px;
      left: 20px;
      right: 20px;
      z-index: 5;
      pointer-events: none;
      text-align: center;
    }

    #main-title {
      font-size: 24px;
      font-weight: bold;
      color: var(--text-color);
      background-color: var(--bg-color);
      border: 1px solid var(--surface-color);
      padding: 15px 25px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
      display: inline-block;
      max-width: 80%;
      line-height: 1.3;
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }

    #status-display {
      position: absolute;
      bottom: 20px;
      left: 20px;
      z-index: 10;
      font-size: 12px;
      color: var(--text-color);
      opacity: 0.7;
      background-color: var(--bg-color);
      padding: 5px 10px;
      border-radius: 4px;
      border: 1px solid var(--surface-color);
    }

    .axis {
      stroke: var(--text-color);
      stroke-width: 2px;
    }

    .tick {
      font-size: 12pt;
      fill: var(--text-color);
    }

    .x-tick {
      text-anchor: middle;
      dominant-baseline: hanging;
    }

    .y-tick {
      text-anchor: end;
      dominant-baseline: middle;
    }
  </style>
</head>

<!-- We set `visibility: visible` in JS after some preprocessing -->
<body style="visibility: hidden; background-color: {{ theme_colors.base }}; color: {{ theme_colors.text }}">
  <div id="controls-container">
    <button class="control-button" id="theme-toggle" onclick="toggleTheme()">
      {{ "Dark" if theme == "light" else "Light" }}
    </button>
    <button class="control-button" id="chart-info" onclick="showChartInfo()">
      Charts: {{ chart_count }}
    </button>
  </div>
  <div id="title-container">
    <div id="main-title"></div>
  </div>
  <div id="status-display">
    Use WASD to navigate charts • {{ available_charts }} • No page reloads!
  </div>
  <div id="canvas-container">
    <svg id="svg-canvas" width="100%" height="100%">
      <defs>
        <!-- Define the arrowhead markers -->
        <marker id='arrow-simple' orient="auto" markerWidth='3' markerHeight='4' refX='0.1' refY='2' fill="context-fill"
          stroke="context-stroke">
          <path d='M0,0 V4 L2,2 Z' />
        </marker>
        <!-- Define the grid pattern -->
        <pattern id="grid" width="120" height="120" patternUnits="userSpaceOnUse">
          <path id="grid-path" d="M 120 0 L 0 0 0 120" class="grid" style="fill: none;"/>
        </pattern>
      </defs>
      <g id="content-group" class="svg-pan-zoom_viewport">
        <!-- Translate the grid so that it covers the appropriate region outside the first quadrant -->
        <g id="origin-translate">
          <!-- Apply the grid pattern to a background rectangle -->
          <rect
            id="grid-background"
            width="2000px"
            height="2000px"
            fill="url(#grid)"
          />
        </g>
        <!-- Nodes and Edges will be dynamically loaded here -->
        <g id="dynamic-content"></g>
      </g>
      <g id="axes-group">
        <!-- X-axis -->
        <line id="x-axis" class="axis" />
        <!-- Y-axis -->
        <line id="y-axis" class="axis" />
        <!-- Blocks under and to the left to hide the content -->
        <rect id="x-block" x="0" y="0" fill="var(--bg-color)" />
        <rect id="y-block" x="0" y="0" fill="var(--bg-color)" />
        <!-- Tick marks -->
        <g id="ticks" class="tick">
          <g id="x-ticks" class="x-tick"></g>
          <g id="y-ticks" class="y-tick"></g>
        </g>
      </g>
    </svg>
  </div>
  <div id="tooltip"></div>
  <script>
    // Embedded chart datasets
    const CHART_DATA = {{ chart_data_json | safe }};
    const CHART_IDS = {{ chart_ids_json | safe }};
    
    // Theme data for toggling
    const themes = {
      light: {
        base: "#eff1f5", text: "#4c4f69", surface0: "#ccd0da", surface1: "#bcc0cc", surface2: "#acb0be",
        sapphire: "#209fb5", teal: "#179299", green: "#40a02b", yellow: "#df8e1d", red: "#d20f39", 
        blue: "#1e66f5", mauve: "#8839ef", pink: "#ea76cb", peach: "#fe640b"
      },
      dark: {
        base: "#1e1e2e", text: "#cdd6f4", surface0: "#313244", surface1: "#45475a", surface2: "#585b70",
        sapphire: "#74c7ec", teal: "#94e2d5", green: "#a6e3a1", yellow: "#f9e2af", red: "#f38ba8", 
        blue: "#89b4fa", mauve: "#cba6f7", pink: "#f5c2e7", peach: "#fab387"
      }
    };
    
    let currentTheme = sessionStorage.getItem('seqsee-theme') || "{{ theme }}";
    let currentChartId = sessionStorage.getItem('seqsee-current-chart') || "{{ default_chart }}";
    
    function parseChartId(chartId) {
      // Parse chart ID like "S3_E2" -> {n: 3, r: 2}
      const match = chartId.match(/S(\\d+)_E(\\d+)/);
      if (match) {
        return { n: parseInt(match[1]), r: parseInt(match[2]) };
      }
      return { n: 3, r: 2 }; // Default
    }
    
    function buildChartId(n, r) {
      return `S${n}_E${r}`;
    }
    
    function updateTitle() {
      const { n, r } = parseChartId(currentChartId);
      const titleText = `$\\mathrm{E}_{${r}}(S^{${n}})$`;
      document.getElementById('main-title').textContent = titleText;
      
      // Render LaTeX in the main title
      window.renderMathInElement(document.getElementById('main-title'), {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\\\(', right: '\\\\)', display: false},
          {left: '\\\\[', right: '\\\\]', display: true}
        ],
        throwOnError: false
      });
    }
    
    function loadChart(chartId) {
      if (!CHART_DATA[chartId]) {
        console.error(`Chart ${chartId} not found`);
        return false;
      }
      
      console.log(`Loading chart ${chartId}`);
      
      const chartData = CHART_DATA[chartId];
      
      // Clear existing content
      const dynamicContent = document.getElementById('dynamic-content');
      dynamicContent.innerHTML = '';
      
      // Insert new chart SVG content
      dynamicContent.innerHTML = chartData.svg;
      
      // Update grid and layout
      updateChartLayout(chartData);
      
      // Update title
      updateTitle();
      
      // Apply theme colors to new content
      updateThemeColors();
      
      // Store current chart
      sessionStorage.setItem('seqsee-current-chart', chartId);
      currentChartId = chartId;
      
      // Restore viewport if available
      setTimeout(() => {
        restoreViewport();
      }, 100);
      
      console.log(`Chart ${chartId} loaded successfully`);
      return true;
    }
    
    function updateChartLayout(chartData) {
      const spacing = chartData.spacing || 120;
      const bounds = chartData.bounds || { minX: 0, maxX: 10, minY: 0, maxY: 10 };
      
      // Update grid pattern
      const gridPattern = document.getElementById('grid');
      gridPattern.setAttribute('width', spacing * 2);
      gridPattern.setAttribute('height', spacing * 2);
      
      const gridPath = document.getElementById('grid-path');
      gridPath.setAttribute('d', `M ${spacing * 2} 0 L 0 0 0 ${spacing * 2}`);
      
      // Update origin translate for grid
      const originTranslate = document.getElementById('origin-translate');
      originTranslate.setAttribute('transform', `translate(${bounds.minX * spacing} ${-bounds.minY * spacing})`);
      
      // Update grid background size
      const gridBg = document.getElementById('grid-background');
      gridBg.setAttribute('width', `${(bounds.maxX - bounds.minX) * spacing}px`);
      gridBg.setAttribute('height', `${(bounds.maxY - bounds.minY) * spacing}px`);
      
      // Update ticks
      updateTicks(bounds, spacing, chartData.axis_interval || 2);
    }
    
    function updateTicks(bounds, spacing, axisInterval) {
      const xTicks = document.getElementById('x-ticks');
      const yTicks = document.getElementById('y-ticks');
      
      // Clear existing ticks
      xTicks.innerHTML = '';
      yTicks.innerHTML = '';
      
      // Generate x-ticks
      for (let i = bounds.minX; i <= bounds.maxX; i += axisInterval) {
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        tick.setAttribute('x', i * spacing);
        tick.setAttribute('y', 0.5 * spacing);
        tick.textContent = i;
        xTicks.appendChild(tick);
      }
      
      // Generate y-ticks
      for (let j = bounds.minY; j <= bounds.maxY; j += axisInterval) {
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        tick.setAttribute('x', 0.5 * spacing);
        tick.setAttribute('y', j * spacing);
        tick.textContent = j;
        yTicks.appendChild(tick);
      }
    }
    
    function navigateToSphere(delta) {
      const { n, r } = parseChartId(currentChartId);
      const newN = Math.max(2, Math.min(72, n + delta));
      const newChartId = buildChartId(newN, r);
      
      if (newChartId !== currentChartId && CHART_DATA[newChartId]) {
        storeViewport(); // Store current viewport before navigating
        loadChart(newChartId);
      }
    }
    
    function navigateToPage(delta) {
      const { n, r } = parseChartId(currentChartId);
      const newR = Math.max(2, Math.min(10, r + delta));
      const newChartId = buildChartId(n, newR);
      
      if (newChartId !== currentChartId && CHART_DATA[newChartId]) {
        storeViewport(); // Store current viewport before navigating
        loadChart(newChartId);
      }
    }
    
    function showChartInfo() {
      const info = `Available charts: ${CHART_IDS.length}\\nCurrent: ${currentChartId}\\nUse WASD to navigate`;
      alert(info);
    }
    
    // Include all the theme, viewport, and utility functions from template.html.jinja
    {{ theme_and_navigation_functions }}
    
    // Keyboard navigation
    window.addEventListener("keydown", function (event) {
      const panSpeed = 120; // Use standard spacing for pan speed
      switch (event.key) {
        // Add arrow key controls for panning
        case "ArrowUp":
          event.preventDefault();
          window.panZoom.panBy({ x: 0, y: panSpeed });
          break;
        case "ArrowDown":
          event.preventDefault();
          window.panZoom.panBy({ x: 0, y: -panSpeed });
          break;
        case "ArrowLeft":
          event.preventDefault();
          window.panZoom.panBy({ x: panSpeed, y: 0 });
          break;
        case "ArrowRight":
          event.preventDefault();
          window.panZoom.panBy({ x: -panSpeed, y: 0 });
          break;
        // Add + and - key controls for zooming
        case "+":
        case "=":
          event.preventDefault();
          window.panZoom.zoomIn();
          break;
        case "-":
        case "_":
          event.preventDefault();
          window.panZoom.zoomOut();
          break;
        // Add controls for resetting pan and zoom
        case "Backspace":
        case "0":
        case ")":
          event.preventDefault();
          window.panZoom.reset();
          window.panZoom.panBy({ x: 2 * 60, y: -2 * 60 });
          break;
        
        // Enhanced navigation controls
        case "s":
        case "S":
          event.preventDefault();
          navigateToSphere(1); // n -> n+1
          break;
        case "w":
        case "W":
          event.preventDefault();
          navigateToSphere(-1); // n -> n-1
          break;
        case "d":
        case "D":
          event.preventDefault();
          navigateToPage(1); // r -> r+1
          break;
        case "a":
        case "A":
          event.preventDefault();
          navigateToPage(-1); // r -> r-1
          break;
      }
    });
    
    // Initialize when DOM is loaded
    window.addEventListener('DOMContentLoaded', () => {
      console.log('SeqSee Multi-Chart Navigator initializing...');
      console.log(`Available charts: ${CHART_IDS.length}`);
      console.log(`Current chart: ${currentChartId}`);
      
      // Load initial chart
      if (!loadChart(currentChartId)) {
        // Fallback to first available chart
        const firstChart = CHART_IDS[0];
        if (firstChart) {
          console.log(`Falling back to ${firstChart}`);
          loadChart(firstChart);
        }
      }
      
      // Show the page after setup
      document.body.style.visibility = "visible";
      
      console.log('SeqSee Multi-Chart Navigator ready!');
    });
  </script>
</body>
</html>'''
    
    return template_content


def generate_multi_chart_html(chart_files, output_file, theme="light", title=None):
    """
    Generate a single HTML file with multiple embedded charts using SeqSee machinery.
    
    Args:
        chart_files: List of JSON file paths to embed
        output_file: Output HTML file path
        theme: Theme to use ("light" or "dark")
        title: Custom title for the navigator
    
    Returns:
        True if successful, False otherwise
    """
    print(f"Generating multi-chart HTML with {len(chart_files)} charts...")
    
    # Load and process all charts
    charts = load_multiple_charts(chart_files, theme)
    if not charts:
        print("No valid charts found!")
        return False
    
    print(f"Processed {len(charts)} charts: {list(charts.keys())}")
    
    # Get theme colors
    theme_colors = get_theme_colors(theme)

    # Unified CSS: load_multiple_charts never resets global_css, so this is the
    # accumulated union of the styles from every embedded chart.
    css_styles = global_css.generate()
    
    # Prepare chart data for JavaScript embedding
    chart_data = {}
    for chart_id, chart_info in charts.items():
        chart_data[chart_id] = {
            'svg': chart_info['svg'],
            'spacing': chart_info['spacing'],
            'axis_interval': chart_info['axis_interval'],
            'bounds': chart_info['bounds'],
            'is_adams_novikov': chart_info['is_adams_novikov']
        }
    
    chart_ids = sorted(charts.keys())
    default_chart = chart_ids[0] if chart_ids else "S3_E2"
    
    # Extract theme and navigation functions from existing template
    theme_functions = """
    function toggleTheme() {
      currentTheme = currentTheme === "light" ? "dark" : "light";
      sessionStorage.setItem('seqsee-theme', currentTheme);
      updateThemeColors();
      const button = document.getElementById("theme-toggle");
      button.textContent = currentTheme === "light" ? "Dark" : "Light";
    }

    function updateThemeColors() {
      const colors = themes[currentTheme];
      const root = document.documentElement;
      
      // Update CSS variables
      root.style.setProperty('--bg-color', colors.base);
      root.style.setProperty('--text-color', colors.text);
      root.style.setProperty('--surface-color', colors.surface0);
      root.style.setProperty('--grid-color', colors.surface1);
      root.style.setProperty('--button-bg', colors.surface1);
      root.style.setProperty('--button-hover', colors.surface2);
      root.style.setProperty('--button-text', colors.text);
      
      // Update SVG elements
      document.querySelectorAll('.axis').forEach(el => {
        el.style.stroke = colors.text;
      });
      
      document.querySelectorAll('.tick text').forEach(el => {
        el.style.fill = colors.text;
      });
      
      document.querySelectorAll('#x-block, #y-block').forEach(el => {
        el.setAttribute('fill', colors.base);
      });
      
      // Update nodes and edges
      document.querySelectorAll('.defaultNode, .gray').forEach(el => {
        if (!hasSpecificColorClass(el)) {
          el.style.setProperty('fill', colors.text, 'important');
          el.style.setProperty('stroke', colors.text, 'important');
        }
      });
      
      document.querySelectorAll('.defaultEdge').forEach(el => {
        if (!hasSpecificColorClass(el)) {
          el.style.setProperty('stroke', colors.text, 'important');
        }
      });
      
      // Update differential colors
      const colorMappings = {
        'd2': colors.teal, 'd3': colors.red, 'd4': colors.green, 'd5': colors.blue,
        'd6': colors.yellow, 'd7': colors.peach, 'd8': colors.mauve,
        'n2': colors.teal, 'n3': colors.red, 'n4': colors.green, 'n5': colors.blue,
        'n6': colors.yellow, 'n7': colors.peach, 'n8': colors.mauve
      };
      
      Object.entries(colorMappings).forEach(([className, color]) => {
        document.querySelectorAll(`.${className}`).forEach(el => {
          el.style.setProperty('fill', color, 'important');
          el.style.setProperty('stroke', color, 'important');
        });
      });
    }
    
    function hasSpecificColorClass(element) {
      return Array.from(element.classList).some(cls => 
        cls.match(/^[dn]\\d+$/) || ['dr', 'nulldif'].includes(cls)
      );
    }
    
    function getCurrentViewport() {
      if (!window.panZoom) return null;
      const pan = window.panZoom.getPan();
      const zoom = window.panZoom.getZoom();
      return { x: pan.x, y: pan.y, zoom: zoom };
    }
    
    function storeViewport() {
      const viewport = getCurrentViewport();
      if (viewport) {
        sessionStorage.setItem('seqsee_viewport', JSON.stringify(viewport));
      }
    }
    
    function restoreViewport() {
      const stored = sessionStorage.getItem('seqsee_viewport');
      if (stored && window.panZoom) {
        try {
          const viewport = JSON.parse(stored);
          setTimeout(() => {
            if (window.panZoom) {
              window.panZoom.zoom(viewport.zoom);
              window.panZoom.pan({ x: viewport.x, y: viewport.y });
            }
          }, 100);
        } catch (e) {
          console.log("Failed to restore viewport:", e);
        }
      }
    }
    
    // Initialize svg-pan-zoom
    window.panZoom = svgPanZoom("#svg-canvas", {
      zoomEnabled: true,
      panEnabled: true,
      fit: false,
      center: false,
      minZoom: 0.1,
      maxZoom: 10,
      zoomScaleSensitivity: 0.15
    });
    """
    
    # Create template and render
    template_content = create_multi_chart_template()
    env = Environment()
    template = env.from_string(template_content)
    
    # Summary of available charts for status display
    available_charts = f"E{min(int(cid.split('_E')[1]) for cid in chart_ids)}-E{max(int(cid.split('_E')[1]) for cid in chart_ids)}"
    
    html_output = template.render(
        title=title,
        theme=theme,
        theme_colors=theme_colors,
        css_styles=css_styles,
        chart_data_json=json.dumps(chart_data),
        chart_ids_json=json.dumps(chart_ids),
        default_chart=default_chart,
        chart_count=len(charts),
        available_charts=available_charts,
        theme_and_navigation_functions=theme_functions
    )
    
    # Write output
    with open(output_file, 'w') as f:
        f.write(html_output)
    
    print(f"Generated {output_file} with {len(charts)} embedded charts!")
    print(f"Charts: {chart_ids}")
    print(f"Theme: {theme}")
    print(f"Ready to open - no page reloads, no color flashing!")
    
    return True


def main():
    # Pull the optional `--max-stem N` flag out first so it composes with the
    # positional call style. It clamps the chart's x-axis (stem) at N; 0 or
    # negative disables. Only the per-sphere charts pass it.
    global MAX_STEM
    if "--max-stem" in sys.argv:
        i = sys.argv.index("--max-stem")
        val = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        try:
            MAX_STEM = int(val)
        except ValueError:
            MAX_STEM = None
        del sys.argv[i:i + 2]
        if MAX_STEM is not None and MAX_STEM <= 0:
            MAX_STEM = None

    if len(sys.argv) >= 2 and sys.argv[1] == "--sidebyside":
        # Side-by-side mode:
        #   --sidebyside <source.json> <target.json> <output.html>
        #                [theme] [back_url] [map_key] [source_title] [target_title]
        if len(sys.argv) < 5:
            print("Usage: seqsee --sidebyside <source.json> <target.json> <output.html> "
                  "[theme] [back_url] [map_key] [source_title] [target_title]")
            sys.exit(1)
        source_json = sys.argv[2]
        target_json = sys.argv[3]
        output_file = sys.argv[4]
        theme = sys.argv[5] if len(sys.argv) > 5 else "light"
        back_url = sys.argv[6] if len(sys.argv) > 6 else ""
        map_key = sys.argv[7] if len(sys.argv) > 7 else ""
        source_title = sys.argv[8] if len(sys.argv) > 8 else None
        target_title = sys.argv[9] if len(sys.argv) > 9 else None
        generate_sidebyside_html(source_json, target_json, output_file, theme, back_url,
                                 map_key, source_title, target_title)
        sys.exit(0)

    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: seqsee <input.json> <output.html> [theme] [view_mode] [filter_value]")
        print("       seqsee --multi <output.html> <chart1.json> [chart2.json] ... [theme]")
        print("       seqsee --sidebyside <so.json> <sphere.json> <output.html> [theme] [back_url]")
        print("  theme: 'light' or 'dark' (default: light)")
        print("  view_mode: 'sphere' or 'stem' (default: sphere)")
        print("  filter_value: integer to filter by (n for sphere, s for stem)")
        sys.exit(1)

    if sys.argv[1] == "--multi":
        # Multi-chart mode
        if len(sys.argv) < 4:
            print("Usage: seqsee --multi <output.html> <chart1.json> [chart2.json] ... [theme]")
            sys.exit(1)
        
        output_file = sys.argv[2]
        chart_files = []
        theme = "light"
        
        # Parse arguments - everything except last arg (if it's a theme) is a chart file
        for i in range(3, len(sys.argv)):
            arg = sys.argv[i]
            if arg in THEME_PALETTES and i == len(sys.argv) - 1:
                theme = arg
            else:
                chart_files.append(arg)
        
        if not chart_files:
            print("No chart files specified!")
            sys.exit(1)
        
        success = generate_multi_chart_html(chart_files, output_file, theme)
        sys.exit(0 if success else 1)
    else:
        # Single chart mode (existing functionality)
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        theme = sys.argv[3] if len(sys.argv) > 3 else "light"
        view_mode = sys.argv[4] if len(sys.argv) > 4 else "sphere"
        filter_value = int(sys.argv[5]) if len(sys.argv) > 5 else None

        process_json(input_file, output_file, theme, view_mode, filter_value)


if __name__ == "__main__":
    main()
