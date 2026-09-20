"""
Spectral sequence computation engine for Adams spectral sequences.

This module provides the core classes for computing Adams spectral sequences:
- SpectralSequencePage: Represents a single E_r page with differentials and maps
- SpectralSequence: Container managing multiple pages (E_2, E_3, E_4, ...)

Key operations:
- compute(): Apply naturality constraints (E, H, P maps) to determine differentials
- next_page(): Compute E_{r+1} from E_r via homology H(E_r, d_r)
- save_to_directory(): Export results to CSV files

Typical workflow:
    >>> ss = SpectralSequence.from_data("E2", r=2, tot=79)
    >>> ss.current_page.compute()
    >>> ss.next_page(save=True)

The computation uses:
- Naturality of suspension (E), Hopf (H), and power (P) maps
- Leibniz rule for products
- d² = 0 constraint
- Stability for high connectivity elements

Uncertainty tracking allows for partial computations when differentials
cannot be fully determined from available constraints.

Conventions at this module's boundaries:
- Tridegree (n, s, f): n indexes the sphere column (n = 0 is the Lambda(C2)
  column; n = 1 is not charted), s is the stem, f is the Adams filtration
  (see the write_spheres column headers).
- On page r the differential is d_r: (n, s, f) -> (n, s - 1, f + r); see
  d_target/d_source.
- In-memory matrices act on ROW vectors by right multiplication: rows are
  indexed by the SOURCE basis and each row is the image vector in the TARGET
  basis (so `x.vect * M` is the image of x). This holds for map matrices,
  the multiplication matrices L/R/M/EM, and differentials via as_matrix.

Failure policy (see _apply_constraint): a ContradictionError -- an empty
constraint intersection, meaning the input data or an earlier deduction is
wrong -- is always fatal and reported with its full deduction trace. Any
other failure in a constraint application (typically a matrix shape mismatch
at a data boundary) is also fatal in generic constraint code; only sites
where a data-coverage edge is expected (the C2 reverse sweep) pass
soft_structural=True to warn and skip instead.
"""
import ast
import copy as cpy
import csv
import itertools
import json
import os
import re
import sys
from collections import defaultdict
import sage.all
from sage.all import GF, Hom, Sequence, VectorSpace, matrix, span, vector
from sage.geometry.hyperplane_arrangement.affine_subspace import AffineSubspace

from lib import *
from uncertainty import UncertaintyManager
from differentials import DifferentialsPage
import spurious  # the spurious-uncertainty resolver (see spurious.py)

# Computation bounds and thresholds

# C2-naturality deductions are only applied when both the source tridegree and
# the differential target stay within this total degree (s+f). This keeps C2
# deductions safely inside the range where the n=0 Lambda(C2) column and the
# stable input files are complete; past it they would rest on partial data.
MAX_TOTAL_DEGREE_THRESHOLD = C2_DATA_COMPLETE_TOT

# Sphere charts omit rows above this n. It sits far above any n the loader
# admits (n <= 2*tot), so it only guards against pathological entries.
MAX_N_FOR_CHART_OUTPUT = 400

DEFAULT_CHARTS_DIR = "charts"  # Default directory for chart output

class SpectralSequencePage:
    """Represents a single E_r page of a spectral sequence"""
    def __init__(self, max_n=None, max_s=None, max_f=None, max_t=None, r=2):
        self.max_n = max_n
        self.max_s = max_s
        self.max_f = max_f
        self.max_t = max_t

        self.dimension = defaultdict(int)
        self.page = {}
        self.products = {}
        self.maps = {}
        self.d = DifferentialsPage(r=r, dimension_dict=self.dimension)
        self.names = {}  # Dictionary mapping element strings to their names

        self.uncertainty_manager = UncertaintyManager()
        self.pairs = {}
        self.turned_page = None
        # How many dimension-reducing deductions each map contributed during
        # compute(); reported in the end-of-compute summary so it is visible
        # whether e.g. the h0-h3 filtration-1 Leibniz rule ever fired.
        self.map_deduction_counts = defaultdict(int)
        # Same tally for non-map constraint rules (d^2).
        self.rule_deduction_counts = defaultdict(int)

    def compute_max_values(self):
        """Infer max_n/max_s/max_f/max_t from the dimension table (with a safety margin of 2), filling in only the values not already set"""
        if not self.dimension:
            return
        max_n = 0
        max_s = 0
        max_t = 0

        for (n, s, f), dim in self.dimension.items():
            if dim > 0:
                max_n = max(max_n, n)
                max_s = max(max_s, s)
                max_t = max(max_t, s + f)
        if self.max_n is None:
            self.max_n = max_n - 2
        if self.max_s is None:
            self.max_s = max_s - 2
        if self.max_t is None:
            self.max_t = max_t - 2
        # max_f is the highest Adams filtration inside the total-degree window
        # s + f <= max_t, with no safety margin: the f <= max_f bound in
        # is_computable must not exclude a class the window itself contains.
        if self.max_f is None:
            self.max_f = max((f for (n, s, f), dim in self.dimension.items()
                              if dim > 0 and s + f <= self.max_t), default=0)

        print(f"Computed max values: max_n={self.max_n}, max_s={self.max_s}, max_f={self.max_f}, max_t={self.max_t}")

    @property
    def uncertain(self):
        return self.uncertainty_manager.uncertain

    def zero(self, n, s, f):
        """Return zero element with specified tridegree (n, s, f)"""
        dim = self.dimension[n, s, f]
        if dim == 0:
            if (n, s, f) in self.page and self.page[n, s, f]:
                return Element(n, s, f, self.page[n, s, f][0].vect.parent().zero(), spectral_sequence=self)
            else:
                return Element(n, s, f, vector(GF(2), []), spectral_sequence=self)
        else:
            return Element(n, s, f, vector(GF(2), [0] * dim), spectral_sequence=self)

    def d_target(self, n, s, f):
        """
        Return the target tridegree of the differential from (n, s, f).

        The differential d_r: E_r^{n,s,f} -> E_r^{n,s-1,f+r}
        """
        r = self.d.r
        return (n, s - 1, f + r)

    def d_source(self, n, s, f):
        """
        Return the source tridegree of the differential to (n, s, f).

        The differential d_r: E_r^{n,s+1,f-r} -> E_r^{n,s,f}
        """
        r = self.d.r
        return (n, s + 1, f - r)

    def has_elements(self, n, s, f):
        """
        Check if tridegree (n, s, f) has any elements.

        Relies on invariant: (n,s,f) in self.page.keys() ⟺ self.dimension[n,s,f] > 0

        This invariant is maintained by:
        - to_page(): only adds non-empty bases
        - compute_dimensions(): computes len(self.page[n,s,f])
        """
        return (n, s, f) in self.page

    def map_domain(self, map_name):
        """
        Get the domain of a map: all tridegrees (n,s,f) with elements.

        Returns list of (n,s,f) tuples, ordered lexicographically.
        """
        if map_name not in self.maps:
            return []

        map_obj = self.maps[map_name]
        domain = []

        for (n, s, f) in self.page.keys():
            if not map_obj.domain_check(n, s, f):
                continue
            domain.append((n, s, f))

        # Sort lexicographically by (n, s, f)
        domain.sort()
        return domain

    def map_codomain(self, map_name):
        """
        Get the codomain of a map: all tridegrees (n,s,f) with elements.

        Returns list of (n,s,f) tuples, ordered by decreasing n, then increasing s, then increasing f.
        """
        if map_name not in self.maps:
            return []

        map_obj = self.maps[map_name]
        if map_obj.source_transforms is None:
            # Non-invertible degree shift (C2): no codomain indexing; reverse
            # naturality runs source-indexed via _natural_rev_source_sweep.
            return []
        codomain = []

        for (n, s, f) in self.page.keys():
            source = map_obj.source_degree(n, s, f)

            # Verify that the computed source actually maps to this target
            # (source_degree is only an approximation for some maps)
            computed_target = map_obj.target_degree(*source)
            if computed_target != (n, s, f):
                continue

            # The degree round trip is blind to domain restrictions that the
            # transforms don't encode: the hi maps leave n unchanged, so any
            # sphere tridegree "inverts" successfully even though hi only
            # exists on the n=0 column. Enforce the map's actual domain.
            if not map_obj.domain_check(*source):
                continue
            codomain.append((n, s, f))

        # Sort: reverse in n, normal in s and f
        codomain.sort(key=lambda t: (-t[0], t[1], t[2]))
        return codomain

    def all_tridegrees_with_elements(self):
        """
        Get all tridegrees that have elements.
        Returns list of (n,s,f) tuples sorted by total degree.
        """
        return sorted(self.page.keys(), key=lambda t: (t[1] + t[2], t[0], t[1], t[2]))

    def is_cycle(self, y):
        """Return True if d(y) is zero with no uncertainty"""
        dy = self.d(y)
        return dy["offset"].is_zero() and len(dy["uncertainty"]) == 0

    # (stem, filtration) of the filtration-1 multipliers h0-h3, resolved per
    # class at (n, s, f) as the tridegree (n + s, stem, filtration) -- the
    # same convention as compute_induced_hi_products and the chart writer.
    HI_MULTIPLIER_BIDEGREES = [(stem, 1) for stem in HI_STEMS]

    # Hand-picked multiplier set for resolve_spurious_uncertainties: stable
    # bidegrees that each contain a single basis element, h0-h3 plus the
    # low-stem stable classes chosen for the product deductions.
    SPURIOUS_MULTIPLIER_BIDEGREES = HI_MULTIPLIER_BIDEGREES + [
        (8, 3), (9, 5), (11, 5), (14, 4), (16, 7), (17, 9), (19, 9),
        (19, 3), (20, 4),
    ]

    # Maps whose images carry the two deductions: F commutes with d_r, so
    # F(x) not a boundary => x not a boundary for all four, while the cycle
    # transfer x cycle => F(x) cycle is only used for E, H, P (not C2).
    MAP_NB_NAMES = ("E", "H", "P", "C2")
    MAP_CYCLE_NAMES = ("E", "H", "P")

    def resolve_spurious_uncertainties(self, multiplier_bidegrees=None,
                                       map_names=None, max_stem=None):
        """Close still-open differential values by the product/map reasoning
        in the spurious module (the self-contained replacement for the
        Spurious{r}.txt hand data); see spurious.resolve_spurious_uncertainties
        for the full contract. Returns its counts dict."""
        return spurious.resolve_spurious_uncertainties(
            self, multiplier_bidegrees=multiplier_bidegrees,
            map_names=map_names, max_stem=max_stem)

    def build_pairs(self):
        """Build multiplication pairs for Leibniz constraints."""
        self.pairs = {}

        # Pre-index page by n-value for fast lookup
        by_n = {}
        for n, s, f in self.page:
            if n not in by_n:
                by_n[n] = []
            by_n[n].append((n, s, f))

        # Iterate over x_coords in sorted order by (n, s, f)
        for x_n, x_s, x_f in sorted(self.page.keys()):
            if x_n > self.max_s:
                continue

            # Skip (n, 0, 0) elements - they multiply to zero with everything
            # and create massive numbers of useless pairs that slow down leibniz()
            if x_s == 0 and x_f == 0:
                continue

            x_coords = (x_n, x_s, x_f)
            y_n = x_n + x_s
            bucket = []

            # Only iterate over tridegrees that actually exist with the correct n-value
            if y_n in by_n:
                for y_coords in by_n[y_n]:
                    y_n_actual, y_s, y_f = y_coords

                    # Check that (y_n - 1, y_s, y_f) exists (needed for E-map in Leibniz)
                    if (y_n_actual - 1, y_s, y_f) not in self.page:
                        continue

                    xy_coords = (x_n, x_s + y_s, x_f + y_f)
                    sources = (x_coords, y_coords, xy_coords)

                    # Keep boundary check for Leibniz differential computation
                    if all(self.is_in_computed_polygon_source(*s) for s in sources):
                        bucket.append(y_coords)

            # Sort by (n, s, f) - lowest n, then lowest s, then lowest f
            bucket.sort()
            if bucket:
                self.pairs[x_coords] = bucket

    def is_tridegree_unknown(self, n, s, f):
        """Return True if the uncertainty manager marks the tridegree (n, s, f) as unknown"""
        return self.uncertainty_manager.is_tridegree_unknown(n, s, f)

    def are_any_tridegrees_unknown(self, tridegrees):
        return self.uncertainty_manager.are_any_tridegrees_unknown(tridegrees)

    def remove_uncertain_tridegree(self, tridegree):
        return self.uncertainty_manager.remove_uncertain_tridegree(tridegree)

    def restore_uncertain_tridegree(self, tridegree, data):
        self.uncertainty_manager.restore_uncertain_tridegree(tridegree, data)

    def has_simple_uncertainty(self, n, s, f):
        return self.uncertainty_manager.has_simple_uncertainty(n, s, f)

    def load_d(self, filename):
        """Load differential data from a JSON file into the existing
        DifferentialsPage. Returns the number of entries that were reshaped to
        this session's (smaller) dimensions -- nonzero means the cache covers
        a larger range than this session loaded."""
        return self.d.load_from_file(filename)

    def load_d_csv(self, csv_file):
        """Load differential data from a CSV file into the existing DifferentialsPage"""
        self.d.load_from_csv(csv_file)

    def get_simple_uncertainties_for_r(self, target_r):
        return self.uncertainty_manager.get_simple_uncertainties_for_r(target_r)

    def get_all_uncertainties_for_r(self, target_r):
        return self.uncertainty_manager.get_all_uncertainties_for_r(target_r, self.max_t)

    def get_all_uncertainties(self):
        return self.uncertainty_manager.get_all_uncertainties(self.d)

    def is_element_uncertain(self, element):
        return self.uncertainty_manager.is_element_uncertain(element)

    def element_is_uncertain_plus(self, element):
        r = self.d.r
        return self.uncertainty_manager.element_is_uncertain_plus(element, r)

    def degree_is_uncertain_plus(self, n, s, f):
        """Ask the uncertainty manager whether the tridegree (n, s, f) is uncertain-plus at the current page r"""
        r = self.d.r
        return self.uncertainty_manager.degree_is_uncertain_plus(n, s, f, r)

    def get_lowest_r_uncertainty_for_element(self, element, n, s, f):
        return self.uncertainty_manager.get_lowest_r_uncertainty_for_element(element, n, s, f)

    def add_unknown_tridegree(self, n, s, f, r_value=None):
        self.uncertainty_manager.add_unknown_tridegree(n, s, f, r_value, self.d)

    def save_uncertainties(self, filename):
        self.uncertainty_manager.save_uncertainties(filename)

    def load_uncertainties(self, filename):
        return self.uncertainty_manager.load_uncertainties(filename, self)

    def update_uncertainties_from_differentials(self):
        return self.uncertainty_manager.update_uncertainties_from_differentials(self)

    def propagate_uncertainties_forward(self, previous_uncertain, turned_page):
        self.uncertainty_manager.propagate_uncertainties_forward(previous_uncertain, turned_page, self)

    def transform_uncertainties(self, old_uncertain, make_new_element):
        self.uncertainty_manager.transform_uncertainties(old_uncertain, make_new_element, self)

    def save_d(self, filename):
        """Save the differentials to a JSON file and write the still-unforced tridegrees to a companion _unknown.csv"""
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filename, "w") as diffs:
            json.dump(self.d.to_json(), diffs)
        base_filename = os.path.splitext(filename)[0]
        csv_filename = f"{base_filename}_unknown.csv"
        unknown_tridegrees = []
        for (n, s, f) in self.d.keys():
            if not self.d[n, s, f].is_forced:
                unknown_tridegrees.append((n, s, f))
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["n", "s", "f"])
            for n, s, f in unknown_tridegrees:
                writer.writerow([n, s, f])

    def save_to_directory(self, directory):
        """Save page data to directory (pure file I/O, no computation).

        The data/E{r} directories this writes are ARCHIVAL OUTPUTS, not
        reload inputs: reloading a later page is unsupported (it would lack
        the induced hi product tables, the complete_through coverage bounds,
        and the uncertainty data accumulated over earlier pages -- and the
        loader reads E2-prefixed filenames while this writes E{r}-prefixed
        ones). Every run recomputes all pages fresh from the E2 data."""
        os.makedirs(directory, exist_ok=True)
        r_value = self.d.r

        # Write dimension data
        write_dimension_to_csv(f"{directory}/E{r_value}_rank.csv", self.page)

        # Write products
        write_products(f"{directory}/E{r_value}_relations.csv", self.products)

        # Write the E/H/P/C2 map tables (the hi product tables are charted,
        # not saved)
        for map_name in standard_map_names(self.has_C2, with_hi=False):
            if map_name in self.maps and hasattr(self.maps[map_name], 'table'):
                write_maps(f"{directory}/E{r_value}_{map_name}.csv",
                          self.maps[map_name].table)

    def to_page(self, turned_page):
        """
        Convert turned_page to element page dictionary.

        Maintains invariant: (n,s,f) in page.keys() ⟺ dimension[n,s,f] > 0
        Only adds non-empty bases to avoid storing empty lists.
        """
        element_page = {}  # Regular dict (not defaultdict)
        for (n, s, f), turned_bidegree in turned_page.items():
            if turned_bidegree.basis:  # Only add if non-empty
                element_page[n, s, f] = turned_bidegree.basis
        return element_page

    def compute_dimensions(self):
        """Compute dimension dictionary from page elements"""
        for (n, s, f), bidegree_elements in self.page.items():
            self.dimension[n, s, f] = len(bidegree_elements)

    @property
    def has_C2(self):
        """True when the C2 mapping-cone data is loaded on this page (the C2
        map, and with it the hi product maps, were initialized)."""
        return 'C2' in self.maps

    # Maps whose declared STANDARD_MAPS domain restriction is enforced at
    # runtime. E/H/P/C2 historically ran with an always-true domain; their
    # data tables carry no off-domain rows (and the n=1 column is empty), so
    # off-domain applications were vacuous zero-map constraints -- verified
    # byte-identical on the tot=40 harness with the restrictions enforced.
    HONOR_DOMAIN_CHECKS = ('h0', 'h1', 'h2', 'h3', 'C2', 'P', 'H', 'E')

    def initialize_maps(self, with_C2=True):
        """Initialize map objects from standard templates"""
        for map_name in standard_map_names(with_C2):
            map_template = STANDARD_MAPS[map_name]
            self.maps[map_name] = Map(
                map_template.name,
                map_template.n,
                map_template.s,
                map_template.f,
                domain_check=(map_template.domain_check
                              if map_name in self.HONOR_DOMAIN_CHECKS else None),
                source_transforms=map_template.source_transforms,
            )

    def map_matrix(self, map_name, n, s, f):
        if map_name not in self.maps:
            raise ValueError(f"Unknown map: {map_name}")
        return self.maps[map_name].matrix(n, s, f, self)
    
    def is_in_computed_polygon_source(self, n, s, f):
        return self.is_computable(n, s, f, check_polygon=True, check_uncertainty=False, source=True)

    def is_computable(self, n, s, f, check_polygon=True, check_uncertainty=True, source=False):
        if check_polygon:
            if not all([n >= -1, s >= -1, f >= -1]):
                return False
            if not (f <= self.max_f and s <= self.max_s and s + f <= self.max_t):
                return False
            max_n_bound = 2 * self.max_n - 3 if source else self.max_n
            if n > max_n_bound:
                return False
        if check_uncertainty:
            if self.degree_is_uncertain_plus(n, s, f):
                return False
        return True


    def _product_matrix(self, fixed, var_nsf, xy, fixed_left):
        """Matrix of multiplication by `fixed` on the basis of tridegree
        `var_nsf`: row i is the product of `fixed` with the i-th basis
        element, as a vector in the target tridegree `xy`."""
        nrows = self.dimension[var_nsf]
        ncols = self.dimension[xy]
        if var_nsf not in self.page:
            return matrix(ring=GF(2), nrows=nrows, ncols=ncols, entries=[])

        zero = self.zero(*xy)
        products = []
        for other in self.page[var_nsf]:
            prod = fixed * other if fixed_left else other * fixed
            products.append(prod.vect if prod != zero else zero.vect)

        return matrix(ring=GF(2), nrows=nrows, ncols=ncols, entries=products)

    def L_matrix(self, x, y_n, y_s, y_f):
        """Matrix of left multiplication by `x` on the basis of (y_n, y_s, y_f)."""
        xy = (x.n, x.s + y_s, x.f + y_f)
        return self._product_matrix(x, (y_n, y_s, y_f), xy, fixed_left=True)

    def R_matrix(self, y, x_n, x_s, x_f):
        """Matrix of right multiplication by `y` on the basis of (x_n, x_s, x_f)."""
        xy = (x_n, x_s + y.s, x_f + y.f)
        return self._product_matrix(y, (x_n, x_s, x_f), xy, fixed_left=False)

    def M_matrix(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Full matrix for the map (page[X], page[Y]) -> page[XY]"""
        xy = (x_n, x_s + y_s, x_f + y_f)
        products = [
            (x * y).vect if x * y != self.zero(*xy) else self.zero(*xy).vect
            for x in self.page[x_n, x_s, x_f]
            for y in self.page[y_n, y_s, y_f]
        ]
        return matrix(
            ring=GF(2),
            nrows=self.dimension[x_n, x_s, x_f] * self.dimension[y_n, y_s, y_f],
            ncols=self.dimension[*xy],
            entries=products,
        )

    def EM_matrix(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Full matrix for the map (page[X], page[E^{-1}Y]) -> page[XY]"""
        xy = (x_n, x_s + y_s, x_f + y_f)
        products = [
            (x * y.E()).vect if (x * y.E()) != self.zero(*xy) else self.zero(*xy).vect
            for x in self.page[x_n, x_s, x_f]
            for y in self.page[y_n - 1, y_s, y_f]
        ]
        return matrix(
            ring=GF(2),
            nrows=self.dimension[x_n, x_s, x_f] * self.dimension[y_n - 1, y_s, y_f],
            ncols=self.dimension[*xy],
            entries=products,
        )

    def ELM_dagger(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Rows indexed by the basis of (x_n, x_s, x_f); the row for x is
        the vectorified composite "E then left-multiply by x", as a map
        (y_n - 1, y_s, y_f) -> (x_n, x_s + y_s, x_f + y_f)."""
        xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f
        return matrix(
            ring=GF(2),
            nrows=self.dimension[x_n, x_s, x_f],
            ncols=(self.dimension[y_n - 1, y_s, y_f] * self.dimension[xy_n, xy_s, xy_f]),
            entries=[
                vectorify(self.map_matrix('E', y_n - 1, y_s, y_f) * self.L_matrix(x, y_n, y_s, y_f))
                for x in self.page[x_n, x_s, x_f]
            ],
        )

    def LM_dagger(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Adjoint to left multiplication: rows indexed by the basis of
        (x_n, x_s, x_f); the row for x is the vectorified matrix of left
        multiplication by x on (y_n, y_s, y_f)."""
        xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f
        return matrix(
            ring=GF(2),
            nrows=self.dimension[x_n, x_s, x_f],
            ncols=(self.dimension[y_n, y_s, y_f] * self.dimension[xy_n, xy_s, xy_f]),
            entries=[
                vectorify(self.L_matrix(x, y_n, y_s, y_f)) for x in self.page[x_n, x_s, x_f]
            ],
        )

    def RM_dagger(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Adjoint to right multiplication: rows indexed by the basis of
        (y_n, y_s, y_f); the row for y is the vectorified matrix of right
        multiplication by y on (x_n, x_s, x_f)."""
        xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f
        return matrix(
            ring=GF(2),
            nrows=self.dimension[y_n, y_s, y_f],
            ncols=(self.dimension[x_n, x_s, x_f] * self.dimension[xy_n, xy_s, xy_f]),
            entries=[vectorify(self.R_matrix(y, x_n, x_s, x_f)) for y in self.page[y_n, y_s, y_f]],
        )

    def ERM_dagger(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Rows indexed by the basis of (y_n - 1, y_s, y_f); the row for y is
        the vectorified matrix of right multiplication by E(y) on
        (x_n, x_s, x_f), or a zero row when E(y) = 0."""
        xy = (x_n, x_s + y_s, x_f + y_f)
        ncols = self.dimension[x_n, x_s, x_f] * self.dimension[*xy]
        zero_vec = vector(GF(2), [0] * ncols)

        entries = [
            zero_vec
            if (y_E := y.E()) == self.zero(y_n, y_s, y_f)
            else vectorify(self.R_matrix(y_E, x_n, x_s, x_f))
            for y in self.page[y_n - 1, y_s, y_f]
        ]

        return matrix(
            ring=GF(2),
            nrows=self.dimension[y_n - 1, y_s, y_f],
            ncols=self.dimension[x_n, x_s, x_f] * self.dimension[*xy],
            entries=entries,
        )
    def _constraint_failure(self, label, exc, tridegrees):
        """Report an inconsistent constraint application: print the failing
        constraint, the full proof chain of every tridegree involved (the exact
        deduction steps that led to the contradiction), and dump the chains to
        debug_graphs/ as .dot files (render them with why.py). Reaching this
        means the accumulated constraints contradict each other, so either the
        input data or a recorded differential is wrong; the caller re-raises."""
        print(f"\n{'=' * 72}")
        print(f"ERROR in {label}: {type(exc).__name__}: {exc}")
        print(f"(page r={self.d.r}, proof counter={self.d.counter})")

        # Console trace: the deduction steps behind every differential that fed
        # this constraint. Step labels name the map/rule; (x)/(y) are the
        # tridegrees that step combined. Cross-reference with self.d.get_step().
        seen = set()
        for td in tridegrees:
            if td in seen:
                continue
            seen.add(td)
            print(f"\n--- d{td} ---")
            try:
                print(self.d[td[0], td[1], td[2]].describe())
            except Exception as describe_exc:
                print(f"  (unavailable: {describe_exc})")

        os.makedirs("debug_graphs", exist_ok=True)
        print()
        for td in seen:
            filename = f"debug_graphs/why_error_{td[0]}_{td[1]}_{td[2]}.dot"
            try:
                import why
                why.write_why_graph(self.d, td, filename=filename)
                print(f"  proof chain for d{td}: {filename}")
            except Exception as graph_exc:
                print(f"  (no proof chain for d{td}: {graph_exc})")
        print(f"{'=' * 72}")

    def natural_rev(self, y_n, y_s, y_f, map_name, locked=True):
        """Apply naturality constraint in reverse direction (from codomain to domain)"""
        map_obj = self.maps[map_name]
        x_n, x_s, x_f = map_obj.source_degree(y_n, y_s, y_f)

        # The computed source must actually be in the map's domain (the degree
        # inverse alone cannot see restrictions like the hi maps' n == 0).
        if not map_obj.domain_check(x_n, x_s, x_f):
            return False

        return self._reverse_naturality_at(map_name, x_n, x_s, x_f,
                                           y_n, y_s, y_f, locked=locked)

    def _natural_rev_source_sweep(self, map_name, extra_guard=None,
                                  soft_structural=False):
        """Reverse naturality indexed from the map's DOMAIN, for maps whose
        degree shift is not invertible and so cannot be swept from the
        codomain (C2: every odd n >= 3 sphere lands on the n=0 column).
        extra_guard(x_n, x_s, x_f, y_n, y_s, y_f) may veto individual sites;
        returns True if any differential space shrank."""
        changed = False
        map_obj = self.maps[map_name]
        for x_n, x_s, x_f in self.map_domain(map_name):
            y_n, y_s, y_f = map_obj.target_degree(x_n, x_s, x_f)
            if extra_guard and not extra_guard(x_n, x_s, x_f, y_n, y_s, y_f):
                continue
            changed |= self._reverse_naturality_at(
                map_name, x_n, x_s, x_f, y_n, y_s, y_f,
                soft_structural=soft_structural)
        return changed

    def _reverse_naturality_at(self, map_name, x_n, x_s, x_f, y_n, y_s, y_f,
                               locked=True, soft_structural=False):
        """Core of reverse naturality: constrain d at the map's source x
        against d at its target y through the naturality square. With
        soft_structural, non-contradiction failures (shape mismatches at
        data boundaries) warn and skip instead of aborting."""
        r = self.d.r
        map_obj = self.maps[map_name]

        # Skip if locked and already forced
        if locked and self.d[x_n, x_s, x_f].is_forced:
            return False

        relevant_tridegrees = [
            (x_n, x_s, x_f),
            self.d_target(x_n, x_s, x_f),
            (y_n, y_s, y_f),
            self.d_target(y_n, y_s, y_f)
        ]
        if not self._tridegrees_ok(relevant_tridegrees):
            return False
        if not self._map_data_ok(map_obj, x_n, x_s, x_f):
            return False

        old_dx = self.d[x_n, x_s, x_f].dimension()

        def apply():
            map_matrix_x = self.map_matrix(map_name, x_n, x_s, x_f)
            map_matrix_source = self.map_matrix(map_name, x_n, x_s - 1, x_f + r)
            self.d[x_n, x_s, x_f] &= (map_matrix_x * self.d[y_n, y_s, y_f]) // map_matrix_source
            return True

        if not self._apply_constraint(
                f"natural_rev({map_name}) at source ({x_n},{x_s},{x_f}), target ({y_n},{y_s},{y_f})",
                [(x_n, x_s, x_f), (y_n, y_s, y_f),
                 self.d_target(x_n, x_s, x_f), self.d_target(y_n, y_s, y_f)],
                apply, soft_structural=soft_structural):
            return False

        new_dx = self.d[x_n, x_s, x_f].dimension()
        changed = old_dx > new_dx

        if changed:
            old_dy = self.d[y_n, y_s, y_f].dimension()
            new_dy = old_dy
            self.outcome_map(map_name, x_n, x_s, x_f, y_n, y_s, y_f, old_dx, new_dx, old_dy, new_dy)

        return changed

    def natural(self, x_n, x_s, x_f, map_name, locked=True):
        """Apply naturality constraint in forward direction (from domain to codomain)"""
        r = self.d.r
        map_obj = self.maps[map_name]

        # The source must lie in the map's domain (mirrors natural_rev; for
        # E/H/P/C2 this is currently always true, see initialize_maps).
        if not map_obj.domain_check(x_n, x_s, x_f):
            return False

        y_n, y_s, y_f = map_obj.target_degree(x_n, x_s, x_f)

        # Skip if locked and already forced
        if locked and self.d[y_n, y_s, y_f].is_forced:
            return False

        relevant_tridegrees = [
            (x_n, x_s, x_f),
            self.d_target(x_n, x_s, x_f),
            (y_n, y_s, y_f),
            self.d_target(y_n, y_s, y_f)
        ]
        if not self._tridegrees_ok(relevant_tridegrees):
            return False
        if not self._map_data_ok(map_obj, x_n, x_s, x_f):
            return False

        old_dy = self.d[y_n, y_s, y_f].dimension()

        def apply():
            map_matrix_x = self.map_matrix(map_name, x_n, x_s, x_f)
            map_matrix_source = self.map_matrix(map_name, x_n, x_s - 1, x_f + r)
            self.d[y_n, y_s, y_f] &= map_matrix_x // (self.d[x_n, x_s, x_f] * map_matrix_source)

        self._apply_constraint(
            f"natural({map_name}) at source ({x_n},{x_s},{x_f}), target ({y_n},{y_s},{y_f})",
            [(x_n, x_s, x_f), (y_n, y_s, y_f),
             self.d_target(x_n, x_s, x_f), self.d_target(y_n, y_s, y_f)],
            apply)
        new_dy = self.d[y_n, y_s, y_f].dimension()
        changed = old_dy > new_dy

        if changed:
            old_dx = self.d[x_n, x_s, x_f].dimension()
            new_dx = old_dx
            self.outcome_map(map_name, x_n, x_s, x_f, y_n, y_s, y_f, old_dx, new_dx, old_dy, new_dy)

        return changed

    def _c2_reverse_guard(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Extra C2-only bound for the reverse sweep: both the sphere source
        and the n=0 column target must lie inside the C2 data's complete band
        (total degree <= MAX_TOTAL_DEGREE_THRESHOLD). Note complete_through
        (checked by _map_data_ok) bounds the C2 TABLE's source coverage,
        while this also bounds the column target's d-target total degree --
        related but not interchangeable quantities."""
        r = self.d.r
        if (y_s - 1 + y_f + r > MAX_TOTAL_DEGREE_THRESHOLD
                or x_s + x_f > MAX_TOTAL_DEGREE_THRESHOLD):
            return False
        return y_s >= 0 and y_f >= 0

    def d_squared(self):
        """Check d² = 0 constraints for all tridegrees with elements"""
        r = self.d.r
        changed = False

        for n, s, f in self.all_tridegrees_with_elements():
            source_td = (n, s + 1, f - r)
            target_td = (n, s, f)
            
            d1 = self.d.get(source_td)
            d2 = self.d.get(target_td)
            if d1 is None or d2 is None:
                continue
            if d1.is_forced and not d2.is_forced:
                if source_td in self.uncertain or target_td in self.uncertain:
                    continue
                if d1.v.is_zero():
                    continue
                d = d1.as_matrix(d1.v)
                old_dim = d2.dimension()
                zero_space = AffineMatrixSubspace(d1.nrows, d2.ncols)
                zero_space.set_subspace(
                    zero_space.ambient.zero(),
                    zero_space.ambient.span([])
                )
                constraint = d // zero_space

                def apply():
                    self.d[target_td] = d2 & constraint
                self._apply_constraint(
                    f"d^2=0 constraining d{target_td} from forced d{source_td}",
                    [source_td, target_td], apply)
                new_dim = self.d[target_td].dimension()
                if old_dim > new_dim:
                    self.outcome_map('d^2', *source_td, *target_td, 0, 0, old_dim, new_dim)
                    changed = True
            elif not d1.is_forced and d2.is_forced:
                if source_td in self.uncertain or target_td in self.uncertain:
                    continue
                if d2.v.is_zero():
                    continue
                old_dim = d1.dimension()
                d = d2.as_matrix(d2.v)
                zero_space = AffineMatrixSubspace(d1.nrows, d2.ncols)
                zero_space.set_subspace(
                    zero_space.ambient.zero(),
                    zero_space.ambient.span([])
                )
                def apply():
                    self.d[source_td] = d1 & (zero_space // d)
                self._apply_constraint(
                    f"d^2=0 constraining d{source_td} from forced d{target_td}",
                    [source_td, target_td], apply)
                new_dim = self.d[source_td].dimension()
                if old_dim > new_dim:
                    self.outcome_map('d^2', *source_td, *target_td, old_dim, new_dim, 0, 0)
                    changed = True
        return changed

    def desuspend(self):
        """Apply desuspension (E map stability) using E map codomain"""
        r = self.d.r
        changed = False
        for y_n, y_s, y_f in self.map_codomain('E'):
            x_n, x_s, x_f = self.maps['E'].source_degree(y_n, y_s, y_f)
            relevant_tridegrees = [
                (y_n, y_s - 1, y_f + r),
                (y_n, y_s, y_f),
                (x_n, x_s - 1, x_f + r),
                (x_n, x_s, x_f),
            ]
            if not self._tridegrees_ok(relevant_tridegrees):
                continue
            if not self._map_data_ok(self.maps['E'], x_n, x_s, x_f):
                continue
            old_dx = self.d[x_n, x_s, x_f].dimension()

            def apply():
                map_matrix_x = self.map_matrix('E', x_n, x_s, x_f)
                map_matrix_source = self.map_matrix('E', x_n, x_s - 1, x_f + r)
                self.d[x_n, x_s, x_f] &= \
                    (map_matrix_x * self.d[y_n, y_s, y_f]) // map_matrix_source
            self._apply_constraint(
                f"desuspend at ({x_n},{x_s},{x_f})",
                [(x_n, x_s, x_f), (y_n, y_s, y_f)], apply)

            # This constraint only touches d[x]; the y-side dims are passed
            # equal so outcome_stable never reports a y change.
            new_dx = self.d[x_n, x_s, x_f].dimension()
            dy = self.d[y_n, y_s, y_f].dimension()
            self.outcome_stable('E', x_n, x_s, x_f, y_n, y_s, y_f,
                                old_dx, new_dx, dy, dy)
            changed |= old_dx > new_dx
        return changed

    def leibniz(self, pairs_to_use=None):
        """Apply Leibniz constraints for products.

        Args:
            pairs_to_use: Optional pairs dictionary to use instead of self.pairs.
                         If None, uses self.pairs.
        """
        if pairs_to_use is None:
            pairs_to_use = self.pairs

        changed = False
        for x_coords, y_coords_list in pairs_to_use.items():
            x_n, x_s, x_f = x_coords
            for y_coords in y_coords_list:
                y_n, y_s, y_f = y_coords
                xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f

                # Display current tridegrees being processed (updates in place)
                print(f"\r  Computing Leibniz: x=({x_n},{x_s},{x_f}) \u00d7 y=({y_n},{y_s},{y_f}) \u2192 xy=({xy_n},{xy_s},{xy_f})    ", end='', flush=True)

                relevant_tridegrees = [
                    (x_n, x_s, x_f),
                    self.d_target(x_n, x_s, x_f),
                    (y_n-1, y_s, y_f),
                    self.d_target(y_n-1, y_s, y_f),
                    (y_n, y_s, y_f),
                    self.d_target(y_n, y_s, y_f),
                    (xy_n, xy_s, xy_f),
                    self.d_target(xy_n, xy_s, xy_f)
                ]
                if not self._tridegrees_ok(relevant_tridegrees):
                    continue

                is_worth_it = not all([
                        self.d[x_n, x_s, x_f].is_forced,
                        self.d[y_n, y_s, y_f].is_forced,
                        self.d[xy_n, xy_s, xy_f].is_forced,
                    ])

                while is_worth_it:
                    old_dx = self.d[x_n, x_s, x_f].dimension()
                    old_dy = self.d[y_n, y_s, y_f].dimension()
                    old_dxy = self.d[xy_n, xy_s, xy_f].dimension()

                    if self.d[x_n, x_s, x_f].is_cycle():
                        for x in self.page[x_n, x_s, x_f]:
                            self.leibniz_x(x, y_n, y_s, y_f)
                    elif self.d[y_n, y_s, y_f].is_cycle():
                        for y in self.page[y_n - 1, y_s, y_f]:
                            if y.E() != self.zero(y_n, y_s, y_f) and self.is_cycle(y.E()):
                                self.leibniz_y(y, x_n, x_s, x_f)
                    else:
                        self.leibniz_full(x_n, x_s, x_f, y_n, y_s, y_f)

                    new_dx = self.d[x_n, x_s, x_f].dimension()
                    new_dy = self.d[y_n, y_s, y_f].dimension()
                    new_dxy = self.d[xy_n, xy_s, xy_f].dimension()

                    progress_made = old_dx > new_dx or old_dy > new_dy or old_dxy > new_dxy
                    self.outcome(x_n, x_s, x_f, y_n, y_s, y_f, old_dx, new_dx, old_dy, new_dy, old_dxy, new_dxy)
                    changed |= progress_made

                    self.natural(x_n, x_s, x_f, 'E', locked=True)

                    is_worth_it = progress_made and not all([
                        self.d[x_n, x_s, x_f].is_forced,
                        self.d[y_n, y_s, y_f].is_forced,
                        self.d[xy_n, xy_s, xy_f].is_forced,
                    ])

            # Apply E map naturality after processing all y_coords for this x_coords
            self.natural(x_n, x_s, x_f, 'E', locked=True)

        print()  # New line after leibniz completes
        return changed

    def leibniz_full(self, x_n, x_s, x_f, y_n, y_s, y_f):
        """Apply the full Leibniz rule d(xy) = d(x)y + x d(y) at the level of affine matrix subspaces, constraining the differentials at x, y, and the product tridegree"""
        r = self.d.r
        xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f

        def apply():
            if not self.d[x_n, x_s, x_f].is_forced:
                self.d[x_n, x_s, x_f] &= (
                    self.ELM_dagger(x_n, x_s, x_f, y_n, y_s, y_f)
                    * self.d[xy_n, xy_s, xy_f].lower_star(self.dimension[y_n - 1, y_s, y_f])
                    + self.LM_dagger(x_n, x_s, x_f, y_n, y_s - 1, y_f + r)
                    * (self.map_matrix('E', y_n - 1, y_s, y_f) * self.d[y_n, y_s, y_f]).upper_star(self.dimension[xy_n, xy_s - 1, xy_f + r])
                ) // self.LM_dagger(x_n, x_s - 1, x_f + r, y_n - 1, y_s, y_f)
            if not self.d[y_n, y_s, y_f].is_forced:
                self.d[y_n, y_s, y_f] &= (
                    self.map_matrix('E', y_n - 1, y_s, y_f) // (
                        self.ERM_dagger(x_n, x_s, x_f, y_n, y_s, y_f) * self.d[xy_n, xy_s, xy_f].lower_star(self.dimension[x_n, x_s, x_f])
                        + self.RM_dagger(x_n, x_s - 1, x_f + r, y_n - 1, y_s, y_f) * self.d[x_n, x_s, x_f].upper_star(self.dimension[xy_n, xy_s - 1, xy_f + r])
                    ) // self.RM_dagger(x_n, x_s, x_f, y_n, y_s - 1, y_f + r)
                )
            if not self.d[xy_n, xy_s, xy_f].is_forced:
                self.d[xy_n, xy_s, xy_f] &= self.EM_matrix(x_n, x_s, x_f, y_n, y_s, y_f) // (
                    self.d[x_n, x_s, x_f].tensor(self.dimension[y_n - 1, y_s, y_f]) * self.M_matrix(x_n, x_s - 1, x_f + r, y_n - 1, y_s, y_f)
                    + (self.map_matrix('E', y_n - 1, y_s, y_f) * self.d[y_n, y_s, y_f]).rtensor(self.dimension[x_n, x_s, x_f]) * self.M_matrix(x_n, x_s, x_f, y_n, y_s - 1, y_f + r)
                )

        self._apply_constraint(
            f"leibniz_full: x=({x_n},{x_s},{x_f}) * y=({y_n},{y_s},{y_f}) -> xy=({xy_n},{xy_s},{xy_f})",
            [(x_n, x_s, x_f), (x_n, x_s - 1, x_f + r),
             (y_n, y_s, y_f), (y_n - 1, y_s, y_f), (y_n, y_s - 1, y_f + r),
             (xy_n, xy_s, xy_f), (xy_n, xy_s - 1, xy_f + r)],
            apply)

    def leibniz_x(self, x, y_n, y_s, y_f):
        """Apply the Leibniz rule with a fixed cycle x (so the d(x) term vanishes): constrain the differentials at (y_n, y_s, y_f) and at the product tridegree via left-multiplication by x and the E map"""
        r = self.d.r
        x_n, x_s, x_f = x.n, x.s, x.f
        xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f

        relevant_tridegrees = [
            (y_n, y_s, y_f),
            (y_n, y_s - 1, y_f + r),
            (y_n - 1, y_s, y_f),
            (xy_n, xy_s, xy_f),
            (xy_n, xy_s - 1, xy_f + r)
        ]
        if not self._tridegrees_ok(relevant_tridegrees):
            return

        if (self.d[y_n, y_s, y_f].is_forced
                and self.d[xy_n, xy_s, xy_f].is_forced):
            return

        # Both branches solve the same commutative relation
        #   e_matrix . d[y] . L_source = e_matrix . L_target . d[xy]
        # for whichever side is not yet forced; the second branch reads d[y]
        # after the first may have shrunk it.
        def apply():
            e_matrix = self.map_matrix('E', y_n - 1, y_s, y_f)
            L_target = self.L_matrix(x, y_n, y_s, y_f)
            L_source = self.L_matrix(x, y_n, y_s - 1, y_f + r)

            if not self.d[y_n, y_s, y_f].is_forced:
                self.d[y_n, y_s, y_f] &= e_matrix // (
                    e_matrix * L_target * self.d[xy_n, xy_s, xy_f]
                ) // L_source

            if not self.d[xy_n, xy_s, xy_f].is_forced:
                self.d[xy_n, xy_s, xy_f] &= (e_matrix * L_target) // (
                    e_matrix * self.d[y_n, y_s, y_f] * L_source
                )

        self._apply_constraint(
            f"leibniz_x: cycle x={x} at ({x_n},{x_s},{x_f}), y at ({y_n},{y_s},{y_f})",
            [(y_n, y_s, y_f), (y_n, y_s - 1, y_f + r),
             (xy_n, xy_s, xy_f), (xy_n, xy_s - 1, xy_f + r)],
            apply)

    def leibniz_y(self, y, x_n, x_s, x_f):
        """Apply the Leibniz rule with a fixed cycle y (so the d(y) term vanishes): constrain the differentials at (x_n, x_s, x_f) and at the product tridegree via right-multiplication by y and its suspension y.E()"""
        r = self.d.r
        y_n, y_s, y_f = y.n + 1, y.s, y.f
        xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f

        relevant_tridegrees = [
            (x_n, x_s, x_f),
            (x_n, x_s - 1, x_f + r),
            (xy_n, xy_s, xy_f),
            (xy_n, xy_s - 1, xy_f + r)
        ]
        if not self._tridegrees_ok(relevant_tridegrees):
            return

        def apply():
            if not self.d[x_n, x_s, x_f].is_forced:
                self.d[x_n, x_s, x_f] &= (
                    self.R_matrix(y.E(), x_n, x_s, x_f)
                    * self.d[xy_n, xy_s, xy_f]
                ) // self.R_matrix(y, x_n, x_s - 1, x_f + r)

            if not self.d[xy_n, xy_s, xy_f].is_forced:
                self.d[xy_n, xy_s, xy_f] &= self.R_matrix(y.E(), x_n, x_s, x_f) // (self.d[x_n, x_s, x_f] * self.R_matrix(y, x_n, x_s - 1, x_f + r))

        self._apply_constraint(
            f"leibniz_y: cycle y={y} at ({y.n},{y_s},{y_f}), x at ({x_n},{x_s},{x_f})",
            [(x_n, x_s, x_f), (x_n, x_s - 1, x_f + r),
             (xy_n, xy_s, xy_f), (xy_n, xy_s - 1, xy_f + r)],
            apply)


    def _tridegrees_ok(self, tridegrees, *, polygon=True, uncertainty=True, source=True):
        """Shared guard: every tridegree lies in the computed polygon and is
        free of uncertain-plus contamination. Used by the constraint methods
        (natural, natural_rev, C2 naturality, desuspend, leibniz family).
        d_squared intentionally differs: it checks raw membership in
        self.uncertain and skips zero differentials instead."""
        return all(
            self.is_computable(*td, check_polygon=polygon,
                               check_uncertainty=uncertainty, source=source)
            for td in tridegrees
        )

    def _map_data_ok(self, map_obj, x_n, x_s, x_f):
        """The map matrices at x and at its d-target must come from complete
        table data; beyond the data file's coverage a missing entry would be
        read as a false zero map, producing bogus constraints."""
        r = self.d.r
        return (map_obj.data_complete(x_n, x_s, x_f)
                and map_obj.data_complete(x_n, x_s - 1, x_f + r))

    def _apply_constraint(self, label, tridegrees, apply_fn, soft_structural=False):
        """Run apply_fn() under the standard failure policy: a
        ContradictionError (the math is inconsistent) is always fatal and
        reported with the full deduction trace; any other failure --
        typically a matrix shape mismatch at a data boundary -- is also
        fatal in generic constraint code, but warns and skips when
        soft_structural (expected at the C2 data edge). Returns apply_fn()'s
        value, or False when a soft failure was skipped."""
        try:
            return apply_fn()
        except ContradictionError as e:
            self._constraint_failure(label, e, tridegrees)
            raise
        except Exception as e:
            if soft_structural:
                print(f"\n  warning: {label} skipped: {type(e).__name__}: {e}")
                return False
            self._constraint_failure(label, e, tridegrees)
            raise

    def _record_deduction(self, w_nsf, x_nsf, y_nsf, label):
        """Record the proof reason for a shrink at tridegree w (state
        mutation only; call exactly once per actual shrink)."""
        self.d.counter += 1
        self.d[w_nsf].add_reason(self.d.counter, *x_nsf, *y_nsf, label)

    def _print_deduction(self, w_nsf, old_dim, new_dim):
        """Console report of one dimension drop, listing the newly forced
        differentials when the constraint space becomes a point."""
        w_n, w_s, w_f = w_nsf
        r = self.d.r
        print(f"    dim d[{w_n},{w_s},{w_f}]: {old_dim} => {new_dim}")
        if new_dim == 0:
            diff = self.d[w_nsf].as_matrix(self.d[w_nsf].v)
            print(f"    new differential{'s' if self.dimension[w_nsf] > 1 else ''} found!")
            if w_nsf in self.page:
                for z in self.page[w_nsf]:
                    dz = Element(w_n, w_s - 1, w_f + r, z.vect * diff,
                                 spectral_sequence=self)
                    print(f"        d({z}) = {dz}")

    def _outcome(self, header, updates, x_nsf, y_nsf, label):
        """Print and record every actual shrink in `updates`
        ([(w_nsf, old_dim, new_dim)]); the provenance pair (x_nsf, y_nsf)
        and the proof label go into each recorded reason."""
        if not any(old > new for _, old, new in updates):
            return
        print(header)
        for w_nsf, old, new in updates:
            if old > new:
                self._print_deduction(w_nsf, old, new)
                self._record_deduction(w_nsf, x_nsf, y_nsf, label)

    def outcome_stable(self, map_name, x_n, x_s, x_f, y_n, y_s, y_f, old_dx, new_dx, old_dy, new_dy):
        self._outcome(
            f"From {map_name}({x_n},{x_s},{x_f}):                    ",
            [((x_n, x_s, x_f), old_dx, new_dx), ((y_n, y_s, y_f), old_dy, new_dy)],
            (x_n, x_s, x_f), (y_n, y_s, y_f), "stable")

    def outcome_map(self, map_name, x_n, x_s, x_f, y_n, y_s, y_f, old_dx, new_dx, old_dy, new_dy):
        if old_dx > new_dx or old_dy > new_dy:
            # Real maps land in the naturality summary; constraint rules
            # like d^2 are tallied separately.
            counts = (self.map_deduction_counts if map_name in self.maps
                      else self.rule_deduction_counts)
            counts[map_name] += 1
        self._outcome(
            f"From {map_name}({x_n},{x_s},{x_f}):                    ",
            [((x_n, x_s, x_f), old_dx, new_dx), ((y_n, y_s, y_f), old_dy, new_dy)],
            (x_n, x_s, x_f), (y_n, y_s, y_f), map_name)

    def outcome(self, x_n, x_s, x_f, y_n, y_s, y_f, old_dx, new_dx, old_dy, new_dy, old_dxy, new_dxy):
        xy_nsf = (x_n, x_s + y_s, x_f + y_f)
        self._outcome(
            f"From ({x_n},{x_s},{x_f}) o ({y_n},{y_s},{y_f}):                    ",
            [((x_n, x_s, x_f), old_dx, new_dx), ((y_n, y_s, y_f), old_dy, new_dy),
             (xy_nsf, old_dxy, new_dxy)],
            (x_n, x_s, x_f), (y_n, y_s, y_f), "")

    def get_active_pairs(self):
        """Get only pairs where at least one of (x, y, xy) is not forced.

        This significantly speeds up Leibniz computation by skipping pairs
        where all three differentials are already determined.

        Returns:
            Dictionary mapping x_coords -> [y_coords] for active pairs only
        """
        active_pairs = {}

        for x_coords, y_coords_list in self.pairs.items():
            x_n, x_s, x_f = x_coords
            active_y_list = []

            for y_coords in y_coords_list:
                y_n, y_s, y_f = y_coords
                xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f

                # Check if at least one differential is not forced
                if not (self.d[x_n, x_s, x_f].is_forced and
                        self.d[y_n, y_s, y_f].is_forced and
                        self.d[xy_n, xy_s, xy_f].is_forced):
                    active_y_list.append(y_coords)

            if active_y_list:
                active_pairs[x_coords] = active_y_list

        return active_pairs

    def _natural_sweep(self):
        """One forward-naturality pass over every map's domain; True if any
        differential space shrank."""
        changed = False
        for map_name in self.maps.keys():
            for (n, s, f) in self.map_domain(map_name):
                changed |= self.natural(n, s, f, map_name, locked=True)
        return changed

    def _natural_rev_sweep(self):
        """One reverse-naturality pass over every map's codomain; True if any
        differential space shrank."""
        changed = False
        for map_name in self.maps.keys():
            for (n, s, f) in self.map_codomain(map_name):
                changed |= self.natural_rev(n, s, f, map_name, locked=True)
        return changed

    def compute(self, use_active_pairs=True):
        """Compute differentials using constraints.

        Iterates naturality, C2 naturality, d^2 = 0, and Leibniz passes until
        a full outer pass makes no change anywhere (a true fixpoint).

        Args:
            use_active_pairs: If True, only process Leibniz pairs where at least
                            one differential is not forced (significant speedup!)
        """
        # Impose the authoritative known-differential inputs (the entry-level
        # CSVs stable/stable_sphere_diffs.csv and stable/c2_diffs.csv; see
        # stable/README.md) FIRST, so every deduction below starts from the
        # full input data. check_stable() reads no in-page state, so this
        # single call pins everything the inputs cover; repeating it inside
        # the loop below would be a no-op.
        self.d.check_stable()
        if self.has_C2:
            self._natural_rev_source_sweep(
                'C2', extra_guard=self._c2_reverse_guard, soft_structural=True)
        loop_count = 0

        # Get active pairs once at the start if using optimization
        if use_active_pairs:
            active_pairs = self.get_active_pairs()
            total_pairs = sum(len(self.pairs[x]) for x in self.pairs)
            active_count = sum(len(active_pairs[x]) for x in active_pairs)
            print(f"Active pairs optimization: {active_count}/{total_pairs} pairs active ({100*active_count/total_pairs:.1f}%)")
        else:
            active_pairs = None

        while True:
            loop_count += 1
            progress = False
            ehp_loop_count = 0
            while True:
                ehp_loop_count += 1
                maps_changed = self._natural_sweep()
                rev_changed = self._natural_rev_sweep()
                print(f"  EHP iteration {ehp_loop_count}: maps_changed={maps_changed}, rev_changed={rev_changed}")
                if not maps_changed and not rev_changed:
                    break
                progress = True
            # Reverse C2 naturality (n=0 column -> odd spheres): C2's degree
            # shift is not invertible, so this runs source-indexed instead of
            # inside the codomain-indexed generic sweep above. With complete
            # input CSVs it is a no-op after the initial call, but n=0
            # tridegrees beyond the inputs' coverage (r >= 6 past total
            # degree HIGH_R_TRIVIAL_TOT) can still shrink mid-page and feed
            # the odd spheres.
            if self.has_C2:
                progress |= self._natural_rev_source_sweep(
                    'C2', extra_guard=self._c2_reverse_guard,
                    soft_structural=True)
            progress |= self.d_squared()
            progress |= self._natural_rev_sweep()
            progress |= self._natural_sweep()
            progress |= self.leibniz(pairs_to_use=active_pairs)
            progress |= self._natural_rev_sweep()
            progress |= self._natural_sweep()
            progress |= self.d_squared()
            # Exit only when an entire outer pass changed nothing;
            # --verify-converged in run.py independently re-checks this.
            if not progress:
                break
        # One desuspension pass as a final consistency check: unlike the
        # locked natural_rev sweeps, desuspend() re-applies the E-map square
        # at already-forced sources, so two inconsistent forced differentials
        # raise a contradiction instead of being silently skipped. On a
        # converged page it deduces nothing new (it is the same constraint
        # as natural_rev on the E map).
        self.desuspend()
        print(f"compute() finished after {loop_count} main iterations")
        if self.map_deduction_counts:
            summary = ", ".join(
                f"{name}: {count}"
                for name, count in sorted(self.map_deduction_counts.items()))
            print(f"  naturality deductions by map: {summary}")
        else:
            print("  naturality deductions by map: none")
        if self.rule_deduction_counts:
            summary = ", ".join(
                f"{name}: {count}"
                for name, count in sorted(self.rule_deduction_counts.items()))
            print(f"  rule deductions: {summary}")

    def next_page(self):
        """
        Compute the next page E_{r+1} from E_r (pure mathematical computation).

        This method focuses solely on the mathematical aspects:
        - Computing homology (turning the page)
        - Setting up the basic structure of the next page
        - Propagating uncertainties

        It does NOT:
        - Compute induced maps or products (handled by SpectralSequence)
        - Save to disk (use save_to_directory() separately)
        - Build caches (done after induced structures are computed)
        """
        # Carry max_t forward unchanged: the usable region already shrinks by
        # one total degree per page through the growing differential reach
        # (d_r raises total degree by r-1, enforced by the target polygon
        # check in is_computable).
        new_ss = SpectralSequencePage(max_t=self.max_t)

        # Turn the page: compute homology H(E_r, d_r) = E_{r+1}
        turned_page = self.d.turn_page_with_uncertainty(self.page, new_ss)
        self.turned_page = turned_page  # Store for lift/quotient operations

        # Extract basis elements from turned page
        new_ss.page = new_ss.to_page(turned_page)
        new_ss.compute_dimensions()
        new_ss.compute_max_values()

        # Propagate uncertainty information forward
        self.update_uncertainties_from_differentials()
        new_ss.propagate_uncertainties_forward(self.uncertain, turned_page)

        # Initialize differential structure for next page
        new_ss.d = DifferentialsPage(r=self.d.r + 1, dimension_dict=new_ss.dimension)

        # Initialize map objects (but not their tables - computed by
        # SpectralSequence), mirroring this page's map set
        new_ss.initialize_maps(with_C2=self.has_C2)

        return new_ss

    def lift(self, x):
        """Lift an element from next page basis back to current page"""
        if self.turned_page is None:
            raise ValueError("Must call next_page() before using lift()")
        x_n, x_s, x_f = x.n, x.s, x.f
        if (x_n, x_s, x_f) not in self.turned_page:
            return self.zero(x_n, x_s, x_f)
        return Element(
            x_n, x_s, x_f,
            self.turned_page[x_n, x_s, x_f].quotient_map.lift(x.vect),
            spectral_sequence=self
        )

    def quotient(self, x, target_n=None, target_s=None, target_f=None):
        """Apply quotient map to element to get element in next page"""
        if self.turned_page is None:
            raise ValueError("Must call next_page() before using quotient()")
        # Use target if provided, otherwise use x's coordinates
        target_n = target_n if target_n is not None else x.n
        target_s = target_s if target_s is not None else x.s
        target_f = target_f if target_f is not None else x.f
        if (target_n, target_s, target_f) not in self.turned_page:
            return self.zero(target_n, target_s, target_f)
        return Element(
            target_n, target_s, target_f,
            self.turned_page[target_n, target_s, target_f].quotient_map(x.vect),
            spectral_sequence=self
        )

    def in_next_basis(self, x):
        """Reduce element against boundaries to express in next page basis"""
        if self.turned_page is None:
            raise ValueError("Must call next_page() before using in_next_basis()")
        x_n, x_s, x_f = x.n, x.s, x.f
        if (x_n, x_s, x_f) not in self.turned_page:
            return x.vect
        return reduce_against(x.vect, self.turned_page[x_n, x_s, x_f].B)

    def _chart_hi_multipliers(self, col):
        """The single basis element at each hi multiplier bidegree
        (col, stem, 1) of chart column col = n + s, aligned with
        C2_PRODUCT_MAPS; None where the bidegree is empty. The [0]
        resolutions treat each bidegree as one-dimensional; a dim >= 2
        bidegree would silently drop products from the chart -- vacuous on
        the shipped range (verified), asserted against future extensions."""
        multipliers = []
        for stem in HI_STEMS:
            if self.has_elements(col, stem, 1):
                if len(self.page[col, stem, 1]) > 1:
                    raise AssertionError(
                        f"write_spheres: hi multiplier bidegree "
                        f"({col}, {stem}, 1) has dimension >= 2; the "
                        f"single-element resolution would drop products")
                multipliers.append(self.page[col, stem, 1][0])
            else:
                multipliers.append(None)
        return multipliers

    def _reduced_residue(self, target):
        """Reduce the coset offset modulo the (echelon) uncertainty
        generators: the result names only the determined components (the
        ambiguous ones are reported in nulldif)."""
        residue = target["offset"].vect
        for gen in target["uncertainty"]:
            pivot = gen.vect.support()[0]
            if residue[pivot] != 0:
                residue = residue + gen.vect
        return residue

    def _chart_dr_cells(self, element, n, s, f):
        """The (target, drinfo, drtarget) values for one chart row. A class
        whose d_r-target lies outside the computable window is still real,
        loaded data: emitted with empty differential columns rather than
        skipped."""
        r = self.d.r
        target = None
        drinfo = ""
        drtarget = ""
        if self.is_computable(n, s - 1, f + r, check_uncertainty=False, source=False):
            try:
                target = self.d(element)
                # The differential is definitely nonzero iff its coset offset
                # + <uncertainty> does not contain 0 -- i.e. the offset is
                # not in the uncertainty span. That is a basis-independent
                # fact even when the exact target is ambiguous, so it must be
                # reported; drtarget names only the determined components.
                residue = self._reduced_residue(target)
                if not residue.is_zero():
                    drinfo = str(r)
                    residue_elt = Element(
                        n, s - 1, f + r, residue,
                        spectral_sequence=element.spectral_sequence,
                    )
                    drtarget = ";".join([str(b) for b in residue_elt.decompose()])
            except (KeyError, ValueError, AttributeError):
                # Differential computation failed or element not decomposable
                target = None
                drinfo = ""
                drtarget = ""
        return target, drinfo, drtarget

    def _chart_hi_cells(self, element, multipliers, row):
        """Sphere hi-product columns for one row: element * hi for each of
        the resolved multipliers (aligned with C2_PRODUCT_MAPS)."""
        for hname, h in zip(C2_PRODUCT_MAPS, multipliers):
            if h is None:
                continue
            try:
                product = element * h
                if hasattr(product, 'decompose'):
                    row[f"{hname}target"] = ";".join(
                        [str(b) for b in product.decompose()])
            except (ValueError, AttributeError, KeyError):
                # Product computation failed or not decomposable
                pass

    def _h0_tower_extension(self, row, n, s, f, names_dict):
        """BANDAID (lexical, not algebraic): extend an established lambda_0
        tower to its top. h0 = lambda_0 climbs the tower (lambda_0^f ->
        lambda_0^{f+1}, and the even stem-(n-1) tower likewise), but the top
        products fall outside the computed relations/multiply window, so the
        column loses its last edge. A tower class' name is a leading index
        followed by all zeros; if the class one filtration up is that name
        with one more trailing zero, draw the structural h0 edge to it."""
        if row["h0target"]:
            return
        tokens = row["label"].split()
        if tokens and all(t == "0" for t in tokens[1:]):
            up = (n, s, f + 1)
            if self.has_elements(*up) and self.dimension[up] == 1:
                up_el = self.page[up][0]
                up_name = names_dict.get(str(up_el), str(up_el))
                if up_name.split() == tokens + ["0"]:
                    row["h0target"] = f"{n}_{s}_{f + 1}"

    def _chart_c2_hi_cells(self, element, row):
        """n=0 column hi-product columns, read from the hi map tables
        (E2_C2_products.csv on E2, induced on later pages) rather than the
        sphere multiplication table the hi cells consult. Deliberately
        .table.get and NOT .apply: a missing entry stays blank (zero product
        and beyond-data-coverage alike), where apply would fabricate and
        cache a false zero image."""
        for hname in C2_PRODUCT_MAPS:
            try:
                image = self.maps[hname].table.get(element)
                if image is not None and hasattr(image, 'decompose'):
                    row[f"{hname}target"] = ";".join(
                        [str(b) for b in image.decompose()])
            except (ValueError, AttributeError, KeyError):
                pass

    def _chart_map_cells(self, element, row):
        """E/H/P/C2 map-image columns for one row."""
        for map_name in standard_map_names(self.has_C2, with_hi=False):
            if map_name in self.maps and hasattr(self.maps[map_name], 'table'):
                try:
                    image = self.maps[map_name].apply(element)
                    if image != self.zero(image.n, image.s, image.f) and hasattr(image, 'decompose'):
                        row[map_name] = ";".join([str(b) for b in image.decompose()])
                    else:
                        row[map_name] = ""
                except (ValueError, AttributeError, KeyError):
                    # Map application failed or image not decomposable
                    row[map_name] = ""

    def write_spheres(self, charts_dir=None):
        """Write the chart CSV {charts_dir}/E{r}_{max_t}.csv: one row per basis
        element in every charted column (spheres, plus the n=0 Lambda(C2)
        column; n=1 is skipped), with columns for the h0-h3 products, the
        E/H/P/C2 map images, the d_r target (drinfo/drtarget), and the
        unresolved differential ambiguity (nulldif)."""
        # Use the names trait instead of loading from JSON file
        names_dict = self.names if self.names else {}
        colnames = [
            "name", "n", "stem", "Adams filtration", "shift", "tautorsion", "h0info", "h0target",
            "h1info", "h1target", "h2info", "h2target", "h3info", "h3target", "E", "H", "P", "C2",
            "label", "angle", "drinfo", "drtarget", "nulldif", "XX",
        ]
        charts_output_dir = charts_dir if charts_dir is not None else DEFAULT_CHARTS_DIR
        os.makedirs(charts_output_dir, exist_ok=True)
        filename = f"{charts_output_dir}/E{self.d.r}_{self.max_t}.csv"
        with open(filename, "w", newline="") as chart_out:
            writer = csv.DictWriter(chart_out, colnames, dialect="excel")
            writer.writeheader()
            for n, s, f in self.all_tridegrees_with_elements():
                if n > MAX_N_FOR_CHART_OUTPUT:
                    continue
                # n=0 is the C2 column (emitted so the downstream generator makes
                # an S0/C2 chart); n=1 is not charted.
                if n == 1:
                    continue
                if (n, s, f) not in self.page:
                    continue
                multipliers = self._chart_hi_multipliers(n + s)
                adams_elements = self.page[n, s, f]
                if hasattr(adams_elements, 'basis'):
                    element_list = adams_elements.basis
                elif hasattr(adams_elements, '__iter__'):
                    element_list = list(adams_elements)
                else:
                    element_list = [adams_elements]
                for element in element_list:
                    target, drinfo, drtarget = self._chart_dr_cells(element, n, s, f)
                    shift = 0
                    try:
                        if hasattr(adams_elements, 'basis'):
                            basis_list = list(adams_elements.basis)
                            if element in basis_list:
                                shift = -len(basis_list) + 2 * basis_list.index(element) + 1
                    except (AttributeError, ValueError, TypeError):
                        # Element not in basis or basis computation failed
                        pass
                    if self.get_lowest_r_uncertainty_for_element(element, n, s, f):
                        # Decompose each uncertainty generator into basis elements
                        uncertainty_list = self.get_lowest_r_uncertainty_for_element(element, n, s, f)
                        nulldif = ";".join([str(b) for unc in uncertainty_list for b in (unc.decompose() if hasattr(unc, 'decompose') else [unc])])
                    elif target is not None:
                        # Decompose each uncertainty generator into basis elements, matching drtarget format
                        nulldif = ";".join([str(b) for unc in target["uncertainty"] for b in (unc.decompose() if hasattr(unc, 'decompose') else [unc])])
                    else:
                        nulldif = ""
                    row = {
                        "name": str(element),
                        "n": n,
                        "stem": getattr(element, 's', s),
                        "Adams filtration": getattr(element, 'f', f),
                        "shift": shift,
                        "tautorsion": 0,
                        "h0info": "",
                        "h0target": "",
                        "h1info": "",
                        "h1target": "",
                        "h2info": "",
                        "h2target": "",
                        "h3info": "",
                        "h3target": "",
                        "E": "",
                        "H": "",
                        "P": "",
                        "C2": "",
                        "label": names_dict.get(str(element), str(element)),
                        "angle": "",
                        "drinfo": drinfo,
                        "drtarget": drtarget,
                        "nulldif": nulldif,
                        "XX": "XX",
                    }
                    self._chart_hi_cells(element, multipliers, row)
                    self._h0_tower_extension(row, n, s, f, names_dict)
                    if n == 0 and 'h0' in self.maps:
                        # The n=0 column's hi products live in the hi map
                        # tables -- without this the C2 chart would show no
                        # hi lines at all.
                        self._chart_c2_hi_cells(element, row)
                    self._chart_map_cells(element, row)
                    writer.writerow(row)




class SpectralSequence:
    """
    Container for multiple spectral sequence pages.

    A spectral sequence consists of pages E_2, E_3, E_4, ... where each
    page is computed from the previous page using differentials.

    This class orchestrates:
    - Page transitions (computing homology)
    - Induced maps and products
    - File I/O and serialization
    """

    def __init__(self, initial_r=2):
        self.pages = {}  # Dictionary mapping r -> SpectralSequencePage
        self.current_r = initial_r

    @property
    def has_C2(self):
        """C2 availability, read off the current page's map set."""
        return self.current_page is not None and self.current_page.has_C2

    @classmethod
    def from_data(cls, prefix, r=2, tot=None, differential_file=None, build_pairs=True):
        """
        Load a spectral sequence from CSV data files.

        This is the primary way to initialize a spectral sequence from input data.
        The method handles:
        - Loading page data (dimensions, products, maps, names)
        - Creating the spectral sequence container
        - Optionally loading known differentials from file

        Known stable/C2 differentials from stable/*.txt are not seeded here;
        check_stable() resolves them during compute().

        Args:
            prefix: Directory or file prefix containing the data (e.g., "E2" or "E2/E2")
            r: The page number to load (default: 2)
            tot: Maximum total degree s+f to load (optional, loads all if None)
            differential_file: Optional file to load differentials from

        Returns:
            SpectralSequence object initialized on the E_r page

        Example:
            # Load E2 page
            ss = SpectralSequence.from_data("E2", r=2, tot=79)

            # Load E2 page and load differentials from JSON file
            ss = SpectralSequence.from_data("E2", r=2, tot=79, differential_file="d2")

            # Load E2 page and load differentials from CSV file
            ss = SpectralSequence.from_data("E2", r=2, tot=79, differential_file="d2-known.csv")
        """
        # Load the spectral sequence page
        if tot is None:
            tot = float('inf')  # Load everything

        page = load_spectral_sequence(prefix, r, tot, build_pairs=build_pairs)

        # Handle differentials
        if differential_file is not None:
            # Load differentials from file (CSV or JSON)
            if differential_file.endswith('.csv'):
                page.load_d_csv(differential_file)
            else:
                page.load_d(differential_file)

        # Create spectral sequence container and add the page
        ss = cls(initial_r=r)
        ss.add_page(r, page)

        return ss

    def add_page(self, r, page):
        """Add a SpectralSequencePage to the sequence"""
        self.pages[r] = page
        page.r = r

    def get_page(self, r):
        """Get the E_r page"""
        return self.pages.get(r)

    def compute(self):
        """Compute differentials on the current page"""
        if self.current_page:
            self.current_page.compute()

    def write_spheres(self, charts_dir=None):
        """Write spheres chart for the current page"""
        if self.current_page:
            self.current_page.write_spheres(charts_dir=charts_dir)

    @property
    def current_page(self):
        """Get the current page (E_{current_r})"""
        return self.pages.get(self.current_r)

    def compute_names(self, r):
        """
        Compute names for E_{r+1} page from E_r page.

        Uses the lifted preimages of basis elements to track names through the differential.

        Args:
            r: The page number to use as source (computes names for E_{r+1})

        Returns:
            Dictionary mapping element strings to their names
        """
        source_page = self.pages.get(r)
        target_page = self.pages.get(r + 1)

        if source_page is None:
            raise ValueError(f"No page found for r={r}")
        if target_page is None:
            raise ValueError(f"No page found for r={r+1}")
        if not source_page.turned_page:
            raise ValueError(f"Must call next_page() on E_{r} before computing names")

        # Use the names from the source page
        current_names = source_page.names
        if not current_names:
            return {}

        next_names_dict = {}

        # Iterate through all tridegrees with elements in the target page
        for x_n, x_s, x_f in target_page.all_tridegrees_with_elements():
            for x in target_page.page[x_n, x_s, x_f]:
                x_preimage = source_page.lift(x)
                factor_names = []
                for factor in x_preimage.decompose():
                    factor_name = current_names.get(str(factor))
                    if factor_name is None:
                        continue
                    factor_names.append(factor_name)
                if factor_names:
                    next_names_dict[str(x)] = " + ".join(factor_names)

        # Write names to JSON file
        next_filename = f"data/E{r + 1}/E{r + 1}_names.json"
        with open(next_filename, "w") as f:
            json.dump(next_names_dict, f, indent=2)
        print(f"Computed names for {len(next_names_dict)} elements -> {next_filename}")

        return next_names_dict

    def compute_induced_map(self, map_name, r):
        """
        Compute induced map for E_{r+1} from E_r.

        The induced map on E_{r+1} is computed by:
        1. Lifting elements from E_{r+1} to E_r
        2. Applying the map on E_r
        3. Quotienting back to E_{r+1}

        Args:
            map_name: Name of the map ('E', 'H', 'P', or 'C2')
            r: The source page number (computes induced map for E_{r+1})

        Returns:
            Dictionary mapping elements in E_{r+1} to their images
        """
        source_page = self.pages.get(r)
        target_page = self.pages.get(r + 1)

        if source_page is None:
            raise ValueError(f"No page found for r={r}")
        if target_page is None:
            raise ValueError(f"No page found for r={r+1}")
        if map_name not in source_page.maps:
            raise ValueError(f"Unknown map: {map_name}")

        map_obj = source_page.maps[map_name]
        local_map = {}

        # Iterate through the domain of the map
        for x_n, x_s, x_f in source_page.map_domain(map_name):
            target_n, target_s, target_f = map_obj.target_degree(x_n, x_s, x_f)
            target_tridegree = (target_n, target_s, target_f)

            if target_tridegree not in target_page.page:
                continue
            if (x_n, x_s, x_f) not in target_page.page:
                continue
            # Compute map matrix on source page
            map_matrix = map_obj.matrix(x_n, x_s, x_f, source_page)
            if map_matrix.ncols() == 0:
                continue

            # For each basis element in target_page at this degree
            for x in target_page.page[x_n, x_s, x_f]:
                x_str = str(x)
                if not x_str or "_" not in x_str or x_str in ["1", "0"] or x_str.count("_") < 2:
                    continue

                # Lift to source page, apply map, quotient to target page
                x_preimage = source_page.lift(x)
                map_preimage_vect = x_preimage.vect * map_matrix
                map_result = Element(
                    target_n, target_s, target_f,
                    map_preimage_vect,
                    spectral_sequence=source_page
                )

                if map_preimage_vect.is_zero():
                    continue

                # Reduce against boundaries in target page
                map_reduced = reduce_against(
                    map_preimage_vect,
                    source_page.turned_page[target_tridegree].B
                )

                # Check that result is a cycle (d(result) = 0)
                try:
                    source_r = source_page.d.r
                    if not source_page.d(map_result)["offset"] == source_page.zero(target_n, target_s - 1, target_f + source_r):
                        continue
                except (KeyError, ValueError, AttributeError):
                    # Differential computation failed, skip this element
                    continue

                # Check if target_tridegree has elements
                if target_tridegree not in target_page.page or target_page.dimension[target_tridegree] == 0:
                    continue
                elif not map_reduced.is_zero():
                    local_map[x] = source_page.quotient(
                        Element(target_n, target_s, target_f, map_reduced, spectral_sequence=source_page)
                    )

        return local_map

    def compute_induced_products(self, r):
        """
        Compute induced products for E_{r+1} from E_r.

        The induced product on E_{r+1} is computed by:
        1. Lifting elements from E_{r+1} to E_r
        2. Computing product on E_r
        3. Quotienting back to E_{r+1}

        Args:
            r: The source page number (computes induced products for E_{r+1})

        Returns:
            Dictionary mapping (x, y) pairs to their product x*y
        """
        source_page = self.pages.get(r)
        target_page = self.pages.get(r + 1)

        if source_page is None:
            raise ValueError(f"No page found for r={r}")
        if target_page is None:
            raise ValueError(f"No page found for r={r+1}")

        local_products = {}

        # Iterate through pairs from the target page
        for x_coords, y_coords_list in target_page.pairs.items():
            x_n, x_s, x_f = x_coords
            for y_coords in y_coords_list:
                y_n, y_s, y_f = y_coords
                xy_n, xy_s, xy_f = x_n, x_s + y_s, x_f + y_f
                for x in target_page.page[x_n, x_s, x_f]:
                    x_preimage = source_page.lift(x)
                    for y in target_page.page[y_n, y_s, y_f]:
                        y_preimage = source_page.lift(y)

                        # Compute product on source page
                        xy_preimage = x_preimage * y_preimage

                        if xy_preimage == source_page.zero(xy_n, xy_s, xy_f) or target_page.dimension[xy_n, xy_s, xy_f] == 0:
                            local_products[x, y] = target_page.zero(xy_n, xy_s, xy_f)
                        else:                         # Reduce against boundaries
                            xy_reduced = source_page.in_next_basis(xy_preimage)
                            try:
                                xy = source_page.quotient(
                                    Element(xy_n, xy_s, xy_f, xy_reduced, spectral_sequence=source_page)
                                )
                                local_products[x, y] = xy
                            except Exception:
                                # The quotient can't be formed because the target
                                # tridegree has unresolved differentials, so its
                                # next-page basis is only partially determined:
                                # the product landing there can't be computed
                                # exactly, so leave it absent. (Expected; same
                                # handling as compute_induced_hi_products.)
                                continue

        return local_products

    def compute_induced_hi_products(self, r):
        """
        Compute the induced h0/h1/h2/h3 (filtration-1) products for E_{r+1}.

        These are the module-structure products the charts display: the h0, h1,
        h2, h3 products on every class (including the (n,0,0) bottom class) and,
        as a consequence, the full h0 tower in the 0-stem. They are computed directly
        (lift -> multiply on E_r -> quotient), bypassing the Leibniz-oriented
        filters in build_pairs -- the (n,0,0) skip and the E-map (y_n-1)
        requirement -- which otherwise drop exactly these products from
        compute_induced_products.

        Returns a {(x, h): product} dict to merge into the page's product table.
        Pairs already computed via compute_induced_products are skipped, so this
        never overrides them.
        """
        source_page = self.pages.get(r)
        target_page = self.pages.get(r + 1)
        if source_page is None or target_page is None:
            return {}

        # (stem, filtration) of the h0, h1, h2, h3 multipliers, as looked up by
        # the chart writer at (n + s, stem, 1) for a class at (n, s, f).
        hi_bidegrees = SpectralSequencePage.HI_MULTIPLIER_BIDEGREES
        local_products = {}
        for (n, s, f) in list(target_page.page.keys()):
            for hs, hf in hi_bidegrees:
                h_coords = (n + s, hs, hf)
                if not target_page.has_elements(*h_coords):
                    continue
                xy_n, xy_s, xy_f = n, s + hs, f + hf
                if target_page.dimension[xy_n, xy_s, xy_f] == 0:
                    continue
                for h in target_page.page[h_coords]:
                    h_preimage = source_page.lift(h)
                    for x in target_page.page[n, s, f]:
                        if (x, h) in target_page.products:
                            continue  # already computed via the Leibniz pairs
                        xy_preimage = source_page.lift(x) * h_preimage
                        if xy_preimage == source_page.zero(xy_n, xy_s, xy_f):
                            local_products[x, h] = target_page.zero(xy_n, xy_s, xy_f)
                            continue
                        xy_reduced = source_page.in_next_basis(xy_preimage)
                        try:
                            local_products[x, h] = source_page.quotient(
                                Element(xy_n, xy_s, xy_f, xy_reduced, spectral_sequence=source_page)
                            )
                        except Exception:
                            # The quotient can't be formed (partial coverage at
                            # a data boundary): leave this product absent.
                            continue
        return local_products

    def next_page(self, save=True):
        """
        Compute E_{r+1} from E_r and optionally save to disk.

        This orchestrates the full workflow:
        1. Compute homology H(E_r, d_r) = E_{r+1}
        2. Compute induced maps on E_{r+1}
        3. Compute induced products on E_{r+1}
        4. Build structural caches
        5. Optionally save to disk

        Args:
            save: If True, save results to directory E_{r+1}/

        Returns:
            The next spectral sequence page E_{r+1}
        """
        if not self.current_page:
            raise ValueError(f"No current page at r={self.current_r}")

        current = self.current_page

        # Step 1: Mathematical computation - turn the page
        print(f"Computing E_{self.current_r + 1} from E_{self.current_r}...")
        next_page = current.next_page()

        # Step 2: Build structural information needed for map computation
        next_page.build_pairs()

        # Step 3: Store the page in the sequence (before computing induced structures)
        self.add_page(self.current_r + 1, next_page)

        # Step 4: Compute induced maps (conditionally for C2)
        print(f"Computing induced maps...")
        for map_name in standard_map_names(self.has_C2):
            next_page.maps[map_name].table = self.compute_induced_map(map_name, self.current_r)
            # The induced table can only be complete where the source page's
            # data was; carry the coverage bound forward so naturality on the
            # next page skips the same false-zero region.
            next_page.maps[map_name].complete_through = \
                current.maps[map_name].complete_through

        # Step 5: Compute induced products
        print(f"Computing induced products...")
        next_page.products = self.compute_induced_products(self.current_r)
        # Add the h0-h3 module products (h0 tower + (n,0,0) bottom class) that
        # the Leibniz-filtered pairs drop, so the charts display them.
        next_page.products.update(self.compute_induced_hi_products(self.current_r))

        # Update current_r
        self.current_r += 1

        # Step 6: Optional file I/O
        if save:
            directory = f"data/E{self.current_r}"
            print(f"Saving to {directory}/...")
            next_page.save_to_directory(directory)
            # Compute names and store them in the next page's names trait
            next_page.names = self.compute_names(self.current_r - 1)

        print(f"E_{self.current_r} computation complete.")
        return next_page

    def __getitem__(self, r):
        """Access E_r page via ss[r]"""
        return self.get_page(r)

    def __repr__(self):
        page_list = ", ".join(f"E_{r}" for r in sorted(self.pages.keys()))
        return f"SpectralSequence(pages=[{page_list}], current=E_{self.current_r})"

