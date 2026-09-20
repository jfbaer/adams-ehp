"""
Differential management for spectral sequences.

This module provides the DifferentialsPage class, which manages the differential
d_r: E_r^{n,s,f} → E_r^{n,s-1,f+r} as an affine space of matrices.

Conventions:
- Tridegree (n,s,f): n=grade, s=stem, f=Adams filtration; page number r.
- Differential matrices are SOURCE x TARGET over GF(2): row i is the image
  of the i-th source basis element, applied as element.vect * M.
  (The exported known-differential CSVs use the transposed indexing:
  row = target index, col = source index; see KNOWN_DIFF_CSVS.)

Key Concepts:
- Differentials are represented as AffineMatrixSubspace objects
- Each differential d[n,s,f] is constrained by:
  * Naturality with respect to maps (E, H, P, C2)
  * Leibniz rule d(xy) = d(x)y + (-1)^|x| x·d(y)
  * d² = 0
  * Stability for high connectivity

- Constraints narrow the affine space until d[n,s,f] is "forced" (unique)
- When not fully determined, differentials have "uncertainty"

Classes:
- DifferentialsPage: Dictionary mapping (n,s,f) → AffineMatrixSubspace
  * Inherits from key_defaultdict for automatic initialization
  * Tracks constraint provenance via "why" chains

Main Operations:
- d(element): Apply differential to an element
- turn_page_with_uncertainty(): Compute homology H(E_r, d_r) = E_{r+1}
- check_stable(): Resolve stable-range (n > s+1) and C2 (n=0) differentials
  from the entry-level input CSVs (stable/stable_sphere_diffs.csv,
  stable/c2_diffs.csv; provenance in stable/README.md)

Proof chains are visualized as flow charts by why.py (write_why_graph).

The module supports:
- Incremental constraint application with proof tracking
- Partial differentials with uncertainty quantification
- Visualization of deduction chains for debugging
- Rollback to previous constraint states
"""
import ast

from sage.all import GF, matrix, span, vector
from sage.geometry.hyperplane_arrangement.affine_subspace import AffineSubspace
from sage.all import VectorSpace

from lib import (
    Element,
    TurnedBidegree,
    AffineMatrixSubspace,
    C2_DATA_COMPLETE_TOT,
    ContradictionError,
    key_defaultdict,
    create_differential_coset,
    relative_row_echelon,
)

# Special n-values for stability checking
C2_N_VALUE = 0        # n-value for C2 differential processing

# The not-a-boundary machinery enumerates 2^(ncols-1) functionals of the
# target space; past this many columns that enumeration (here 2^16) is too
# large, so nb / certainly_not_boundary decline rather than attempt it.
MAX_NB_TARGET_NCOLS = 17


def _subspace_contains(big, small):
    """True if the affine subspace `big` contains `small` (same linear part up
    to inclusion, and their offsets differ by a vector of big's linear part)."""
    return (
        small.linear_part().is_subspace(big.linear_part())
        and (small.point() - big.point()) in big.linear_part()
    )


def _not_in_image_pieces(diff, y):
    """The exact solution set of "y is not in the row space of M", over the affine
    matrix space `diff`, as a pruned list of (phi, AffineSubspace) pairs.

    {M : y not in im M} is NOT a flat: it is the union, over the 2^(ncols-1)
    functionals phi with phi(y) = 1, of the linear flats {M : M phi^T = 0}
    (row space contained in ker phi).  Each returned piece is the intersection
    of diff.subspace with one such flat, tagged by its functional; empty
    intersections are dropped and any piece contained in another is absorbed
    (keeping the containing piece's phi).
    """
    ambient = diff.ambient
    target = VectorSpace(GF(2), diff.ncols)
    # {phi : phi . y = 1} = phi0 + y-perp
    phi0 = target.basis()[y.support()[0]]
    y_perp = matrix(GF(2), [y]).right_kernel()
    pieces = []
    for z in y_perp:
        phi = phi0 + z
        # {M : M phi^T = 0}: one linear condition per source row i on the
        # flattened entries w[i*ncols + j] (same convention as _impose_entry)
        conditions = matrix(GF(2), diff.nrows, ambient.dimension())
        for i in range(diff.nrows):
            for j in phi.support():
                conditions[i, i * diff.ncols + j] = 1
        piece = diff.subspace.intersection(
            AffineSubspace(ambient.zero(), conditions.right_kernel())
        )
        if piece is not None:
            pieces.append((phi, piece))

    pruned = []
    for phi, piece in pieces:
        if any(_subspace_contains(other, piece) for _, other in pruned):
            continue
        pruned = [(p, other) for p, other in pruned
                  if not _subspace_contains(piece, other)]
        pruned.append((phi, piece))
    return pruned


def _prune_pieces(pieces):
    """Prune a list of AffineSubspaces: drop any piece contained in another"""
    pruned = []
    for piece in pieces:
        if any(_subspace_contains(other, piece) for other in pruned):
            continue
        pruned = [other for other in pruned if not _subspace_contains(piece, other)]
        pruned.append(piece)
    return pruned

class DifferentialsPage(key_defaultdict):
    """The d_r differentials of one page: a key_defaultdict mapping each tridegree (n, s, f)
    to the AffineMatrixSubspace of possible differentials there, with a global counter for
    tracking proof steps"""
    def __init__(self, r=2, dimension_dict=None):
        if dimension_dict is None:
            raise ValueError("dimension_dict is required for DifferentialsPage")
        self.dimension_dict = dimension_dict
        super().__init__(lambda nsf: create_differential_coset(nsf, r, dimension_dict))
        self.r = r
        self.counter = 0

        # Pre-populate all keys from dimension_dict to avoid lazy initialization
        # Create a static list to avoid RuntimeError: dictionary changed size during iteration
        existing_keys = list(dimension_dict.keys())
        for tridegree in existing_keys:
            if dimension_dict[tridegree] > 0:  # Only create entries for non-zero dimensions
                self[tridegree] = create_differential_coset(tridegree, r, dimension_dict)

    def __call__(self, element):
        """Apply this page's differential to an element.  Returns a dict:
        "offset" is d(element) under the current representative matrix, and
        "uncertainty" is a list of Elements spanning how the image can still
        vary over the remaining candidates in the constraint space."""
        n, s, f, r = element.n, element.s, element.f, self.r
        diffspace = self[n, s, f]

        v_vect = element.vect * diffspace.as_matrix(diffspace.v)
        v = Element(n, s - 1, f + r, v_vect, spectral_sequence=element.spectral_sequence)

        # Compute uncertainty generators
        basis = diffspace.subspace.linear_part().basis()
        gens = [element.vect * diffspace.as_matrix(gen) for gen in basis]
        gens = span(gens, base_ring=GF(2)).basis()
        gens = list(map(lambda vec: Element(n, s - 1, f + r, vec, spectral_sequence=element.spectral_sequence), gens))

        return {"offset": v, "uncertainty": gens}

    def most_complex(self):
        """
        Find the (n, s, f) with the longest proof chain (most recorded `why` steps).

        Returns:
            tuple: (n, s, f) with the longest `why` list.
        """
        max_key = None
        max_why_len = -1  # Start with a value lower than any possible length

        for key in self.keys():
            if len(self[key].why) > max_why_len:
                max_why_len = len(self[key].why)
                max_key = key

        return max_key

    def __copy__(self):
        """Shallow copy: the new page SHARES this page's AffineMatrixSubspace
        entries, so an in-place restriction on either copy is visible to
        both."""
        new_d = DifferentialsPage(r=self.r, dimension_dict=self.dimension_dict)
        for bidegree in self:
            new_d[bidegree] = self[bidegree]
        new_d.counter = self.counter
        return new_d

    def get_step(self, step_number=0):
        """Return the proof-step record with the given counter, together with its bidegree; non-positive step numbers count back from the current counter"""
        if step_number <= 0:
            step_number += self.counter
        for bidegree in self:
            for w in self[bidegree].why:
                if w["counter"] == step_number:
                    return {"bidegree": bidegree, **w}
        raise IndexError(f"Step number {step_number} not found")

    def set_differential(self, bidegree, mat):
        """Force the differential at a bidegree to the single matrix `mat`, incrementing the proof counter (rolled back on error)"""
        counter = self.counter
        try:
            self.counter += 1
            self[tuple(bidegree)].set_differential(mat, self.counter)
        except Exception:
            # Rollback counter on any error before re-raising
            self.counter = counter
            raise

    def load_from_csv(self, csv_file):
        """
        Load differentials from a CSV file.

        The CSV file should have columns: n, s, f, matrix
        where matrix is a string representation of a 2D list (e.g., "[[0, 1], [1, 0]]")

        Args:
            csv_file: Path to the CSV file
        """
        import csv

        print(f"Loading differentials from {csv_file}...")
        loaded_count = 0
        skipped_count = 0

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')  # Tab-delimited based on the sample
                for line in reader:
                    try:
                        n = int(line['n'])
                        s = int(line['s'])
                        f = int(line['f'])
                        matrix_str = line['matrix']

                        # Parse the matrix string (e.g., "[[0, 1], [1, 0]]")
                        matrix_list = ast.literal_eval(matrix_str)

                        # Determine dimensions
                        if not matrix_list:
                            continue
                        nrows = len(matrix_list)
                        ncols = len(matrix_list[0]) if matrix_list else 0

                        # Create the matrix
                        mat = matrix(GF(2), nrows, ncols, matrix_list)

                        # Set the differential
                        self.set_differential((n, s, f), mat)
                        loaded_count += 1

                    except (KeyError, ValueError, SyntaxError, TypeError):
                        # A malformed row (missing column, unparseable matrix):
                        # skip it but keep a count so the loss is not silent.
                        skipped_count += 1
                        continue

            print(f"  Loaded {loaded_count} differentials from {csv_file}"
                  + (f" ({skipped_count} rows skipped as malformed)"
                     if skipped_count else ""))

        except FileNotFoundError:
            print(f"  Warning: {csv_file} not found")

    def set_element_differential(self, element, target_differential, adams_page):
        """
        Force the differential of a specific element to be a specific value.

        Args:
            element: Element in domain space
            target_differential: Element in codomain space (desired d(element))
            adams_page: The adams_page dict to determine element ordering
        """
        bidegree = (element.n, element.s, element.f)

        # Find the index of the element in the Adams page basis
        if bidegree not in adams_page:
            raise ValueError(f"Bidegree {bidegree} not found in Adams page")

        basis_elements = adams_page[bidegree]
        try:
            element_index = basis_elements.index(element)
        except ValueError:
            # Element is not a single basis element (likely a linear combination)
            # Skip setting differential for linear combinations
            return

        # Call the underlying set_element_differential method
        counter = self.counter
        try:
            self.counter += 1
            self[bidegree].set_element_differential(element, target_differential, element_index, self.counter)
        except Exception:
            # Rollback counter on any error before re-raising
            self.counter = counter
            raise

    def nb(self, element, label="nb", reason=None, verbose=True):
        """Force the fact that `element` is NOT a boundary on this page: no
        remaining candidate for the differential targeting its tridegree may
        contain it in its image.  `label`/`reason` override the recorded
        proof-step provenance (see AffineMatrixSubspace.restrict);
        `verbose=False` suppresses the refusal certificate and cap messages
        (for bulk callers that count refusals instead).

        Looks up the incoming differential d[n, s+1, f-r] and intersects its
        affine space with the exact solution set of "element.vect not in the
        row space".  That set is a union of flats (one per functional phi with
        phi(element.vect) = 1, cut out by M phi^T = 0), so the fact can only be
        imposed when the pruned union collapses to a single flat.

        Returns True when the space shrank, False when the fact already holds
        for every remaining candidate (no-op), and None when it could not be
        imposed (the survivors form a genuine multi-flat union, reported as a
        per-phi certificate, or the target dimension exceeds the functional-
        enumeration cap).  Raises ContradictionError when every remaining
        candidate hits the element.
        """
        n, s, f = element.n, element.s, element.f
        source = (n, s + 1, f - self.r)
        diff = self[source]
        y = element.vect
        if len(y) != diff.ncols:
            raise ValueError(
                f"nb: element at {(n, s, f)} has length {len(y)} but the "
                f"incoming differential d{self.r}{source} targets dimension "
                f"{diff.ncols}"
            )
        if y.is_zero():
            raise ValueError("nb: zero is in the image of every differential")
        if diff.nrows == 0 or diff.is_cycle():
            return False
        if diff.is_forced:
            if y in diff.as_matrix(diff.v).row_space():
                raise ContradictionError(
                    f"nb: element {list(y)} at {(n, s, f)} is in the image of "
                    f"the forced differential d{self.r}{source}",
                    left=diff,
                )
            return False
        if diff.ncols > MAX_NB_TARGET_NCOLS:
            if verbose:
                print(f"  nb: skipping d{self.r}{source}: target dimension "
                      f"{diff.ncols} exceeds the 2^16-functional cap")
            return None

        pieces = _not_in_image_pieces(diff, y)
        if not pieces:
            raise ContradictionError(
                f"nb: every remaining candidate for d{self.r}{source} contains "
                f"{list(y)} at {(n, s, f)} in its image",
                left=diff,
            )
        if len(pieces) == 1:
            piece = pieces[0][1]
            # piece is a subset of the current space, so equal dimensions mean
            # the fact already holds for every remaining candidate
            if piece.dimension() == diff.dimension():
                return False
            counter = self.counter
            try:
                self.counter += 1
                diff.restrict(piece, self.counter, label=label, reason=reason)
            except Exception:
                self.counter = counter
                raise
            return True
        # A genuine union of flats is not representable as one affine space;
        # refuse rather than impose a lossy hull.
        if verbose:
            print(f"  nb: cannot impose 'not a boundary' for {list(y)} at "
                  f"{(n, s, f)}: the surviving candidates for d{self.r}{source} "
                  f"form a union of {len(pieces)} flats, not a single affine "
                  f"space; leaving it open")
            for phi, piece in pieces:
                basis = [list(b) for b in piece.linear_part().basis()]
                print(f"    phi={list(phi)}: point={list(piece.point())} "
                      f"dim={piece.dimension()} basis={basis}")
        return None

    def nb_multi(self, elements, label="nb", reason=None, verbose=True,
                 max_pieces=64):
        """Force that NONE of `elements` (all in one tridegree) is a boundary,
        by intersecting the per-element unions of avoiding-flats.  Combining
        facts often collapses to a single flat even when each element alone
        does not: excluding both e1 and e2 from a fully open 1x2-per-row
        space leaves exactly the flat of matrices with rows in {0, e1+e2},
        while each exclusion separately is a genuine 2-flat union.

        Zero elements are ignored.  Returns True when the space shrank,
        False when the combined fact already held (or nothing to do), None
        when the combined survivors are still a multi-flat union (or a cap
        was hit).  Raises ContradictionError when no candidate avoids all
        the elements."""
        elements = [e for e in elements if not e.vect.is_zero()]
        if not elements:
            return False
        n, s, f = elements[0].n, elements[0].s, elements[0].f
        source = (n, s + 1, f - self.r)
        diff = self[source]
        for e in elements:
            if (e.n, e.s, e.f) != (n, s, f) or len(e.vect) != diff.ncols:
                raise ValueError(
                    f"nb_multi: element {e} does not live in {(n, s, f)} "
                    f"with dimension {diff.ncols}")
        if diff.nrows == 0 or diff.is_cycle():
            return False
        if diff.is_forced:
            rows = diff.as_matrix(diff.v).row_space()
            for e in elements:
                if e.vect in rows:
                    raise ContradictionError(
                        f"nb_multi: element {list(e.vect)} at {(n, s, f)} is "
                        f"in the image of the forced differential "
                        f"d{self.r}{source}",
                        left=diff,
                    )
            return False
        if diff.ncols > MAX_NB_TARGET_NCOLS:
            return None

        combined = None
        for e in elements:
            pieces = [p for _, p in _not_in_image_pieces(diff, e.vect)]
            if combined is None:
                combined = pieces
            else:
                combined = _prune_pieces([
                    c for a in combined for b in pieces
                    if (c := a.intersection(b)) is not None
                ])
            if not combined:
                raise ContradictionError(
                    f"nb_multi: every remaining candidate for "
                    f"d{self.r}{source} hits one of the excluded elements "
                    f"at {(n, s, f)}",
                    left=diff,
                )
            if len(combined) > max_pieces:
                if verbose:
                    print(f"  nb_multi: piece cap ({max_pieces}) exceeded at "
                          f"d{self.r}{source}; leaving it open")
                return None
        if len(combined) == 1:
            piece = combined[0]
            if piece.dimension() == diff.dimension():
                return False
            counter = self.counter
            try:
                self.counter += 1
                diff.restrict(piece, self.counter, label=label, reason=reason)
            except Exception:
                self.counter = counter
                raise
            return True
        if verbose:
            print(f"  nb_multi: {len(elements)} facts at {(n, s, f)} still "
                  f"leave a union of {len(combined)} flats at "
                  f"d{self.r}{source}; leaving it open")
        return None

    def certainly_not_boundary(self, element):
        """Return True when NO remaining candidate for the differential
        targeting `element`'s tridegree contains it in its image -- i.e. the
        element is certainly not a boundary on this page, whatever the still-
        open differentials turn out to be.  Conservative: False whenever
        certainty cannot be established (zero element, basis-length drift, or
        a target dimension past the functional-enumeration cap)."""
        n, s, f = element.n, element.s, element.f
        diff = self[n, s + 1, f - self.r]
        y = element.vect
        if y.is_zero() or len(y) != diff.ncols:
            return False
        if diff.nrows == 0 or diff.is_cycle():
            return True
        if diff.is_forced:
            return y not in diff.as_matrix(diff.v).row_space()
        if diff.ncols > MAX_NB_TARGET_NCOLS:
            return False
        pieces = _not_in_image_pieces(diff, y)
        return len(pieces) == 1 and pieces[0][1].dimension() == diff.dimension()

    def force_cycle(self, element, label="cycle", reason=None):
        """Force d(element) = 0 for a single element (possibly a linear
        combination of basis vectors, which set_element_differential silently
        skips): intersect the affine space at the element's tridegree with the
        linear flat {M : element.vect . M = 0}.

        Returns True when the space shrank and False when the constraint
        already held for every candidate (no proof step recorded).  Raises
        ContradictionError when no candidate satisfies it.  `label`/`reason`
        override the recorded proof-step provenance."""
        n, s, f = element.n, element.s, element.f
        diff = self[n, s, f]
        y = element.vect
        if len(y) != diff.nrows:
            raise ValueError(
                f"force_cycle: element at {(n, s, f)} has length {len(y)} but "
                f"d{self.r}{(n, s, f)} has source dimension {diff.nrows}"
            )
        if y.is_zero() or diff.ncols == 0:
            return False
        if (y * diff.as_matrix(diff.v)).is_zero() and all(
            (y * diff.as_matrix(g)).is_zero()
            for g in diff.subspace.linear_part().basis()
        ):
            return False
        # {M : y.M = 0}: one linear condition per target column j on the
        # flattened entries w[i*ncols + j] (same convention as _impose_entry)
        conditions = matrix(GF(2), diff.ncols, diff.ambient.dimension())
        for j in range(diff.ncols):
            for i in y.support():
                conditions[j, i * diff.ncols + j] = 1
        flat = AffineSubspace(diff.ambient.zero(), conditions.right_kernel())
        counter = self.counter
        try:
            self.counter += 1
            diff.restrict(flat, self.counter, label=label, reason=reason)
        except Exception:
            self.counter = counter
            raise
        return True

    def _turn_one(self, nsf, spectral_sequence):
        """Compute the TurnedBidegree at one tridegree from the CURRENT
        representative differentials (kernel of the outgoing rep modulo the
        row space of the incoming rep); returns None when d^2 != 0 there."""
        n, s, f = nsf
        d1 = self[n, s + 1, f - self.r].as_matrix(self[n, s + 1, f - self.r].v)
        d2 = self[n, s, f].as_matrix(self[n, s, f].v)
        Z = d2.kernel()
        B = d1.row_space()
        if not B.is_subspace(Z):
            print(f"Warning: d^2 != 0 at tridegree ({n}, {s}, {f}), skipping")
            return None
        H = relative_row_echelon(B, Z)
        new_vector_space = VectorSpace(GF(2), H.dimension())
        elements = [
            Element(n, s, f, bv, spectral_sequence=spectral_sequence)
            for bv in new_vector_space.basis()
        ]
        return TurnedBidegree(
            elements,
            H.hom(matrix.identity(H.dimension()), new_vector_space),
            B,
        )

    def turn_page_with_uncertainty(self, AdamsPage, spectral_sequence):
        """Compute homology at each tridegree using the representative differentials (even where they are uncertain), returning a dict of TurnedBidegree for the next page; tridegrees where d^2 != 0 are skipped with a warning"""
        next_page = {}
        for n, s, f in AdamsPage:
            turned = self._turn_one((n, s, f), spectral_sequence)
            if turned is None:
                continue
            if len(AdamsPage[n, s, f]) != 0:
                next_page[n, s, f] = turned
        return next_page

    @staticmethod
    def _get_counter(why_list):
        """Helper function for `from_json`"""
        try:
            return why_list[-1]["counter"]
        except IndexError:
            return 0

    @staticmethod
    def from_json(obj, dimension_dict=None, **kwargs):
        """Reconstruct a DifferentialsPage from JSON, optionally overriding the saved matrix dimensions with `dimension_dict`, and restore the proof counter from the recorded reasons"""
        d = DifferentialsPage(int(obj["r"]), dimension_dict=dimension_dict)
        empty = True
        for key in obj:
            if key == "r":
                continue
            else:
                if dimension_dict is not None:
                    # Use the provided dimension dictionary to override saved dimensions
                    tridegree = ast.literal_eval(key)
                    n, s, f = tridegree
                    r = int(obj["r"])
                    saved_data = obj[key]
                    source_dim = dimension_dict[n, s, f]
                    target_dim = dimension_dict[n, s - 1, f + r]

                    # The reconciled dimension_dict can disagree with an entry's
                    # own saved shape when the cache spans a wider degree range
                    # than is internally self-consistent (dims[n,s-1,f+r] is
                    # written both as this entry's ncols and as the neighboring
                    # entry's nrows). When that happens the saved v/linear_part
                    # no longer fit the reconciled ambient; replay the entry at
                    # its own saved shape instead.
                    if source_dim * target_dim != len(saved_data["v"]):
                        source_dim = saved_data["nrows"]
                        target_dim = saved_data["ncols"]

                    # Create new AffineMatrixSubspace with correct dimensions
                    reconstructed = AffineMatrixSubspace(source_dim, target_dim)
                    reconstructed.v = vector(GF(2), saved_data["v"])
                    reconstructed.subspace = AffineSubspace(
                        vector(GF(2), saved_data["v"]),
                        reconstructed.ambient.span(saved_data["linear_part"])
                    )
                    reconstructed.restore_why(saved_data["why"])
                    reconstructed.is_forced = reconstructed.subspace.dimension() == 0
                    d[tridegree] = reconstructed
                else:
                    d[ast.literal_eval(key)] = AffineMatrixSubspace.from_json(obj[key])
                empty = False
        if not empty:
            d.counter = DifferentialsPage._get_counter(
                max((d[k].why for k in d), key=DifferentialsPage._get_counter)
            )
        return d

    def load_from_file(self, filename):
        """
        Load differential information from a JSON file into this existing DifferentialsPage.
        Only loads keys that exist in the file, preserving existing data.

        Every loaded entry is reconciled with THIS session's dimensions: the
        invariant d[n,s,f] : dim(n,s,f) x dim(n,s-1,f+r) must hold, but a
        cache computed at a larger degree range stores larger shapes for
        tridegrees whose differential target lies outside the current polygon.
        Keeping the saved shape breaks every consumer that compares against
        session-shaped zeros (e.g. the induced-map cycle check silently drops
        live map entries, which then read as false zero maps and produce
        naturality contradictions). Entries are therefore projected onto the
        session's block via AffineMatrixSubspace.restricted_to -- exactly the
        state a fresh truncated computation would reach. Entries SMALLER than
        the session's shape (cache from a smaller range) are skipped, leaving
        the unconstrained default, since padding would assert d = 0 on classes
        the cache never saw.

        Returns the number of entries that had to be reshaped.

        Args:
            filename (str): Path to the JSON file containing differential data
        """
        import json
        import ast

        with open(filename, "r") as f:
            obj = json.load(f)

        # Update r if it exists in the file
        if "r" in obj:
            self.r = int(obj["r"])

        # Load differential data for each key that exists in the file
        truncated = 0
        skipped_smaller = 0
        for key in obj:
            if key == "r":
                continue
            bidegree = ast.literal_eval(key)
            n, s, f = bidegree
            entry = AffineMatrixSubspace.from_json(obj[key])
            src = self.dimension_dict[n, s, f]
            tgt = self.dimension_dict[n, s - 1, f + self.r]
            if (entry.nrows, entry.ncols) != (src, tgt):
                if src > entry.nrows or tgt > entry.ncols:
                    skipped_smaller += 1  # cache has less data here than this run
                    continue
                entry = entry.restricted_to(src, tgt)
                truncated += 1
            self[bidegree] = entry

        # Update counter to be consistent with loaded data
        if any(key != "r" for key in obj):
            max_counter = 0
            for bidegree in self:
                if self[bidegree].why:
                    bidegree_max = DifferentialsPage._get_counter(self[bidegree].why)
                    max_counter = max(max_counter, bidegree_max)
            self.counter = max_counter

        if truncated:
            print(f"  reshaped {truncated} loaded differentials to this "
                  f"session's dimensions (cache computed at a larger range)")
        if skipped_smaller:
            print(f"  skipped {skipped_smaller} saved d{self.r} entries with "
                  f"less data than this session (cache from a smaller range)")
        return truncated

    # Entry-level known-differential CSVs (stable/, provenance documented in
    # stable/README.md). 7 columns
    # r,n,s,f,row,col,value with row = TARGET index and col = SOURCE index in
    # the exporting session's basis; our matrices are source x target, so an
    # entry (row, col, v) pins M[col, row] = v. Explicit zeros are information.
    # Tridegrees absent from the file stay OPEN (unlike the txt tables, where
    # absence forces the whole matrix to zero): only what the session actually
    # determined is imposed. Multi-dimensional C2 spots are only meaningful
    # against the same C2 dataset the session was loaded with.
    KNOWN_DIFF_CSVS = ("stable_sphere_diffs.csv", "c2_diffs.csv")

    def _known_diff_entries(self):
        """Load (once) the exported known-differential CSVs for this page's r.

        Returns (stable_table, c2_table), each {(s, f): [(target_idx,
        source_idx, value)]}, or None when the CSVs are absent or hold no rows
        for this r (fall back to the txt tables; r = 6..8 always land there)."""
        if hasattr(self, "_known_csv"):
            return self._known_csv
        import csv as csv_mod
        import os
        tables = []
        for name in self.KNOWN_DIFF_CSVS:
            path = os.path.join("stable", name) if os.path.isdir("stable") else name
            table = {}
            if os.path.exists(path):
                with open(path) as fh:
                    for rec in csv_mod.DictReader(fh):
                        if int(rec["r"]) != self.r:
                            continue
                        key = (int(rec["s"]), int(rec["f"]))
                        table.setdefault(key, []).append(
                            (int(rec["row"]), int(rec["col"]), int(rec["value"]))
                        )
            tables.append(table)
        self._known_csv = tuple(tables) if any(tables) else None
        return self._known_csv

    def _impose_entry(self, tridegree, target_idx, source_idx, value, label=""):
        """Pin the single matrix entry M[source_idx, target_idx] = value at a
        tridegree; returns True if the constraint space actually shrank, None
        if the entry indices exceed this dataset's dimensions (data edge).
        `label` names the assumed input in the recorded proof reason ("stable",
        "C2", "unstable", "spurious") so why graphs can terminate there instead
        of showing an unexplained act-of-God constraint."""
        diff = self[tridegree]
        if source_idx >= diff.nrows or target_idx >= diff.ncols:
            return None
        flat = source_idx * diff.ncols + target_idx
        determined = all(
            b[flat] == 0 for b in diff.subspace.linear_part().basis()
        )
        if determined and diff.v[flat] == value:
            return False
        # A conflicting determined value falls through: the intersection below
        # is empty and restrict() raises the standard ContradictionError.
        e = diff.ambient.basis()
        constraint = AffineSubspace(
            e[flat] * value,
            diff.ambient.span([e[k] for k in range(len(e)) if k != flat]),
        )
        counter = self.counter
        try:
            self.counter += 1
            diff.restrict(constraint, self.counter, label=label)
        except Exception:
            self.counter = counter
            raise
        return True

    def apply_entry_list(self, entries, label=""):
        """Impose a list of entry-level differentials on this page at its r.

        `entries` is an iterable of (n, s, f, row, col, value); each is applied
        via `_impose_entry` (row = target index, col = source index). Used for
        the hand-written Contradiction3/Spurious4 lists; `label` records which
        list the entries came from in the proof reasons. Entries whose tridegree
        is absent from this page, or whose row/col exceed its dimensions, are
        skipped with a warning — a drifted basis on a turned page must never
        be misapplied against the wrong indices. Returns the number actually
        imposed."""
        applied = 0
        skipped = 0
        for (n, s, f, row, col, value) in entries:
            if (n, s, f) not in self:
                skipped += 1
                continue
            result = self._impose_entry((n, s, f), row, col, value, label=label)
            if result is None:
                skipped += 1
            elif result:
                applied += 1
        if skipped:
            print(f"  {skipped} list entries skipped (tridegree absent or "
                  f"row/col out of range on this page)")
        return applied

    def _check_stable_csv(self, stable_table, c2_table):
        """Impose every exported entry that applies to this page: stable rows
        (recorded at the representative n = s + 2) to every sphere with
        n > s + 1, C2 rows to the n == 0 column; returns True on any change"""
        count = 0
        skipped = 0
        for (n, s, f) in list(self.keys()):
            if n > s + 1:
                entries = stable_table.get((s, f))
            elif n == C2_N_VALUE:
                entries = c2_table.get((s, f))
            else:
                continue
            if not entries and self.r >= 6 and s + f <= self.HIGH_R_TRIVIAL_TOT:
                diff = self[n, s, f]
                if diff.nrows and diff.ncols and not diff.is_forced:
                    entries = [(ti, si, 0)
                               for si in range(diff.nrows)
                               for ti in range(diff.ncols)]
            if not entries:
                continue
            for (target_idx, source_idx, value) in entries:
                changed = self._impose_entry((n, s, f), target_idx, source_idx, value,
                                             label="stable" if n > s + 1 else "C2")
                if changed is None:
                    skipped += 1
                elif changed:
                    count += 1
        if skipped and not getattr(self, "_csv_skip_warned", False):
            self._csv_skip_warned = True
            print(
                f"  WARNING: {skipped} known d_{self.r} entries skipped — their "
                f"row/col indices exceed this dataset's dimensions (the CSVs "
                f"were exported from a session with a wider data range)"
            )
        return count > 0

    def check_stable(self):
        """Resolve stable-range (n > s + 1) and C2 (n == 0) differentials
        from the entry-level CSVs in stable/ (see stable/README.md for
        format and provenance); returns True if any differential was set."""
        stable_csv, c2_csv = self._known_diff_entries() or ({}, {})
        return self._check_stable_csv(stable_csv, c2_csv)

    # d_r vanishes for r >= 6 through this total degree (stable range and C2).
    HIGH_R_TRIVIAL_TOT = C2_DATA_COMPLETE_TOT

    def ratio_solved(self):
        """
        Calculate the percentage of nontrivial differentials that are solved.

        Returns:
            float: Percentage of forced differentials among nontrivial ones
        """
        nontrivial_count = 0
        forced_count = 0

        for bidegree in self.keys():
            diff_space = self[bidegree]

            # Skip trivial spaces (zero rows or columns)
            if diff_space.nrows == 0 or diff_space.ncols == 0:
                continue

            nontrivial_count += 1

            # Count if this differential is forced (solved)
            if diff_space.is_forced:
                forced_count += 1

        if nontrivial_count == 0:
            return 0.0

        return (forced_count / nontrivial_count) * 100.0

    def count_proof_reasons(self, max_n=None):
        """
        Count tridegrees by their proof reasons for length 1 proofs.

        Args:
            max_n: Maximum n value to consider (to match write_proofs filtering)

        Returns:
            dict: Counts of different proof types
        """
        stable_count = 0
        c2_count = 0
        other_count = 0

        for bidegree in self.keys():
            n, s, f = bidegree
            diff_space = self[bidegree]

            # Skip if n exceeds max limit (to match write_proofs filtering)
            if max_n is not None and n > max_n:
                continue

            # Skip trivial spaces (zero rows or columns)
            if diff_space.nrows == 0 or diff_space.ncols == 0:
                continue

            # Check if this has a length 1 proof (exactly one step in why list)
            if len(diff_space.why) == 1:
                why_entry = diff_space.why[0]
                label = why_entry.get("label", "")

                if label == "stable":
                    stable_count += 1
                elif label == "C2":
                    c2_count += 1
                else:
                    other_count += 1

        return {
            'stable': stable_count,
            'c2': c2_count,
            'other': other_count,
            'total_length_1': stable_count + c2_count + other_count
        }

    def to_json(self, **kwargs):
        """Serialize the page to a JSON-compatible dict keyed by bidegree string, plus the page number r"""
        ret = {str(bidegree): self[bidegree].to_json() for bidegree in self}
        ret["r"] = self.r
        return ret
