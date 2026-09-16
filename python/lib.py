"""
Core algebra library for the EHP spectral sequence computation.

This module owns the foundational data structures and operations:

Classes:
- Element: an element of an E_r page — a coefficient row vector over GF(2)
  at a tridegree (n,s,f)
- Map: maps between pages (suspension E, Hopf H, power P, C2, and the
  filtration-1 h0..h3 product maps on the n=0 column)
- AffineMatrixSubspace: constraints on differentials as affine subspaces of
  a matrix space
- TurnedBidegree: homology data for one tridegree after turning the page

Conventions at this module's boundaries:
- Tridegree (n,s,f): n=grade, s=stem, f=Adams filtration; the page number
  is r.
- Matrices (map tables, differentials) are SOURCE x TARGET over GF(2):
  row i is the image of the i-th source basis element, and elements act as
  row vectors on the left (image = v * M).
- Elements print as basis monomials "n_s_f_i" ("n_s_f" when the bidegree
  has dimension 1), sums as "a + b + ...", zero as "0".
- Products are bilinear over GF(2); the product cache stores each computed
  pair under both operand orders (see Element.multiply).

File I/O Functions:
- load_spectral_sequence(): Load page data from CSV files
- write_products(), write_maps(): Save computation results
- write_dimension_to_csv(): Export dimensions

Standard maps are defined in STANDARD_MAPS:
- E (suspension): (n,s,f) → (n+1,s,f)
- H (Hopf): (n,s,f) → (2n-1,s-n+1,f-1)
- P (power): (n,s,f) → ((n-1)//2,s+(n-1)//2-1,f+2) for odd n
- C2: (n,s,f) → (0,s-(n-2),f-1) for odd n
- h0/h1/h2/h3: (0,s,f) → (0,s+stem,f+1) with stems 0/1/3/7
"""
import csv
import json
import os
import re
from collections import defaultdict

import sage.all
from sage.all import GF, Hom, Sequence, VectorSpace, matrix, span, vector
from sage.geometry.hyperplane_arrangement.affine_subspace import AffineSubspace
from sage.modules.vector_space_morphism import VectorSpaceMorphism

class Element:
    """An element of an Adams E_r page: a coefficient row vector over GF(2)
    (self.vect) at tridegree (n, s, f)"""

    match_sfi = re.compile(r"(?P<n>\d+)_(?P<s>\d+)_(?P<f>\d+)(_(?P<i>\d+))?")

    @staticmethod
    def from_str(desc, spectral_sequence=None):
        """Parse an element from a string of monomials like "5_2_3_0 + 5_2_3_1" (or "0" for the zero element)"""
        if desc.strip() == "0":  # Handle the special case where the input is "0"
            if not spectral_sequence:
                raise ValueError("Element.from_str() requires spectral_sequence parameter")
            return spectral_sequence.zero(0, 0, 0)

        monomials = desc.split(" + ")
        if len(monomials) == 1:
            return Element.from_str_monomial(desc, spectral_sequence)

        first_monomial = monomials[0]
        sfi = Element.match_sfi.match(first_monomial)
        n = int(sfi.group("n"))
        s = int(sfi.group("s"))
        f = int(sfi.group("f"))
        if spectral_sequence is None:
            raise ValueError("Element.from_str() requires spectral_sequence parameter")
        dim = spectral_sequence.dimension[n, s, f]
        res = Element(n, s, f, [0 for _ in range(dim)], spectral_sequence=spectral_sequence)

        for m in monomials:
            i = int(Element.match_sfi.match(m).group("i"))
            res += Element(n, s, f, make_basis_vector(dim, i), spectral_sequence=spectral_sequence)

        return res

    @staticmethod
    def from_str_monomial(desc, spectral_sequence=None):
        """Parse a single monomial string "n_s_f" or "n_s_f_i" into the corresponding basis Element"""
        sfi = Element.match_sfi.match(desc)
        n = int(sfi.group("n"))
        s = int(sfi.group("s"))
        f = int(sfi.group("f"))
        option_i = sfi.group("i")
        if option_i is None:
            return Element(n, s, f, spectral_sequence=spectral_sequence)
        else:
            if spectral_sequence is None:
                raise ValueError("Element.from_str_monomial() requires spectral_sequence parameter")
            i = int(option_i)
            dim = spectral_sequence.dimension[n, s, f]
            return Element(n, s, f, make_basis_vector(dim, i), spectral_sequence=spectral_sequence)

    @staticmethod
    def from_str_page(desc, page, spectral_sequence=None):
        """Parse a monomial string "n_s_f" or "n_s_f_i" and look up the corresponding element stored in the given page"""
        sfi = Element.match_sfi.match(desc)
        n = int(sfi.group("n"))
        s = int(sfi.group("s"))
        f = int(sfi.group("f"))
        option_i = sfi.group("i")
        if option_i is None:
            elem = page[n, s, f][0]
            if spectral_sequence and not elem.spectral_sequence:
                elem.spectral_sequence = spectral_sequence
            return elem
        else:
            i = int(option_i)
            elem = page[n, s, f][i]
            if spectral_sequence and not elem.spectral_sequence:
                elem.spectral_sequence = spectral_sequence
            return elem

    def __init__(self, n, s, f, vect=None, spectral_sequence=None, r=None):
        # Default to the single generator [1]; a fresh list each call (never a
        # shared mutable default).
        vect = vector(GF(2), [1] if vect is None else vect)

        # Always preserve the tridegree, even for zero vectors
        self.n = n
        self.s = s
        self.f = f
        self.hash = 0
        self.vect = vect

        if vect.is_zero():
            self.dim = len(vect) if len(vect) > 0 else 1
        else:
            self.dim = len(vect)

        self.spectral_sequence = spectral_sequence
        self.r = r


    def __add__(self, other):

        if self.vect == 0:
            return other
        elif other.vect == 0:
            return self
        else:
            assert (
                self.s == other.s and self.f == other.f
            ), "Trying to sum elements in different bidegrees"
            return Element(self.n, self.s, self.f, self.vect + other.vect, spectral_sequence=self.spectral_sequence)

    def __contains__(self, other):
        if self.n != other.n or self.s != other.s or self.f != other.f:
            return False
        else:
            return all(i1 == 1 or i2 == 0 for i1, i2 in zip(self.vect, other.vect))

    def __eq__(self, other):
        return self.n == other.n and self.s == other.s and self.f == other.f and self.vect == other.vect

    def __hash__(self):
        if self.hash == 0:
            self.hash = hash(str(self))
        return self.hash

    def __lt__(self, other):
        if self.s != other.s:
            return self.s < other.s
        elif self.f != other.f:
            return self.f < other.f
        else:
            return self.vect < other.vect

    def __le__(self, other):
        return self < other or self == other

    def __mul__(self, other):
        if self.spectral_sequence is None:
            raise ValueError("Element must have spectral_sequence reference for multiplication")
        table = self.spectral_sequence.products
        return self.multiply(other, table)

    def __pow__(self, exp: int):
        if exp == 1:
            return self
        else:
            return self * self ** (exp - 1)

    def __str__(self):
        n = self.n
        s = self.s
        f = self.f
        vect = self.vect
        if vect == 0:
            return "0"
        elif len(vect) == 1:
            return f"{n}_{s}_{f}"
        else:
            ret = []
            for pos, i in enumerate(vect):
                if i == 1:
                    ret.append(f"{n}_{s}_{f}_{pos}")
            return " + ".join(ret)

    __radd__ = __add__

    __rmul__ = __mul__

    __repr__ = __str__

    def is_zero(self):
        """Return True if the underlying coefficient vector is zero"""
        return self.vect.is_zero()

    def decompose(self, mat=None):
        """Yield the basis monomials (one per nonzero coordinate) whose sum is this element; the mat argument is not implemented"""
        if mat is None:
            return (
                Element(self.n, self.s, self.f, make_basis_vector(self.dim, pos))
                for pos, i in enumerate(self.vect)
                if i == 1
            )
        else:
            raise NotImplementedError

    def multiply(self, other, products):
        """Multiply by another element using the given product table, summing products of basis monomials; results are cached, and products beyond max_s/max_f are zero"""
        # Check cache
        if (self, other) in products:
            return products[self, other]
        # Require spectral_sequence for all multiplication
        if not self.spectral_sequence:
            raise ValueError("Element must have spectral_sequence reference for multiplication")
        # Product tridegree
        prod_n = self.n
        prod_s = self.s + other.s
        prod_f = self.f + other.f
        # Zero is easy to multiply
        if self.dim == 0 or other.dim == 0:
            zero_prod = self.spectral_sequence.zero(prod_n, prod_s, prod_f)
            products[self, other] = zero_prod
            products[other, self] = zero_prod
            return zero_prod
        # The fundamental class is the composition unit (1 ∘ E^0 y = y and
        # x ∘ E^f(ι) = x); the relations table never records unit products
        # (the rust enumeration skips empty-word factors), so resolve them
        # here exactly instead of silently reading zero off the table.
        if self.s == 0 and self.f == 0 and not self.vect.is_zero():
            products[self, other] = other
            products[other, self] = other
            return other
        if other.s == 0 and other.f == 0 and not other.vect.is_zero():
            products[self, other] = self
            products[other, self] = self
            return self
        # Check if the product is outside the boundaries
        max_mult_s = self.spectral_sequence.max_s
        max_mult_f = self.spectral_sequence.max_f
        if prod_s > max_mult_s:
            zero_prod = self.spectral_sequence.zero(prod_n, prod_s, prod_f)
            products[self, other] = zero_prod
            products[other, self] = zero_prod
            return zero_prod
        elif (self.s != 0 or other.s != 0) and prod_f > max_mult_f:
            zero_prod = self.spectral_sequence.zero(prod_n, prod_s, prod_f)
            products[self, other] = zero_prod
            products[other, self] = zero_prod
            return zero_prod
        # Beyond the relations table's complete band an absent pair is
        # indistinguishable from a never-computed one, so a nonzero-dimension
        # target there must fail loudly instead of silently multiplying to
        # zero (the t=77 relations-hole failure mode). Set only on the loaded
        # E2 page; induced tables on later pages have their own semantics.
        complete_through = getattr(
            self.spectral_sequence, "products_complete_through", None)
        if (complete_through is not None
                and prod_s + prod_f > complete_through
                and self.spectral_sequence.dimension[prod_n, prod_s, prod_f] > 0):
            raise ValueError(
                f"product target ({prod_n}, {prod_s}, {prod_f}) has total "
                f"degree {prod_s + prod_f}, beyond the relations table's "
                f"complete band (t <= {complete_through}); its value cannot "
                f"be read off the table")
        # No more tricks, we've got to compute
        res = self.spectral_sequence.zero(prod_n, prod_s, prod_f)
        for summand1 in self.decompose():
            for summand2 in other.decompose():
                # The table may store a product under either operand order (the
                # algebra is commutative); missing the transposed key here would
                # silently read as zero and then be cached over the good entry.
                if (summand1, summand2) in products:
                    res += products[summand1, summand2]
                elif (summand2, summand1) in products:
                    res += products[summand2, summand1]
        # Sum might be zero
        if res.vect.is_zero():
            res = self.spectral_sequence.zero(prod_n, prod_s, prod_f)
        # Cache that stuff
        products[self, other] = res
        products[other, self] = res
        return res

    def apply_map(self, map_name):
        """Apply a map to this element"""
        if not self.spectral_sequence:
            raise ValueError("Element must have a spectral_sequence to apply maps")
        if map_name not in self.spectral_sequence.maps:
            raise ValueError(f"Unknown map: {map_name}")
        return self.spectral_sequence.maps[map_name].apply(self)
    
    def E(self):
        """Apply the E map to this element."""
        return self.apply_map('E')
    
    def indecomposable(self):
        """Returns True if this element cannot be written as a linear combination
        of elements that appear in the multiplication table outputs"""

        if self.spectral_sequence is None:
            raise ValueError("Element must have spectral_sequence reference for indecomposable check")

        if self == self.spectral_sequence.zero(self.n, self.s, self.f):
            return False  # Zero is always decomposable

        table = self.spectral_sequence.products
        
        # Collect all multiplication outputs in the same tridegree as self
        same_tridegree_products = []
        for product in table.values():
            if product.n == self.n and product.s == self.s and product.f == self.f:
                same_tridegree_products.append(product)
        
        if not same_tridegree_products:
            return True  # No products in this tridegree means indecomposable
        
        # Check if self can be written as a linear combination of these products
        # Create a vector space spanned by the products
        try:
            from sage.modules.free_module import span
            product_vectors = [p.vect for p in same_tridegree_products]
            product_span = span(product_vectors)

            # Check if self.vect is in the span
            return self.vect not in product_span
        except (ValueError, TypeError, AttributeError):
            # Fallback if span computation fails (empty list, incompatible vectors, etc.)
            return self not in same_tridegree_products
        
    def H(self):
        """Apply the H map to this element."""
        return self.apply_map('H')

    def P(self):
        """Apply the P map to this element."""
        return self.apply_map('P')

class Map:
    """A map between spectral sequence pages with degree shift; `table`
    caches images keyed by source Element"""

    def __init__(self, name, n_transform, s_transform, f_transform, table=None, domain_check=None):
        self.name = name
        self.n = n_transform  # Function: (n,s,f) → n_output
        self.s = s_transform  # Function: (n,s,f) → s_output
        self.f = f_transform  # Function: (n,s,f) → f_output
        self.table = table or {}
        self.domain_check = domain_check or (lambda n,s,f: True)
        # Highest source total degree (s+f) through which the table is known
        # to be COMPLETE (absent entry = zero image). None = complete
        # everywhere. The C2 map and hi product tables come from data files
        # with a hard degree cutoff; beyond it an absent entry means
        # "not computed", and building a matrix there would silently assert a
        # false zero map. Set from the loaded data; checked by data_complete().
        self.complete_through = None

    def data_complete(self, n, s, f):
        """True if the table can be trusted at source tridegree (n, s, f):
        within the completeness bound, a missing entry genuinely means zero."""
        return self.complete_through is None or s + f <= self.complete_through
    
    def target_degree(self, n, s, f):
        """Get target tridegree for given source tridegree"""
        return (self.n(n, s, f), self.s(n, s, f), self.f(n, s, f))
    
    def source_degree(self, target_n, target_s, target_f):
        """Get source tridegree that would map to the given target tridegree"""
        if self.name == "E":
            # E: n' = n+1, s' = s, f' = f
            return (target_n - 1, target_s, target_f)
        elif self.name == "H":
            # H: n' = 2n-1, s' = s-n+1, f' = f-1
            n = (target_n + 1) // 2
            s = target_s + n - 1
            f = target_f + 1
            return (n, s, f)
        elif self.name == "P":
            # P: n' = (n-1)//2, s' = s+(n-1)//2-1, f' = f+2
            n = 2 * target_n + 1
            s = target_s - target_n + 1
            f = target_f - 2
            return (n, s, f)
        elif self.name in _HI_STEM:
            # hi multiplication: n unchanged, s' = s - stem, f' = f - 1.
            return (target_n, target_s - _HI_STEM[self.name], target_f - 1)
        else:
            raise ValueError(f"source_degree not implemented for {self.name} map")
    
    def apply(self, element):
        """Apply this map to an element"""
        # Check cache
        if element in self.table:
            return self.table[element]

        # Require spectral_sequence for all map applications
        if not element.spectral_sequence:
            raise ValueError("Element must have spectral_sequence reference for map application")

        # Compute target tridegree
        target_n, target_s, target_f = self.target_degree(element.n, element.s, element.f)

        # Zero is easy to handle
        if element.dim == 0:
            zero_result = element.spectral_sequence.zero(target_n, target_s, target_f)
            self.table[element] = zero_result
            return zero_result

        # Check if the result is outside the boundaries
        max_n = element.spectral_sequence.max_n
        max_s = element.spectral_sequence.max_s
        max_f = element.spectral_sequence.max_f
        if target_n > max_n or target_s > max_s or target_f > max_f:
            zero_result = element.spectral_sequence.zero(target_n, target_s, target_f)
            self.table[element] = zero_result
            return zero_result

        # Special handling for E map: stable elements (n > s+1) map to same vector
        if self.name == "E" and element.n > element.s + 1:
            # Element is stable, E sends it to (n+1, s, f) with same vector
            stable_image = Element(target_n, target_s, target_f, element.vect,
                                 spectral_sequence=element.spectral_sequence)
            self.table[element] = stable_image
            return stable_image

        # Compute the result by decomposing the element
        res = element.spectral_sequence.zero(target_n, target_s, target_f)
        for summand in element.decompose():
            if summand in self.table:
                res += self.table[summand]

        # Sum might be zero
        if res.vect.is_zero():
            res = element.spectral_sequence.zero(target_n, target_s, target_f)

        # Cache the result
        self.table[element] = res
        return res
    
    def load_from_csv(self, csv_file, spectral_sequence):
        """Load map data from CSV file"""
        try:
            with open(csv_file, "r") as map_file:
                map_reader = csv.DictReader(map_file)
                for line in map_reader:
                    # Parse element from bidegrees (similar to load_spectral_sequence)
                    element = Element.from_str_page(line["element"], spectral_sequence.page, spectral_sequence)

                    # Handle "0" specially - create zero at correct target tridegree
                    if line["image"].strip() == "0":
                        target_n, target_s, target_f = self.target_degree(
                            element.n, element.s, element.f
                        )
                        image = spectral_sequence.zero(target_n, target_s, target_f)
                    else:
                        image = Element.from_str(line["image"], spectral_sequence)

                    self.table[element] = image
        except FileNotFoundError:
            # If CSV file doesn't exist, just continue with empty table
            pass
    
    def matrix(self, n, s, f, spectral_sequence):
        """Generate matrix representation for this map at given source degree"""
        # Check if map is defined on this tridegree
        if not self.domain_check(n, s, f):
            return matrix(ring=GF(2), nrows=spectral_sequence.dimension[n, s, f], ncols=0, entries=[])

        target_n, target_s, target_f = self.target_degree(n, s, f)

        # Check if source or target dimension is 0
        source_dim = spectral_sequence.dimension[n, s, f]
        target_dim = spectral_sequence.dimension[target_n, target_s, target_f]

        if source_dim == 0 or target_dim == 0:
            return matrix(ring=GF(2), nrows=source_dim, ncols=target_dim, entries=[])

        # Check if source key exists in page (should exist if source_dim > 0, but be safe)
        if (n, s, f) not in spectral_sequence.page:
            return matrix(ring=GF(2), nrows=source_dim, ncols=target_dim, entries=[])

        # Build matrix entries by applying map to each basis element
        entries = []
        for x in spectral_sequence.page[n, s, f]:
            mapped_x = self.apply(x)
            entries.append(mapped_x.vect)

        return matrix(
            ring=GF(2),
            nrows=source_dim,
            ncols=target_dim,
            entries=entries,
        )


# Standard map definitions
STANDARD_MAPS = {
    'E': Map("E", 
             n_transform=lambda n,s,f: n+1,
             s_transform=lambda n,s,f: s, 
             f_transform=lambda n,s,f: f,
             domain_check=lambda n,s,f: n >= 2),
             
    'H': Map("H",
             n_transform=lambda n,s,f: 2*n-1,
             s_transform=lambda n,s,f: s-n+1,
             f_transform=lambda n,s,f: f-1,
             domain_check=lambda n,s,f: n >= 2),
             
    'P': Map("P",
             n_transform=lambda n,s,f: (n-1)//2,
             s_transform=lambda n,s,f: s+(n-1)//2-1,
             f_transform=lambda n,s,f: f+2,
             domain_check=lambda n,s,f: n >= 5 and n % 2 == 1),
             
    'C2': Map("C2",
              n_transform=lambda n,s,f: 0 if n % 2 == 1 else n,
              s_transform=lambda n,s,f: s-(n-2) if n % 2 == 1 else s,
              f_transform=lambda n,s,f: f-1 if n % 2 == 1 else f,
              domain_check=lambda n,s,f: n >= 3 and n % 2 == 1),

    # Multiplication by the filtration-1 permanent cycles h0, h1, h2, h3 on the
    # n=0 Lambda(C2) column (stems 0, 1, 3, 7; each raises f by 1). Since the hi
    # never support differentials, d_r commutes with hi-multiplication (the
    # filtration-1 Leibniz rule d(x*hi) = d(x)*hi), so registering them as maps
    # lets natural/natural_rev resolve column differentials in both directions.
    # Their tables are loaded from E2_C2_products.csv.
    'h0': Map("h0", n_transform=lambda n,s,f: n, s_transform=lambda n,s,f: s,
              f_transform=lambda n,s,f: f+1, domain_check=lambda n,s,f: n == 0),
    'h1': Map("h1", n_transform=lambda n,s,f: n, s_transform=lambda n,s,f: s+1,
              f_transform=lambda n,s,f: f+1, domain_check=lambda n,s,f: n == 0),
    'h2': Map("h2", n_transform=lambda n,s,f: n, s_transform=lambda n,s,f: s+3,
              f_transform=lambda n,s,f: f+1, domain_check=lambda n,s,f: n == 0),
    'h3': Map("h3", n_transform=lambda n,s,f: n, s_transform=lambda n,s,f: s+7,
              f_transform=lambda n,s,f: f+1, domain_check=lambda n,s,f: n == 0),
}

# The four C2 filtration-1 product maps (h0/h1/h2/h3), registered alongside C2.
C2_PRODUCT_MAPS = ['h0', 'h1', 'h2', 'h3']

# stem of each hi multiplier, shared by Map.source_degree and any callers.
_HI_STEM = {'h0': 0, 'h1': 1, 'h2': 3, 'h3': 7}


class ContradictionError(ArithmeticError):
    """Raised when a constraint intersection comes up empty: the new constraint
    is incompatible with everything deduced so far, so either the input data or
    an earlier recorded differential is wrong.

    Carries the two inconsistent AffineMatrixSubspace states (`left` is the
    accumulated space, `right` the incoming constraint) so the failure site can
    print both proof chains."""

    def __init__(self, message, left=None, right=None):
        super().__init__(message)
        self.left = left
        self.right = right


class AffineMatrixSubspace:
    """An affine subspace of a space of matrices"""

    # Decoration

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.ambient = VectorSpace(GF(2), nrows * ncols)
        self.v = self.ambient.zero()
        self.subspace = AffineSubspace(self.v, self.ambient)
        self.why = []
        self.is_forced = nrows == 0 or ncols == 0

    def __str__(self):
        return self.subspace.__str__()

    __repr__ = __str__

    # Mathematics

    def __and__(self, other):
        intersection = AffineMatrixSubspace(self.nrows, self.ncols)
        intersected = self.subspace.intersection(other.subspace)
        if intersected is None:  # Sage returns None for an empty intersection
            raise ContradictionError(
                "constraint intersection is empty:\n"
                f"--- accumulated space ---\n{self.describe()}\n"
                f"--- incoming constraint ---\n{other.describe()}",
                left=self, right=other,
            )
        # Use set_subspace to ensure proper normalization (offset not in span)
        intersection.set_subspace(intersected.point(), intersected.linear_part())
        intersection.why = list(self.why)
        return intersection

    def __add__(self, other):
        if self.nrows != other.nrows or self.ncols != other.ncols:
            raise ArithmeticError(
                f"shape mismatch in +: {self.nrows}x{self.ncols} vs "
                f"{other.nrows}x{other.ncols}")
        union = AffineMatrixSubspace(self.nrows, self.ncols)
        union.set_subspace(
            self.v + other.v,
            self.ambient.span(
                self.subspace.linear_part().basis()
                + other.subspace.linear_part().basis()
            ),
        )
        return union

    def __floordiv__(self, Lx):
        """Compute the space of matrices `m` such that `m * Lx` lies in `self`"""
        lquotient = AffineMatrixSubspace(self.nrows, Lx.nrows())
        genmatrix = matrix(
            ring=GF(2),
            nrows=lquotient.ambient.dimension(),
            ncols=self.ambient.dimension(),
            entries=[
                vectorify(lquotient.as_matrix(gen) * Lx)
                for gen in lquotient.ambient.basis()
            ],
        )
        homspace = Hom(lquotient.ambient, self.ambient)
        blank_Lx = VectorSpaceMorphism(homspace, genmatrix)
        try:
            representative = blank_Lx.preimage_representative(self.v)
        except ValueError:
            # `self.v` is not literally in the image of blank_Lx. We need to adjust the offset.
            image = blank_Lx.image()
            adjusted = self.subspace.intersection(AffineSubspace(image.zero(), image))
            if adjusted is None:
                raise ContradictionError(
                    "// (floordiv) has no solution: no matrix m satisfies "
                    "m * Lx in the constraint space\n"
                    f"--- constraint space ---\n{self.describe()}",
                    left=self,
                )
            representative = blank_Lx.preimage_representative(adjusted.point())
        lquotient.set_subspace(
            representative, blank_Lx.inverse_image(self.subspace.linear_part())
        )
        return lquotient

    def __rfloordiv__(self, Lx):
        """Compute the space of matrices `m` such that `Lx * m` lies in `self`"""
        rquotient = AffineMatrixSubspace(Lx.ncols(), self.ncols)
        genmatrix = matrix(
            ring=GF(2),
            nrows=rquotient.ambient.dimension(),
            ncols=self.ambient.dimension(),
            entries=[
                vectorify(Lx * rquotient.as_matrix(gen))
                for gen in rquotient.ambient.basis()
            ],
        )
        homspace = Hom(rquotient.ambient, self.ambient)
        Lx_blank = VectorSpaceMorphism(homspace, genmatrix)
        try:
            representative = Lx_blank.preimage_representative(self.v)
        except ValueError:
            # `self.v` is not literally in the image of Lx_blank. We need to adjust the offset.
            image = Lx_blank.image()
            adjusted = self.subspace.intersection(AffineSubspace(image.zero(), image))
            if adjusted is None:
                raise ContradictionError(
                    "// (rfloordiv) has no solution: no matrix m satisfies "
                    "Lx * m in the constraint space\n"
                    f"--- constraint space ---\n{self.describe()}",
                    left=self,
                )
            representative = Lx_blank.preimage_representative(adjusted.point())
        rquotient.set_subspace(
            representative, Lx_blank.inverse_image(self.subspace.linear_part())
        )
        return rquotient

    def __mul__(self, Lx):
        """Compute the space of matrices of the form `m * Lx` for `m` in `self`"""
        if self.ncols != Lx.nrows():
            raise ArithmeticError(
                f"shape mismatch in subspace * matrix: subspace is "
                f"{self.nrows}x{self.ncols}, matrix is {Lx.nrows()}x{Lx.ncols()}")
        lproduct = AffineMatrixSubspace(self.nrows, Lx.ncols())
        gens = [
            vectorify(self.as_matrix(gen) * Lx)
            for gen in self.subspace.linear_part().basis()
        ]
        lproduct.set_subspace(
            vectorify(self.as_matrix(self.v) * Lx),
            lproduct.ambient.span(gens, base_ring=GF(2)),
        )
        return lproduct

    def __rmul__(self, Lx):
        """Compute the space of matrices of the form `Lx * m` for `m` in `self`"""
        if Lx.ncols() != self.nrows:
            raise ArithmeticError(
                f"shape mismatch in matrix * subspace: matrix is "
                f"{Lx.nrows()}x{Lx.ncols()}, subspace is {self.nrows}x{self.ncols} "
                f"(a 0-column matrix usually means a map was applied outside "
                f"its domain)")
        rproduct = AffineMatrixSubspace(Lx.nrows(), self.ncols)
        gens = [
            vectorify(Lx * self.as_matrix(gen))
            for gen in self.subspace.linear_part().basis()
        ]
        rproduct.set_subspace(
            vectorify(Lx * self.as_matrix(self.v)),
            rproduct.ambient.span(gens, base_ring=GF(2)),
        )
        return rproduct

    def upper_star(self, target_dimension):
        """Compute the affine subspace of matrices of the precomposition maps g -> d*g, i.e. the induced maps d^*: Hom(target, W) -> Hom(source, W) for W of the given dimension, as d ranges over `self`"""
        homspace = AffineMatrixSubspace(
            self.ncols * target_dimension, self.nrows * target_dimension
        )
        target_hom = AffineMatrixSubspace(self.ncols, target_dimension)
        gens = [
            vectorify(
                [
                    vectorify(self.as_matrix(d_gen) * target_hom.as_matrix(target_gen))
                    for target_gen in target_hom.ambient.basis()
                ]
            )
            for d_gen in self.subspace.linear_part().basis()
        ]
        homspace.set_subspace(
            vectorify(
                [
                    vectorify(self.as_matrix(self.v) * target_hom.as_matrix(target_gen))
                    for target_gen in target_hom.ambient.basis()
                ]
            ),
            homspace.ambient.span(gens, base_ring=GF(2)),
        )
        return homspace

    def lower_star(self, source_dimension):
        """Compute the affine subspace of matrices of the postcomposition maps g -> g*d, i.e. the induced maps d_*: Hom(W, source) -> Hom(W, target) for W of the given dimension, as d ranges over `self`"""
        homspace = AffineMatrixSubspace(
            source_dimension * self.nrows, source_dimension * self.ncols
        )
        source_hom = AffineMatrixSubspace(source_dimension, self.nrows)
        gens = [
            vectorify(
                [
                    vectorify(source_hom.as_matrix(source_gen) * self.as_matrix(d_gen))
                    for source_gen in source_hom.ambient.basis()
                ]
            )
            for d_gen in self.subspace.linear_part().basis()
        ]
        homspace.set_subspace(
            vectorify(
                [
                    vectorify(source_hom.as_matrix(source_gen) * self.as_matrix(self.v))
                    for source_gen in source_hom.ambient.basis()
                ]
            ),
            homspace.ambient.span(gens, base_ring=GF(2)),
        )
        return homspace

    def tensor(self, right_dimension):
        """Compute the affine subspace of matrices `d.tensor_product(I)` for `d` in `self`, where `I` is the identity of the given dimension"""
        ltensor = AffineMatrixSubspace(
            self.nrows * right_dimension, self.ncols * right_dimension
        )
        I = matrix.identity(GF(2), right_dimension)
        gens = [
            vectorify(self.as_matrix(gen).tensor_product(I))
            for gen in self.subspace.linear_part().basis()
        ]
        ltensor.set_subspace(
            vectorify(self.as_matrix(self.v).tensor_product(I)),
            ltensor.ambient.span(gens, base_ring=GF(2)),
        )
        return ltensor

    def rtensor(self, left_dimension):
        """Compute the affine subspace of matrices `I.tensor_product(d)` for `d` in `self`, where `I` is the identity of the given dimension"""
        rtensor = AffineMatrixSubspace(
            left_dimension * self.nrows, left_dimension * self.ncols
        )
        I = matrix.identity(GF(2), left_dimension)
        gens = [
            vectorify(I.tensor_product(self.as_matrix(gen)))
            for gen in self.subspace.linear_part().basis()
        ]
        rtensor.set_subspace(
            vectorify(I.tensor_product(self.as_matrix(self.v))),
            rtensor.ambient.span(gens, base_ring=GF(2)),
        )
        return rtensor

    # Misc

    def add_reason(self, counter, x_n, x_s, x_f, y_n, y_s, y_f, label, **kwargs):
        """Append a proof-reason record to `self.why`, including a snapshot of the current offset and subspace basis"""
        self.why.append(
            {
                "counter": counter,
                "x_n": x_n,
                "x_s": x_s,
                "x_f": x_f,
                "y_n": y_n,
                "y_s": y_s,
                "y_f": y_f,
                "label": label,
                # Add a snapshot of our current state
                "v": self.as_matrix(self.v),
                "subspace": [
                    self.as_matrix(basis_vector)
                    for basis_vector in self.subspace.linear_part().basis()
                ],
                "is_forced": len(self.subspace.linear_part().basis()) == 0
            }
        )

    def as_matrix(self, v):
        """Reshape a flattened ambient vector `v` into an nrows x ncols matrix over GF(2)"""
        return matrix(GF(2), v, nrows=self.nrows, ncols=self.ncols)

    def dimension(self) -> int:
        """Return the dimension of the affine subspace, i.e. the number of free parameters in the differential"""
        return self.subspace.dimension()

    def dimensions(self):
        """Return the matrix shape (nrows, ncols)"""
        return (self.nrows, self.ncols)

    def is_cycle(self) -> bool:
        """Return True if the differential is forced to be the zero matrix"""
        return self.is_forced and self.v.is_zero()

    def restrict(self, restriction, counter, label="", reason=None):
        """Intersect this subspace in place with another affine subspace and
        record a proof reason (empty label = "act of God"; `reason` is an
        optional (x_n, x_s, x_f, y_n, y_s, y_f) provenance tuple)"""
        intersected = self.subspace.intersection(restriction)
        if intersected is None:  # Sage returns None for an empty intersection
            raise ContradictionError(
                "restriction is incompatible with the accumulated space:\n"
                f"--- accumulated space ---\n{self.describe()}\n"
                f"--- restriction ---\n{restriction}",
                left=self,
            )
        # Use set_subspace to ensure proper normalization (offset not in span)
        self.set_subspace(intersected.point(), intersected.linear_part())
        self.add_reason(counter, *(reason or (0, 0, 0, 0, 0, 0)), label)

    def set_differential(self, mat, counter):
        """Restrict this subspace to the single matrix `mat`, recording a proof reason"""
        self.restrict(AffineSubspace(vectorify(mat), self.ambient.span([])), counter)

    def set_element_differential(self, element, target_differential, element_index, counter):
        """
        Force the differential of a specific element to be a specific value.
        """
        # Replace the specified row with the target differential
        current_matrix = self.as_matrix(self.v)
        new_matrix = matrix(current_matrix)
        new_matrix[element_index, :] = target_differential.vect.list() + [0] * (self.ncols - len(target_differential.vect))
        
        new_offset = vectorify(new_matrix)

        # Keep only basis vectors whose row at element_index is zero
        filtered_basis = [
            b for b in self.subspace.linear_part().basis()
            if all(self.as_matrix(b)[element_index, j] == 0 for j in range(self.ncols))
        ]

        new_linear_part = self.ambient.span(filtered_basis, base_ring=GF(2))
        new_subspace = AffineSubspace(new_offset, new_linear_part)

        self.v = new_offset
        self.subspace = new_subspace
        self.is_forced = new_subspace.dimension() == 0
        self.add_reason(counter, 0, 0, 0, 0, 0, 0, "set_element_differential")

    def set_subspace(self, v, span):
        """Set the affine subspace to the coset v + span, reducing v against the span so the stored offset is the canonical coset representative"""
        # Reduce v against the span's echelon basis to get the canonical
        # (minimal) coset representative.  This ensures that uncertain
        # components of the differential have offset zero.
        reduced = v
        for b in span.basis():
            if b.is_zero():
                continue
            pivot = b.support()[0]
            if reduced[pivot] != 0:
                reduced = reduced + b
        self.v = reduced
        self.subspace = AffineSubspace(self.v, span)
        self.is_forced = self.subspace.dimension() == 0

    def restricted_to(self, nrows, ncols):
        """Project this affine matrix space onto its leading nrows x ncols
        block.

        Used when loading a differential cache computed at a larger degree
        range into a smaller session: the current session's basis at each
        tridegree is a prefix of the cache's, so the cached knowledge about
        the loaded classes is exactly the image of the cached affine set under
        the (linear) block projection -- offset block plus the span of the
        basis blocks. No information is invented: components of d landing on
        unloaded classes are simply not represented, which is the same state a
        fresh truncated computation would be in."""
        restricted = AffineMatrixSubspace(nrows, ncols)
        if nrows == 0 or ncols == 0:
            restricted.set_subspace(
                restricted.ambient.zero(), restricted.ambient.span([]))
            restricted.why = list(self.why)
            return restricted

        def block(vec):
            return vectorify(self.as_matrix(vec).submatrix(0, 0, nrows, ncols))

        restricted.set_subspace(
            block(self.v),
            restricted.ambient.span(
                [block(b) for b in self.subspace.linear_part().basis()]),
        )
        restricted.why = list(self.why)
        return restricted

    def describe(self):
        """Human-readable state summary: shape, current offset and subspace
        dimension, and the recorded proof chain. Used by ContradictionError and
        the constraint-failure reporting, so a crash shows every deduction step
        that led to the inconsistent state."""
        lines = [
            f"shape {self.nrows}x{self.ncols}, "
            f"subspace dim {self.subspace.dimension()}, "
            f"forced={self.is_forced}",
            f"offset matrix: {self.as_matrix(self.v).rows()}",
        ]
        if not self.why:
            lines.append("proof chain: (empty - no recorded deductions)")
        else:
            lines.append(f"proof chain ({len(self.why)} steps):")
            for w in self.why:
                label = w.get("label") or "leibniz/act-of-god"
                lines.append(
                    f"  step {w['counter']}: [{label}] "
                    f"x=({w['x_n']},{w['x_s']},{w['x_f']}) "
                    f"y=({w['y_n']},{w['y_s']},{w['y_f']}) "
                    f"forced={w.get('is_forced')} "
                    f"offset={w['v'].rows() if hasattr(w['v'], 'rows') else w['v']}"
                )
        return "\n".join(lines)

    def print_why(self, index=None):
        """Print the recorded proof reasons (all of them, or a single one by index)"""
        if index is None:
            print(self.why)
        else:
            print(self.why[index])

    # Save/load

    @staticmethod
    def from_json(obj, **kwargs):
        """Reconstruct an AffineMatrixSubspace (offset, linear part, and proof reasons) from its JSON representation"""
        reconstructed = AffineMatrixSubspace(obj["nrows"], obj["ncols"])
        linear_part = reconstructed.ambient.span(obj["linear_part"])
        reconstructed.set_subspace(vector(GF(2), obj["v"]), linear_part)
        reconstructed.restore_why(obj["why"])
        reconstructed.is_forced = reconstructed.subspace.dimension() == 0
        return reconstructed

    def restore_why(self, saved_why):
        """Rebuild `self.why` from its JSON form, reshaping the saved offset and
        subspace-basis snapshots back into matrices"""
        self.why = [
            {
                "counter": why["counter"],
                "x_n": why["x_n"],
                "x_s": why["x_s"],
                "x_f": why["x_f"],
                "y_n": why["y_n"],
                "y_s": why["y_s"],
                "y_f": why["y_f"],
                "label": why["label"],
                "v": self.as_matrix(vector(GF(2), why["v"])),
                "subspace": [
                    self.as_matrix(vector(GF(2), basis_vector))
                    for basis_vector in why["subspace"]
                ],
            }
            for why in saved_why
        ]

    def to_json(self, **kwargs):
        """Serialize the offset, linear part, and proof reasons to a JSON-compatible dict"""
        return {
            "nrows": self.nrows,
            "ncols": self.ncols,
            "v": [int(i) for i in self.v],
            "linear_part": [
                [int(i) for i in basis_vector]
                for basis_vector in self.subspace.linear_part().basis()
            ],
            "why": [
                {
                    "counter": why["counter"],
                    "x_n": why["x_n"],
                    "x_s": why["x_s"],
                    "x_f": why["x_f"],
                    "y_n": why["y_n"],
                    "y_s": why["y_s"],
                    "y_f": why["y_f"],
                    "label": why["label"],
                    "v": [int(i) for i in vectorify(why.get("v", []))],
                    "subspace": [
                        [int(i) for i in vectorify(basis_vector)]
                        for basis_vector in why.get("subspace", [])
                    ],
                }
                for why in self.why
            ],
        }
    


class key_defaultdict(defaultdict):
    """A defaultdict whose default_factory receives the missing key as an argument"""
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            self[key] = self.default_factory(key)
            return self[key]


class TurnedBidegree:
    """Homology data for one tridegree after turning the page: a basis of elements, the quotient map onto the new basis, and the boundary space B"""
    def __init__(self, basis, quotient_map, B):
        self.basis = basis
        self.quotient_map = quotient_map
        self.B = B


def reduce_against(v, B):
    """Reduce the vector v modulo the subspace B, clearing v at the
    leading-pivot position of each vector of B's (echelonized) basis"""
    B_basis = B.basis()
    for row in B_basis:
        if v[row.list().index(1)]:
            v += row
    return v


def relative_row_echelon(B, Z):
    """Span of representatives for Z modulo B: each basis vector of Z not
    already in B, reduced against B.  Used by DifferentialsPage._turn_one
    with Z = cycles and B = boundaries to pick the homology basis"""
    Z_basis = Z.basis()
    Zprime = Sequence([], universe=Z, immutable=False, cr=True)
    for v in Z_basis:
        if v not in B:
            Zprime.append(reduce_against(v, B))
    return span(Zprime)


# Helper functions
#####################################################################


def create_differential_coset(nsf, r=2, dimension_dict=None):
    """Fresh, fully unconstrained space of candidate differentials d_r at
    tridegree nsf = (n, s, f): an AffineMatrixSubspace of shape
    dim(n,s,f) x dim(n,s-1,f+r) (source x target)"""
    n, s, f = nsf
    if dimension_dict is None:
        raise ValueError("dimension_dict is required for create_differential_coset")
    return AffineMatrixSubspace(dimension_dict[n, s, f], dimension_dict[n, s - 1, f + r])




def make_basis_vector(dim, pos):
    """Length-dim 0/1 list with a single 1 at index pos; all zeros when pos
    is out of range (deliberate -- see the comment below)"""
    vec = [0 for _ in range(dim)]
    if pos < dim:
        vec[pos] = 1
    # An out-of-range index means the referenced basis element is absent at
    # this tridegree -- routine on a tot-truncated load, where saved
    # products/relations cite classes outside the loaded range. The correct
    # value there is the zero vector, so fall through and return it silently.
    return vec


def trim_multiplication(products):
    """Filter a product table down to the entries with nonzero results (the
    form saved by write_products)"""
    slim = {}
    for (factor1, factor2), result in products.items():
        if result is not None and not result.vect.is_zero():
            slim[factor1, factor2] = result
    return slim




def vectorify(mat):
    """Flatten a matrix (or iterable of vectors) row-by-row into a single
    vector -- the inverse of AffineMatrixSubspace.as_matrix"""
    return vector(sum((list(row) for row in mat), []))


# File operations
def relations_complete_through(relations_file):
    """Largest total degree T such that the product table at `relations_file`
    is complete for every product target of total degree <= T.

    The band is the contiguous run of populated product-target degrees from
    the first populated one; a coverage gap ends it. A band reaching the
    table's top populated degree is trimmed by one: the rust generator only
    partially computes products landing AT its table bound, so the top
    degree of a fresh table is never complete. Missing/empty file -> 0."""
    per_degree = defaultdict(int)
    try:
        with open(relations_file) as fh:
            for line in csv.DictReader(fh):
                try:
                    p1 = line["factor1"].split("_")
                    p2 = line["factor2"].split("_")
                    per_degree[int(p1[1]) + int(p1[2])
                               + int(p2[1]) + int(p2[2])] += 1
                except (KeyError, ValueError, IndexError):
                    continue
    except FileNotFoundError:
        return 0
    if not per_degree:
        return 0
    top = max(per_degree)
    bound = first = min(per_degree)
    for t in range(first, top + 1):
        if per_degree.get(t, 0) == 0:
            break
        bound = t
    if bound == top:
        bound -= 1
    return max(bound, 0)


def load_spectral_sequence(prefix, r, tot, build_pairs=True):
    """Load a spectral sequence page from CSV files and return tuple (SpectralSequencePage, has_C2)"""
    # Import here to avoid circular imports
    from ehp_adams import SpectralSequencePage
    from differentials import DifferentialsPage

    # File paths - check directory first, then fallback to flat files
    import os
    if os.path.exists(prefix) and os.path.isdir(prefix):
        # Files inside directory always use "E2" prefix, directory name is distinguishing factor
        E_file = f"{prefix}/E2_E.csv"
        H_file = f"{prefix}/E2_H.csv"
        P_file = f"{prefix}/E2_P.csv"
        C2_file = f"{prefix}/E2_C2.csv"
        rank_file = f"{prefix}/E2_rank.csv"
        relations_file = f"{prefix}/E2_relations.csv"
        c2_products_file = f"{prefix}/E2_C2_products.csv"
        names_file = f"{prefix}/E2_names.json"
    else:
        E_file = f"{prefix}_E.csv"
        H_file = f"{prefix}_H.csv"
        P_file = f"{prefix}_P.csv"
        C2_file = f"{prefix}_C2.csv"
        rank_file = f"{prefix}_rank.csv"
        relations_file = f"{prefix}_relations.csv"
        c2_products_file = f"{prefix}_C2_products.csv"
        names_file = f"{prefix}_names.json"

    # Detect C2 availability by checking if file exists
    has_C2 = os.path.exists(C2_file)

    # Lower tot to the relations table's complete band: beyond it, absent
    # products would silently read as zero and falsify Leibniz constraints,
    # so the load bound follows the data rather than asking the caller to
    # track generation margins.
    band = relations_complete_through(relations_file)
    if (band and tot > band
            and os.environ.get("LAMBDA_ALLOW_INCOMPLETE_RELATIONS") != "1"):
        print(f"Relations table is complete through total degree {band}; "
              f"lowering tot {tot} -> {band}")
        tot = band

    # Set max_t=tot to prevent boundary shrinkage during compute_max_values
    ss = SpectralSequencePage(r=r, max_t=tot)

    bidegrees = {}  # Regular dict, not defaultdict

    # Load dimensions and create elements
    print(f"Loading dimensions from {rank_file}...")
    with open(rank_file, "r") as rank_in:
        rank_reader = csv.DictReader(rank_in)
        tridegree_count = 0
        for line in rank_reader:
            n, s, f = int(line["n"]), int(line["s"]), int(line["f"])
            if s + f > tot:
                continue
            # Deductions above n = 2 * max_stem are outside the range where
            # the loaded data is complete; drop those tridegrees entirely.
            if n > 2 * tot:
                continue
            dim = int(line["dimension"])
            ss.dimension[n, s, f] = dim
            # Only create and store elements if dimension > 0 (maintain invariant)
            if dim > 0:
                bidegree_vs = VectorSpace(GF(2), dim)
                bidegrees[n, s, f] = list(
                    map(lambda v: Element(n, s, f, v, spectral_sequence=ss), bidegree_vs.basis())
                )
                tridegree_count += 1
    print(f"  Loaded {tridegree_count} tridegrees with non-zero dimension")

    # Compute max values from loaded dimension data
    ss.compute_max_values()

    # Load multiplication table
    print(f"Loading multiplication table from {relations_file}...")
    product_count = 0
    skipped = 0
    with open(relations_file, "r") as relations_in:
        relations_reader = csv.DictReader(relations_in)
        for line in relations_reader:
            try:
                factor1 = Element.from_str_page(line["factor1"], bidegrees, ss)
                factor2 = Element.from_str_page(line["factor2"], bidegrees, ss)
                result = Element.from_str(line["result"], ss)
                ss.products[factor1, factor2] = result
                product_count += 1
            except (KeyError, ValueError, IndexError, AttributeError):
                # A row citing a class outside the loaded range, or a malformed
                # entry: skip it but count so the loss is not silent (a large
                # count is expected and harmless on a tot-truncated load).
                skipped += 1
    print(f"  Loaded {product_count} products"
          + (f" ({skipped} rows skipped: out-of-range or malformed)"
             if skipped else ""))

    # Element.multiply reads an absent pair as a zero product, which is only
    # sound inside the table's complete band (computed above, and already
    # folded into tot); record it so multiply() can fail loudly on any
    # request beyond it.
    ss.products_complete_through = min(band, ss.max_t) if band else ss.max_t
    if band < ss.max_t and os.environ.get(
            "LAMBDA_ALLOW_INCOMPLETE_RELATIONS") == "1":
        print(f"  WARNING: relations complete only through {band} but "
              f"tot={ss.max_t} loaded (override active); products in "
              f"({band}, {ss.max_t}] read as zero")

    # Initialize maps from STANDARD_MAPS (conditionally for C2)
    print("Initializing maps...")
    map_names = ['E', 'H', 'P']
    if has_C2:
        map_names.append('C2')
        map_names.extend(C2_PRODUCT_MAPS)

    for map_name in map_names:
        map_template = STANDARD_MAPS[map_name]
        # The hi product maps rely on their n==0 domain restriction; E/H/P/C2
        # keep the historical always-true default (their transforms already
        # produce zero maps off their intended domains).
        ss.maps[map_name] = Map(
            map_template.name,
            map_template.n,
            map_template.s,
            map_template.f,
            domain_check=map_template.domain_check if map_name in C2_PRODUCT_MAPS else None,
        )

    # Prepare map files to load (conditionally for C2)
    map_files = {
        'E': E_file,
        'H': H_file,
        'P': P_file,
    }
    if has_C2:
        map_files['C2'] = C2_file

    print("Loading map data...")
    for map_name, csv_file in map_files.items():
        map_entry_count = 0
        skipped = 0
        try:
            with open(csv_file, "r") as map_in:
                map_reader = csv.DictReader(map_in)
                for line in map_reader:
                    try:
                        element = Element.from_str_page(line["element"], bidegrees, ss)

                        # Handle "0" specially - create zero at correct target tridegree
                        if line["image"].strip() == "0":
                            target_n, target_s, target_f = ss.maps[map_name].target_degree(
                                element.n, element.s, element.f
                            )
                            image = ss.zero(target_n, target_s, target_f)
                        else:
                            image = Element.from_str(line["image"], ss)

                        ss.maps[map_name].table[element] = image
                        map_entry_count += 1
                    except (KeyError, ValueError, IndexError, AttributeError):
                        # Row citing an out-of-range class or malformed: skip
                        # but count (see the products loader above).
                        skipped += 1
            suffix = f" ({skipped} rows skipped)" if skipped else ""
            if map_name == "E":
                print(f"  {map_name} map: {map_entry_count} entries "
                      f"(stable elements n>s+1 computed automatically){suffix}")
            else:
                print(f"  {map_name} map: {map_entry_count} entries{suffix}")
        except FileNotFoundError:
            # If the CSV file doesn't exist, continue with an empty table.
            print(f"  {map_name} map: file not found, using empty table")

    # The C2 data file has a hard degree cutoff (the rust run's bound), while
    # the page itself may extend further; beyond the cutoff a missing table
    # entry means "not computed", not "zero". Record the observed coverage so
    # naturality skips tridegrees where the map matrix would be a false zero.
    if has_C2:
        c2_map = ss.maps['C2']
        c2_map.complete_through = max(
            (el.s + el.f for el in c2_map.table), default=-1)
        print(f"  C2 map complete through total degree {c2_map.complete_through}")

    # Load the C2 filtration-1 products (h0/h1/h2/h3 multiplication on the n=0
    # column) into the corresponding map tables. Format: factor1,generator,result
    # where generator is one of h0/h1/h2/h3 and factor1/result are n=0 classes.
    if has_C2:
        try:
            with open(c2_products_file, "r") as products_in:
                products_reader = csv.DictReader(products_in)
                hi_entry_count = 0
                skipped = 0
                for line in products_reader:
                    try:
                        generator = line["generator"].strip()
                        if generator not in ss.maps:
                            continue
                        element = Element.from_str_page(line["factor1"], bidegrees, ss)
                        if line["result"].strip() == "0":
                            tn, ts, tf = ss.maps[generator].target_degree(
                                element.n, element.s, element.f
                            )
                            image = ss.zero(tn, ts, tf)
                        else:
                            image = Element.from_str(line["result"], ss)
                        ss.maps[generator].table[element] = image
                        hi_entry_count += 1
                    except (KeyError, ValueError, IndexError, AttributeError):
                        # Out-of-range or malformed row: skip but count.
                        skipped += 1
            print(f"  C2 filtration-1 products: {hi_entry_count} entries"
                  + (f" ({skipped} rows skipped)" if skipped else ""))
        except FileNotFoundError:
            print("  C2 filtration-1 products: file not found, using empty tables")
        # Same completeness bookkeeping as for the C2 map: the products file
        # only covers products landing within the rust run's degree cap, so
        # each hi table is complete only through its observed source coverage.
        for hi_name in C2_PRODUCT_MAPS:
            hi_map = ss.maps[hi_name]
            hi_map.complete_through = max(
                (el.s + el.f for el in hi_map.table), default=-1)
        coverage = {name: ss.maps[name].complete_through for name in C2_PRODUCT_MAPS}
        print(f"  hi product tables complete through source total degree {coverage}")

    # Load names from JSON file
    print("Loading element names...")
    try:
        with open(names_file, "r") as f:
            ss.names = json.load(f)
        print(f"  Loaded {len(ss.names)} element names")
    except FileNotFoundError:
        print("  Names file not found, using empty dictionary")
        ss.names = {}

    # Set the page
    ss.page = bidegrees

    # Create d with the dimension dictionary
    print("Initializing differential structure...")
    ss.d = DifferentialsPage(r, dimension_dict=ss.dimension)

    # Build pairs
    if build_pairs:
        print("Building multiplication pairs...")
        ss.build_pairs()

    print(f"Loaded E_{r} page (C2 {'available' if has_C2 else 'not available'})")
    return (ss, has_C2)



def write_products(filename, products):
    """Write the nonzero entries of a product table to CSV with columns
    factor1, factor2, result (elements rendered by str())"""
    trim_products = trim_multiplication(products)
    colnames = ["factor1", "factor2", "result"]
    with open(filename, "w") as outfile:
        csvout = csv.DictWriter(
            outfile,
            colnames,
            dialect="excel",
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        )
        csvout.writeheader()
        for (factor1, factor2), result in trim_products.items():
            csvout.writerow({"factor1": factor1, "factor2": factor2, "result": result})


def write_maps(filename, maps_dict):
    """
    Write the maps (E, H, P, C2) to a CSV file.

    Args:
        filename (str): The name of the CSV file to write to.
        maps_dict (dict): The dictionary containing the maps.
                          Keys are elements, and values are their images.
    """
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["element", "image"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for element, image in maps_dict.items():
            writer.writerow({"element": str(element), "image": str(image)})


def write_dimension_to_csv(filename, next_page):
    """Write one "n,s,f,dimension" CSV row for every nonempty tridegree of
    next_page"""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n", "s", "f", "dimension"])  # Write header

        for n, s, f in next_page.keys():
            if len(next_page[n, s, f]) != 0:
                writer.writerow([n, s, f, len(next_page[n, s, f])])


