"""Spurious-uncertainty resolver for the EHP spectral sequence.

`resolve_spurious_uncertainties(ss, ...)` closes still-open differential
values by product and map reasoning that the Leibniz/naturality engines in
ehp_adams.py do not attempt. It is the self-contained replacement for the
hand-curated Spurious{r}.txt outside data: the published pipeline calls it as
a page step instead of imposing those lists.

The function takes the SpectralSequence page object (`ss`) and reads only its
public surface -- ss.d, ss.page, ss.maps, ss.d_source/d_target,
ss.is_cycle, ss.has_elements, ss.is_in_computed_polygon_source,
ss.degree_is_uncertain_plus, and the multiplier/map-name class constants -- so
it stays decoupled from the rest of the SpectralSequence implementation.
"""

import itertools

from lib import ContradictionError

# certain_nb enumerates 2**len(basis) affine representatives of an open
# differential's linear part; above this dimension it gives up and certifies
# nothing for images whose incoming differential originates at that source
# tridegree (post-compute spaces
# are almost all dimension 0-3, so this cap is rarely hit).
CERTAIN_NB_MAX_DIMENSION = 12


def resolve_spurious_uncertainties(ss, multiplier_bidegrees=None,
                                   map_names=None, max_stem=None):
    """Resolve still-open differential values by product and map
    reasoning the Leibniz/naturality engines do not attempt.  The scan
    is DEMAND-DRIVEN: deductions can only change open differential
    spaces, so it starts from those (a few hundred after compute) --
    direction (a) examines the elements of each open differential's
    target tridegree, direction (b) inverts the degree arithmetic to
    find the trusted cycles whose images land at the open space.  For
    each such element x and image under a certain-d_r-cycle multiplier
    h (img = x * h) or a map F in `map_names` (img = F(x)):

      (a) img certainly not a boundary  =>  x is not a boundary,
          since x = d(w) would give img = d(w * h) resp. d(F(w));
      (b) x certainly a cycle           =>  img is a cycle (force
          d(img) = 0), since d(x * h) = d(x) * h + x * d(h) resp.
          d(F(x)) = F(d(x)).  Applied for products and for the maps in
          MAP_CYCLE_NAMES (E, H, P) -- NOT for C2 -- and ONLY when x's
          tridegree is not uncertain_plus: is_cycle read at a guessed
          page basis is guess-relative, and (b) writes into the trusted
          region.  Direction (a) is allowed from any x -- its premise is
          read on the trusted side and its conclusion lands at x.

    Not-a-boundary facts are grouped per restricted differential and
    imposed together via nb_multi: combined exclusions often collapse to
    a single flat when each alone is a genuine multi-flat union; groups
    that still refuse are retried on the next fixpoint pass, when other
    forcings may have shrunk the space.

    `multiplier_bidegrees` is a list of (stem, filtration) pairs, default
    SPURIOUS_MULTIPLIER_BIDEGREES (h0-h3 plus the hand-picked single-
    class stable bidegrees); each is resolved per-x at the tridegree
    (x.n + x.s, stem, filtration).  `map_names` defaults to MAP_NB_NAMES,
    filtered to the maps present on this page.  `max_stem` restricts the
    worklist to open differentials with s <= max_stem (premises may
    still live anywhere); None = no cap.

    Soundness guards: missing product-table entries read as zero and yield
    no deduction; maps are only applied where domain_check holds and the
    table is data_complete (beyond complete_through a combination could
    silently map to a WRONG nonzero image); certainty is never read at an
    uncertain_plus tridegree (the image and, for the nb direction, its
    d_r-source must be trusted) nor outside the computed polygon (a
    dimension read as 0 there means "not loaded", not "zero"), and cycle
    premises additionally require x's d_r-target inside the polygon.

    Scans repeat until a full pass changes nothing (each forcing can make
    further images decidable).  Returns a counts dict; contradictions
    raise ContradictionError."""
    if multiplier_bidegrees is None:
        multiplier_bidegrees = ss.SPURIOUS_MULTIPLIER_BIDEGREES
    if map_names is None:
        map_names = ss.MAP_NB_NAMES
    r = ss.d.r
    counts = {"cycles_forced": 0, "nb_applied": 0, "nb_refused": 0,
              "passes": 0}
    warned_multipliers = set()
    # uncertain_plus is static during the resolver (the manager is only
    # updated at page turns), so cache it for the whole call
    up_cache = {}

    def uncertain_plus(td):
        if td not in up_cache:
            up_cache[td] = ss.degree_is_uncertain_plus(*td)
        return up_cache[td]

    while True:
        counts["passes"] += 1
        nB = {}  # x -> (route, img) that certified x is no boundary
        Z = {}   # img -> (route, x) whose cycle fact transfers to img
        cnb_cache = {}   # per-pass: spaces only shrink between passes
        mult_cache = {}  # per-pass: certain-cycle multipliers by coords

        def certain_nb(img):
            """img is certainly not a boundary iff no candidate of the
            incoming differential contains it in its row space.  The
            union of candidate row spaces is enumerated ONCE per source
            tridegree (post-compute spaces are almost all dimension 0-3)
            and shared by every image landing there -- same predicate as
            DifferentialsPage.certainly_not_boundary, without a flats
            computation per image."""
            src = ss.d_source(img.n, img.s, img.f)
            if src not in cnb_cache:
                diff = ss.d[src]
                if diff.nrows == 0 or diff.is_cycle():
                    cnb_cache[src] = frozenset()  # image is 0
                elif diff.dimension() > CERTAIN_NB_MAX_DIMENSION:
                    cnb_cache[src] = None  # too open to certify anything
                else:
                    hits = set()
                    basis = diff.subspace.linear_part().basis()
                    for bits in itertools.product((0, 1),
                                                  repeat=len(basis)):
                        w = diff.v
                        for bit, b in zip(bits, basis):
                            if bit:
                                w = w + b
                        for v in diff.as_matrix(w).row_space():
                            hits.add(tuple(v))
                    cnb_cache[src] = frozenset(hits)
            hits = cnb_cache[src]
            if hits is None:
                return False
            return tuple(img.vect) not in hits

        def multipliers_at(h_coords):
            if h_coords not in mult_cache:
                good = []
                if ss.has_elements(*h_coords):
                    target_ok = ss.is_in_computed_polygon_source(
                        *ss.d_target(*h_coords))
                    for h in ss.page[h_coords]:
                        if target_ok and ss.is_cycle(h):
                            good.append(h)
                        elif h not in warned_multipliers:
                            warned_multipliers.add(h)
                            print(f"  resolve_spurious: skipping "
                                  f"multiplier {h} at {h_coords}: not a "
                                  f"certain d{r}-cycle")
                mult_cache[h_coords] = good
            return mult_cache[h_coords]

        def consider(x, x_is_cycle, img, route,
                     allow_nb=True, allow_cycle=True):
            """Record the deductions carried by one image of x.

            Certainty is only ever read on the trusted side: the image
            tridegree and -- for the not-a-boundary direction -- the
            space that check reads must not be uncertain_plus (a guessed
            page basis certifies nothing) and must lie inside the
            computed polygon."""
            if img.is_zero():
                return
            img_td = (img.n, img.s, img.f)
            src_td = ss.d_source(*img_td)
            # nb READS certainty at img's tridegree and its d-source:
            # both must be trusted (certainty off a guessed basis is not
            # certainty).  The cycle transfer only WRITES at img's
            # tridegree; with trusted premises that is the safe trust
            # direction (trusted fact into an already-unreliable region),
            # so it is allowed into uncertain_plus targets.
            if (allow_nb
                    and not uncertain_plus(img_td)
                    and not uncertain_plus(src_td)
                    and ss.is_in_computed_polygon_source(*src_td)
                    and certain_nb(img)):
                nB.setdefault(x, (route, img))
            if (allow_cycle and x_is_cycle
                    and ss.is_in_computed_polygon_source(*img_td)):
                Z.setdefault(img, (route, x))

        def trusted_cycle(x):
            """The cycle premise must rest on real data: only from a
            trusted (not uncertain_plus) tridegree -- is_cycle at a
            guessed page basis is guess-relative -- and with the
            d_r-target inside the computed polygon (a target that was
            never loaded reads dimension 0, making the 1x0 differential
            trivially "zero"; H would carry that false fact from the
            window edge into the interior)."""
            td = (x.n, x.s, x.f)
            return (not uncertain_plus(td)
                    and ss.is_in_computed_polygon_source(
                        *ss.d_target(*td))
                    and ss.is_cycle(x))

        # Demand-driven worklist: a deduction can only change an OPEN
        # differential space, so start from those (a few hundred after
        # compute) rather than sweeping every element of the page.
        # Candidates come from the page itself (the differential at each
        # populated tridegree and the one entering it) so lazily
        # materialized d-entries are not missed.
        candidates = set()
        for td in list(ss.page.keys()):
            candidates.add(td)
            candidates.add(ss.d_source(*td))
        open_tds = []
        for sd in sorted(candidates):
            if max_stem is not None and sd[1] > max_stem:
                continue  # only resolve differentials through this stem
            diff = ss.d[sd]
            if diff.nrows > 0 and diff.ncols > 0 and not diff.is_forced:
                open_tds.append(sd)
        for sd in open_tds:
            n, s, f = sd
            # Direction (a): "the target of this possible differential
            # supports an extension to an element that cannot be in the
            # image, so the differential must miss it."  For each element
            # y of the open differential's target, certify an image of y
            # and record nb(y), which restricts d[sd].
            for y in ss.page.get(ss.d_target(n, s, f), []):
                y_cyc = trusted_cycle(y)
                for hs, hf in multiplier_bidegrees:
                    for h in multipliers_at((y.n + y.s, hs, hf)):
                        consider(y, y_cyc, y * h, f"*({hs},{hf})")
                for name in map_names:
                    map_obj = ss.maps.get(name)
                    if map_obj is None:
                        continue
                    if not map_obj.domain_check(y.n, y.s, y.f):
                        continue
                    if not map_obj.data_complete(y.n, y.s, y.f):
                        continue
                    consider(y, y_cyc, map_obj.apply(y), name,
                             allow_cycle=name in ss.MAP_CYCLE_NAMES)
            # Direction (b): trusted cycles whose image lands AT the open
            # differential -- invert the degree arithmetic to find them.
            # The write target may be uncertain_plus (trusted facts may
            # flow INTO the unreliable region); the premises may not.
            for hs, hf in multiplier_bidegrees:
                x_td = (n, s - hs, f - hf)
                if x_td not in ss.page:
                    continue
                for h in multipliers_at((n + s - hs, hs, hf)):
                    for x in ss.page[x_td]:
                        if trusted_cycle(x):
                            consider(x, True, x * h, f"*({hs},{hf})",
                                     allow_nb=False)
            for name in ss.MAP_CYCLE_NAMES:
                if name not in map_names:
                    continue
                map_obj = ss.maps.get(name)
                if map_obj is None:
                    continue
                try:
                    x_td = map_obj.source_degree(n, s, f)
                except ValueError:
                    continue
                if (x_td not in ss.page
                        or map_obj.target_degree(*x_td) != sd
                        or not map_obj.domain_check(*x_td)
                        or not map_obj.data_complete(*x_td)):
                    continue
                for x in ss.page[x_td]:
                    if trusted_cycle(x):
                        consider(x, True, map_obj.apply(x), name,
                                 allow_nb=False)

        changed = False
        for img in sorted(Z, key=str):
            route, x = Z[img]
            try:
                forced = ss.d.force_cycle(
                    img, label=f"cycle:{route}",
                    reason=(x.n, x.s, x.f, img.n, img.s, img.f))
            except ContradictionError:
                print(f"  resolve_spurious: CONTRADICTION forcing "
                      f"d{r}({img}) = 0 at ({img.n},{img.s},{img.f}), "
                      f"deduced from cycle x = {x} at ({x.n},{x.s},{x.f}) "
                      f"via {route}")
                raise
            if forced:
                counts["cycles_forced"] += 1
                changed = True
        # Group the not-a-boundary facts by the differential they
        # restrict and impose them together: combined exclusions often
        # collapse to a single flat even when each alone is a genuine
        # multi-flat union.  Refused groups are retried next pass.
        groups = {}
        for x, (route, img) in nB.items():
            groups.setdefault((x.n, x.s + 1, x.f - r), []).append(
                (x, route, img))
        for write_td in sorted(groups):
            facts = groups[write_td]
            xs = [x for x, _, _ in facts]
            routes = ",".join(sorted({route for _, route, _ in facts}))
            x0, _, img0 = facts[0]
            try:
                result = ss.d.nb_multi(
                    xs, label=f"nb:{routes}",
                    reason=(img0.n, img0.s, img0.f, x0.n, x0.s, x0.f),
                    verbose=False)
            except ContradictionError:
                for x, route, img in facts:
                    print(f"  resolve_spurious: CONTRADICTION imposing "
                          f"nb({x}) at ({x.n},{x.s},{x.f}), deduced from "
                          f"non-boundary image {img} at "
                          f"({img.n},{img.s},{img.f}) via {route}")
                raise
            if result is True:
                counts["nb_applied"] += 1
                changed = True
            elif result is None:
                counts["nb_refused"] += 1
        if not changed:
            break
    print(f"  resolve_spurious: {counts['cycles_forced']} cycles forced, "
          f"{counts['nb_applied']} non-boundaries imposed "
          f"({counts['nb_refused']} refused as multi-flat) over "
          f"{counts['passes']} passes")
    return counts
