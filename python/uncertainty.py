"""
Uncertainty management for spectral sequences.

This module provides the UncertaintyManager class for tracking and managing
uncertainties in spectral sequence computations.

When differentials cannot be fully determined from available constraints, the
system tracks "uncertain" tridegrees where d(x) has the form:
    d(x) = offset + <uncertainty_1, uncertainty_2, ...>

where offset is the known part and uncertainty generators span possible additions.

Classes:
- UncertaintyManager: Tracks uncertain tridegrees across multiple r-values

Key Concepts:
- Uncertain tridegree: A tridegree (n,s,f) where some element's differential is not forced
- Uncertainty propagation: When turning pages, uncertain elements map through quotient
- Differential direction: d_q maps (n, s, f) to (n, s-1, f+q), so the d_q-source of
  (n, s, f) is (n, s+1, f-q); the "_plus" checks walk this arithmetic to find
  uncertain sources whose differentials could hit a given tridegree

Data Structure:
    uncertain = {
        (n,s,f): {
            r_value: {
                'elements': {
                    element: {
                        'offset': Element (definite part of d(element)),
                        'uncertainty': [Element, ...] (generators of indeterminacy)
                    }
                }
            }
        }
    }

Main Operations:
- is_tridegree_unknown(): Check if any differential at (n,s,f) is uncertain
- degree_is_uncertain_plus(): Check if (n,s,f) is uncertain or hit by one
- update_uncertainties_from_differentials(): Extract uncertainties from d
- propagate_uncertainties_forward(): Track uncertainties through page turns

Uncertainty tracking enables:
- Partial computation when constraints are insufficient
- Identification of where additional information is needed
- Tracking which uncertainties persist to higher pages
"""

from lib import Element


class UncertaintyManager:
    """Manages uncertainty tracking for spectral sequence computations"""

    def __init__(self):
        """Initialize the uncertainty manager with an empty uncertain dictionary"""
        self.uncertain = {}

    def is_tridegree_unknown(self, n, s, f):
        """Check if a tridegree has any elements with uncertain differentials"""
        if (n, s, f) not in self.uncertain:
            return False

        # Check if any r_value has actual elements with uncertainties
        for r_value, r_data in self.uncertain[n, s, f].items():
            if 'elements' in r_data and len(r_data['elements']) > 0:
                return True

        return False

    def degree_is_uncertain_plus(self, n, s, f, r):
        """Check if tridegree (n, s, f) is uncertain_plus:
        1) Has elements with uncertain differentials, OR
        2) Is in the uncertainty list of some d_q(x) where x is uncertain

        Args:
            n, s, f: The tridegree coordinates
            r: The current r-value from the differentials page
        """

        # Condition 1: Check if tridegree itself has elements with uncertain differentials
        if self.is_tridegree_unknown(n, s, f):
            return True

        # Condition 2: Check if (n, s, f) appears in uncertainty generators from potential sources
        # For each q with 2 <= q < r, check if d_q could hit (n, s, f)
        for q in range(2, r):
            source_tridegree = (n, s + 1, f - q)
            if source_tridegree not in self.uncertain:
                continue

            # Check all r_values for this source tridegree
            for r_value, r_data in self.uncertain[source_tridegree].items():
                if 'elements' not in r_data:
                    continue

                # d_{r_value} from source targets (n, s, f-q+r_value).
                # When r_value == q that's exactly (n, s, f), so any uncertainty
                # here affects us — even if generators were killed by the quotient
                # map during page turn.
                if r_value == q:
                    return True

                # Check each element's uncertainty generators
                for element, element_data in r_data['elements'].items():
                    for unc_element in element_data.get('uncertainty', []):
                        if (unc_element.n, unc_element.s, unc_element.f) == (n, s, f):
                            return True

        return False

    def get_lowest_r_uncertainty_for_element(self, element, n, s, f):
        """Get the uncertainty list for an element from the lowest r-value"""
        if (n, s, f) not in self.uncertain:
            return []

        # Find the lowest r-value where this element has uncertainty
        lowest_r = None
        for r_value, r_data in self.uncertain[n, s, f].items():
            if element in r_data.get('elements', {}):
                if lowest_r is None or r_value < lowest_r:
                    lowest_r = r_value

        if lowest_r is not None:
            return self.uncertain[n, s, f][lowest_r]['elements'][element]['uncertainty']

        return []

    def update_uncertainties_from_differentials(self, spectral_sequence):
        """Update unknown tridegrees based on current differentials page

        Args:
            spectral_sequence: The spectral sequence object containing adams_page and differentials_page

        Returns:
            Set of newly unknown tridegrees
        """
        if not spectral_sequence.d:
            return set()

        newly_unknown = set()
        current_r = spectral_sequence.d.r

        for (n, s, f) in list(spectral_sequence.d.keys()):
            if not spectral_sequence.d[n, s, f].is_forced:
                newly_unknown.add((n, s, f))

                # Initialize nested structure if tridegree doesn't exist
                if (n, s, f) not in self.uncertain:
                    self.uncertain[n, s, f] = {}

                # Rebuild this r-value's layer from the CURRENT differential
                # state (a stale layer from an earlier call would keep
                # since-resolved uncertainty records alive)
                self.uncertain[n, s, f][current_r] = {'elements': {}}

                # Check if (n, s, f) has elements before iterating
                if (n, s, f) in spectral_sequence.page:
                    for element in spectral_sequence.page[n, s, f]:
                        diff_result = spectral_sequence.d(element)
                        if diff_result['uncertainty']:
                            self.uncertain[n, s, f][current_r]['elements'][element] = {
                                'uncertainty': diff_result['uncertainty'],
                                'offset': diff_result['offset']
                            }
            elif ((n, s, f) in self.uncertain
                    and current_r in self.uncertain[n, s, f]):
                # The differential became forced since a previous call:
                # retire the stale current-page record (earlier pages' layers
                # are history and stay untouched)
                del self.uncertain[n, s, f][current_r]
                if not self.uncertain[n, s, f]:
                    del self.uncertain[n, s, f]

        return newly_unknown

    def propagate_uncertainties_forward(self, previous_uncertain, turned_page, spectral_sequence):
        """Propagate uncertainties from previous page using quotient maps from turned_page

        Args:
            previous_uncertain: The uncertain dictionary from the previous page
            turned_page: The turned page containing quotient maps
            spectral_sequence: The spectral sequence object (for creating new Elements)
        """
        for (n, s, f), r_data_dict in previous_uncertain.items():

            # Initialize nested structure even if source doesn't survive to next page,
            # so degree_is_uncertain_plus can detect that the target was affected
            if (n, s, f) not in self.uncertain:
                self.uncertain[n, s, f] = {}

            # Iterate through each r-value in the previous uncertain data
            for r_value, uncertain_data in r_data_dict.items():
                # Initialize this r-value if it doesn't exist
                if r_value not in self.uncertain[n, s, f]:
                    self.uncertain[n, s, f][r_value] = {'elements': {}}

            # Check if this tridegree exists in turned_page for element-level propagation
            if (n, s, f) not in turned_page:
                continue

            # Get the turned bidegree with quotient map
            turned_bidegree = turned_page[n, s, f]

            for r_value, uncertain_data in r_data_dict.items():
                # For each uncertain element in the previous page
                for e2_element, element_data in uncertain_data.get('elements', {}).items():
                    try:
                        # Apply quotient map to get E3 element
                        e3_vect = turned_bidegree.quotient_map(e2_element.vect)
                        if e3_vect.is_zero():
                            continue
                        e3_element = Element(n, s, f, e3_vect, spectral_sequence=spectral_sequence)
                    except (TypeError, ArithmeticError):
                        continue

                    # Apply quotient map to uncertainty elements
                    uncertainty_e3 = []
                    for unc_element in element_data['uncertainty']:
                        try:
                            unc_n, unc_s, unc_f = unc_element.n, unc_element.s, unc_element.f
                            if (unc_n, unc_s, unc_f) in turned_page:
                                unc_turned = turned_page[unc_n, unc_s, unc_f]
                                if hasattr(unc_turned, 'quotient_map'):
                                    unc_e3_vect = unc_turned.quotient_map(unc_element.vect)
                                    if not unc_e3_vect.is_zero():
                                        uncertainty_e3.append(Element(unc_n, unc_s, unc_f, unc_e3_vect, spectral_sequence=spectral_sequence))
                        except (TypeError, ArithmeticError):
                            continue

                    # Apply quotient map to offset element
                    offset_e3 = spectral_sequence.zero(n, s, f)
                    if not element_data['offset'].is_zero():
                        try:
                            offset_n, offset_s, offset_f = element_data['offset'].n, element_data['offset'].s, element_data['offset'].f
                            if (offset_n, offset_s, offset_f) in turned_page:
                                offset_turned = turned_page[offset_n, offset_s, offset_f]
                                if hasattr(offset_turned, 'quotient_map'):
                                    offset_e3_vect = offset_turned.quotient_map(element_data['offset'].vect)
                                    if not offset_e3_vect.is_zero():
                                        offset_e3 = Element(offset_n, offset_s, offset_f, offset_e3_vect, spectral_sequence=spectral_sequence)
                        except (TypeError, ArithmeticError):
                            pass

                    # Store the uncertainty data with preserved r-value
                    self.uncertain[n, s, f][r_value]['elements'][e3_element] = {
                        'uncertainty': uncertainty_e3,
                        'offset': offset_e3
                    }
