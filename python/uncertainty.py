#!/usr/bin/ipython3 -i

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
- Simple uncertainty: One r-value, one element, zero offset, one uncertainty generator
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
- element_is_uncertain_plus(): Check if element appears in uncertainty data
- update_uncertainties_from_differentials(): Extract uncertainties from d
- propagate_uncertainties_forward(): Track uncertainties through page turns
- save_uncertainties() / load_uncertainties(): Persistence

Uncertainty tracking enables:
- Partial computation when constraints are insufficient
- Identification of where additional information is needed
- Tracking which uncertainties persist to higher pages
"""

import json
import ast
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

    def are_any_tridegrees_unknown(self, tridegrees):
        """Check if any tridegrees in a list are unknown"""
        return any(self.is_tridegree_unknown(*td) for td in tridegrees)

    def remove_uncertain_tridegree(self, tridegree):
        """
        Temporarily remove a tridegree from self.uncertain.

        Args:
            tridegree: Tuple (n, s, f) to remove

        Returns:
            The removed data (dict) or None if tridegree doesn't exist
        """
        if tridegree in self.uncertain:
            return self.uncertain.pop(tridegree)
        return None

    def restore_uncertain_tridegree(self, tridegree, data):
        """
        Restore a previously removed tridegree to self.uncertain.

        Args:
            tridegree: Tuple (n, s, f) to restore
            data: The data that was returned from remove_uncertain_tridegree
        """
        if data is not None:
            self.uncertain[tridegree] = data

    def has_simple_uncertainty(self, n, s, f):
        """Check if tridegree has simple uncertainty: one r-value, one element, offset zero, uncertainty length 1"""
        if (n, s, f) not in self.uncertain or len(self.uncertain[n, s, f]) != 1:
            return False
        r_data = next(iter(self.uncertain[n, s, f].values()))
        elements = r_data.get('elements', {})
        if len(elements) != 1:
            return False
        element_data = next(iter(elements.values()))
        return (element_data.get('offset', None) is not None and
                element_data['offset'].is_zero() and
                len(element_data.get('uncertainty', [])) == 1)

    def get_simple_uncertainties_for_r(self, target_r):
        """Find all tridegrees with simple uncertainties for a specific r-value"""
        simple_uncertainties = []
        for tridegree in self.uncertain:
            if self.has_simple_uncertainty(*tridegree):
                r_data_dict = self.uncertain[tridegree]
                r_value = next(iter(r_data_dict.keys()))
                if r_value == target_r:
                    r_data = r_data_dict[r_value]
                    element, element_data = next(iter(r_data['elements'].items()))
                    uncertainty_element = element_data['uncertainty'][0]  # Length 1 guaranteed by has_simple_uncertainty
                    simple_uncertainties.append((tridegree, element, uncertainty_element))
        # Sort by n, s, then f
        simple_uncertainties.sort(key=lambda x: (x[1].n, x[1].s, x[1].f))
        return simple_uncertainties

    def get_all_uncertainties_for_r(self, target_r, max_t):
        """Find all tridegrees with uncertainties for a specific r-value (not just simple ones)

        Args:
            target_r: The r-value to get uncertainties for
            max_t: Total-degree cap; only elements with
                element.s + element.f + target_r < max_t - 2 are included
        """
        all_uncertainties = []
        for tridegree in self.uncertain:
            r_data_dict = self.uncertain[tridegree]
            if target_r in r_data_dict:
                r_data = r_data_dict[target_r]
                for element, element_data in r_data.get('elements', {}).items():
                    # Only include elements where s + f + r < max_t - 2
                    if element.s + element.f + target_r < max_t - 2:
                        all_uncertainties.append((tridegree, element, element_data))
        # Sort by n, s, then f of the element
        all_uncertainties.sort(key=lambda x: (x[1].n, x[1].s, x[1].f))
        return all_uncertainties

    def get_all_uncertainties(self, differentials_page):
        """Find all tridegrees with uncertainties for every r-value from 2 up to,
        but not including, the current page r

        Args:
            differentials_page: The differentials page object containing the current r value
        """
        all_uncertainties = []
        for target_r in range(2, differentials_page.r):
            if not hasattr(differentials_page, 'spectral_sequence') or not differentials_page.spectral_sequence:
                raise ValueError("DifferentialsPage must have spectral_sequence reference to get uncertainties")
            uncertainties_for_r = self.get_all_uncertainties_for_r(target_r, differentials_page.spectral_sequence.max_t)
            all_uncertainties.extend(uncertainties_for_r)
        return all_uncertainties

    def is_element_uncertain(self, element):
        """Check if a specific element has uncertain differential"""
        n, s, f = element.n, element.s, element.f
        if (n, s, f) not in self.uncertain:
            return False

        # Check if element is uncertain in any r-value
        for r_value, r_data in self.uncertain[n, s, f].items():
            if element in r_data.get('elements', {}):
                return True
        return False

    def element_is_uncertain_plus(self, element, r):
        """Check if element x is in uncertain, and also if element x is in offset or
        uncertainty for any element in uncertain[x_n, x_s + 1, x_f - q] for all q with 2 <= q < r

        Args:
            element: The element to check
            r: The current r-value from the differentials page
        """
        x_n, x_s, x_f = element.n, element.s, element.f

        # Check if element is directly in uncertain for any r-value
        if self.is_element_uncertain(element):
            return True

        # Check for each q with 2 <= q < r
        for q in range(2, r):
            target_tridegree = (x_n, x_s + 1, x_f - q)
            if target_tridegree in self.uncertain:
                # Check all r-values for this tridegree
                for r_value, r_data in self.uncertain[target_tridegree].items():
                    if 'elements' in r_data:
                        # Check all elements in this uncertain tridegree
                        for other_element, element_data in r_data['elements'].items():
                            # Check if element x is in the offset or uncertainty list
                            if 'offset' in element_data and element_data['offset'] == element:
                                return True
                            if 'uncertainty' in element_data:
                                for unc_element in element_data['uncertainty']:
                                    if unc_element == element:
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

    def add_unknown_tridegree(self, n, s, f, r_value=None, differentials_page=None):
        """Add a tridegree to the unknown set

        Args:
            n, s, f: Tridegree coordinates
            r_value: The r-value for this uncertainty (if None, uses differentials_page.r or 2)
            differentials_page: Optional differentials page to get r_value from
        """
        if r_value is None:
            r_value = differentials_page.r if differentials_page else 2

        if (n, s, f) not in self.uncertain:
            self.uncertain[n, s, f] = {}

        if r_value not in self.uncertain[n, s, f]:
            self.uncertain[n, s, f][r_value] = {'elements': {}}

    def save_uncertainties(self, filename):
        """Save unknown tridegrees to file"""
        with open(filename, 'w') as f:
            # Convert tuple keys and Element objects to strings for JSON serialization
            uncertain_serialized = {}
            for tridegree, r_data_dict in self.uncertain.items():
                serialized_r_data = {}
                for r_value, uncertain_data in r_data_dict.items():
                    serialized_elements = {}
                    for element, element_data in uncertain_data.get('elements', {}).items():
                        serialized_elements[str(element)] = {
                            'uncertainty': [str(u) for u in element_data['uncertainty']],
                            'offset': str(element_data['offset'])
                        }
                    serialized_r_data[str(r_value)] = {
                        'elements': serialized_elements
                    }
                uncertain_serialized[str(tridegree)] = serialized_r_data
            json.dump(uncertain_serialized, f)

    def load_uncertainties(self, filename, spectral_sequence):
        """Load unknown tridegrees from file

        Args:
            filename: Path to the uncertainties file
            spectral_sequence: The spectral sequence object (needed for Element.from_str)

        Returns:
            True if file loaded successfully, False if file not found
        """
        try:
            with open(filename, 'r') as f:
                uncertain_serialized = json.load(f)

            # Convert string keys back to tuples and strings back to Elements
            self.uncertain = {}
            for tridegree_str, r_data_dict in uncertain_serialized.items():
                tridegree = ast.literal_eval(tridegree_str)
                self.uncertain[tridegree] = {}

                for r_value_str, uncertain_data in r_data_dict.items():
                    r_value = int(r_value_str)
                    elements_dict = {}
                    for element_str, element_data in uncertain_data.get('elements', {}).items():
                        element = Element.from_str(element_str, spectral_sequence=spectral_sequence)
                        uncertainty_elements = [Element.from_str(u, spectral_sequence=spectral_sequence) for u in element_data['uncertainty']]
                        offset_element = Element.from_str(element_data['offset'], spectral_sequence=spectral_sequence)
                        elements_dict[element] = {
                            'uncertainty': uncertainty_elements,
                            'offset': offset_element
                        }
                    self.uncertain[tridegree][r_value] = {
                        'elements': elements_dict
                    }
            return True
        except FileNotFoundError:
            return False

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

    def transform_uncertainties(self, old_uncertain, make_new_element, spectral_sequence):
        """Transform all Element references in an uncertainty dict through a change-of-basis.

        Args:
            old_uncertain: The uncertain dictionary to transform (from previous page's uncertainty_manager)
            make_new_element: Callable (td, old_vect) -> Element that converts old-coordinates
                             vectors to new-page Elements. Should return zero element if vector is zero.
            spectral_sequence: The new spectral sequence page (for creating zero elements)
        """
        for (n, s, f), r_data_dict in old_uncertain.items():
            for r_value, uncertain_data in r_data_dict.items():
                for element, element_data in uncertain_data.get('elements', {}).items():
                    # Transform the element itself
                    elem_td = (element.n, element.s, element.f)
                    new_elem = make_new_element(elem_td, element.vect)
                    if new_elem.is_zero():
                        continue

                    # Transform the offset
                    offset = element_data['offset']
                    if offset.is_zero():
                        new_offset = spectral_sequence.zero(offset.n, offset.s, offset.f)
                    else:
                        offset_td = (offset.n, offset.s, offset.f)
                        new_offset = make_new_element(offset_td, offset.vect)

                    # Transform uncertainty generators, dropping any that become zero
                    new_uncertainty = []
                    for unc in element_data['uncertainty']:
                        unc_td = (unc.n, unc.s, unc.f)
                        new_unc = make_new_element(unc_td, unc.vect)
                        if not new_unc.is_zero():
                            new_uncertainty.append(new_unc)

                    # Initialize nested structure if needed
                    if (n, s, f) not in self.uncertain:
                        self.uncertain[n, s, f] = {}
                    if r_value not in self.uncertain[n, s, f]:
                        self.uncertain[n, s, f][r_value] = {'elements': {}}

                    # Store with preserved r-value
                    self.uncertain[n, s, f][r_value]['elements'][new_elem] = {
                        'offset': new_offset,
                        'uncertainty': new_uncertainty
                    }
