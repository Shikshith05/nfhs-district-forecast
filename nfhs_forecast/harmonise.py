"""STUB: district boundary harmonisation across NFHS rounds. NOT IMPLEMENTED.

Status
------
The base pipeline does no harmonisation. It keeps only districts that match
unambiguously across NFHS-4, NFHS-5 and NFHS-6 and whose NFHS-6
`boundary_change_single_round` flag is False (see match.finalise_district_status).
This drops ~127 districts with no NFHS-4 values (boundaries changed between
NFHS-4 and NFHS-5) and ~33 districts flagged as changed in NFHS-6.

TODO (to be implemented later)
------------------------------
1. Parent-child crosswalk to NFHS-5 boundaries.
   Build a table child_district -> parent_district(s) with area/population
   shares for every split, merge or transfer between rounds (e.g.
   Chengalpattu <- Kanchipuram, Kallakurichi <- Villupuram, Mulugu <-
   Jayashankar Bhupalapally). Sources: Census 2011 codes in India.csv
   (the DISTRICT key holds the parent 2011 district), the LGD district
   directory, and data/raw/geo/Districts.geojson (Census 2011 polygons).

2. Population-weighted aggregation.
   Express every round on NFHS-5 boundaries:
       value_parent = sum_i w_i * value_child_i,   w_i = pop_i / sum pop
   using the appropriate denominator population for each indicator
   (households, women 15-49, children under 5, births ...), not total
   population, where the data allow.

3. SE propagation.
   For independent child estimates:
       se_parent = sqrt(sum_i w_i^2 * se_child_i^2)
   plus extra variance for weight uncertainty where shares are estimated.
   Flag harmonised values so models can down-weight them.

4. Tests: a split district re-aggregated from its children must reproduce
   the parent within tolerance on synthetic data; weights must sum to 1.
"""

from __future__ import annotations

import pandas as pd


def build_crosswalk(*args, **kwargs) -> pd.DataFrame:
    """TODO: parent-child crosswalk to NFHS-5 boundaries (see module docstring)."""
    raise NotImplementedError("Boundary harmonisation is out of scope for the base pipeline.")


def harmonise_to_nfhs5(panel: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """TODO: population-weighted re-aggregation with SE propagation (see module docstring)."""
    raise NotImplementedError("Boundary harmonisation is out of scope for the base pipeline.")
