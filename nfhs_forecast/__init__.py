"""Noise-aware district forecasting of NFHS health indicators.

Train on NFHS-4 -> NFHS-5, freeze, then predict NFHS-5 -> NFHS-6.
See README.md and DECISIONS.md.
"""

__version__ = "0.1.0"

#: Survey round codes used throughout the package.
ROUNDS = (4, 5, 6)
#: The held-out test round. Must never enter training (see models.assert_no_round6).
TEST_ROUND = 6
