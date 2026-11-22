"""Share the pre-READER margin priority between OOF and submission."""

import numpy as np


def rank_reader_rows(refined_lp, glm_winners, v7_rows):
    """Rank eligible rows by the gap between the two leading Apex labels."""
    eligible = np.flatnonzero(~(glm_winners | v7_rows))
    top = np.partition(refined_lp[eligible], -2, axis=1)[:, -2:]
    gap = top[:, 1] - top[:, 0]
    return eligible[np.argsort(gap, kind='stable')]
