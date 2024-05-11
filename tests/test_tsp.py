"""Test travelling salesman solver module."""

from __future__ import annotations
import itertools
import numpy as np
from track import tsp


class Position(tsp.Destination):
    """A position in 2-space."""

    def __init__(self, position: tuple[float, float]):
        self.position = np.array(position)

    def distance_to(self, other: Position) -> int:
        max_error_mag = np.linalg.norm(self.position - other.position)
        # Scale by 1000 to minimize precision loss when quantizing to integer.
        return int(1000 * max_error_mag)


def test_solver():
    """Basic solver test.

    This test case checks for regressions in the solver with a nearly trivial test. This test case
    fails without the bug fix in the tsp module for this issue:
    https://github.com/seeing-things/track/issues/276.
    """
    # Intentionally suboptimal route with diagonal crossings between the vertices of a square.
    positions = [
        Position((0, 0)),
        Position((0, 1)),
        Position((1, 0)),
        Position((1, 1)),
    ]

    positions_sorted = tsp.solve_route(positions)

    # If the solver worked properly the route should traverse the perimeter of a square rather than
    # crossing over the diagonals.
    dist = sum(p1.distance_to(p2) for p1, p2 in itertools.pairwise(positions_sorted))
    assert dist == 4000
