"""Travelling salesman solver for optimizing route to set of multiple positions."""

from typing import List
from abc import ABC, abstractmethod
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np


class Destination(ABC):
    """Abstract class representing a destination along a route"""

    @abstractmethod
    def distance_to(self, other_destination: "Destination") -> int:
        """Returns the distance (cost) of travelling from this to the other destination

        Args:
            other_destination: Another instance of this class

        Returns:
            An integer value representing the distance. Type must be integer because the routing
            solver in Google OR-Tools does all computations with integers.
        """


def solve_route(destinations: List[Destination]) -> List[Destination]:
    """Travelling salesman problem solver

    Takes a list of destinations and sorts them such that the total time taken to visit each one is
    minimized. This is known as the travelling salesman problem.

    Much of the code in this function is borrowed from the example here:
    https://developers.google.com/optimization/routing/tsp

    Args:
        destinations: A list of places to visit. It is assumed that the first destination in the
            list is the first and last destination for the trip, otherwise known as the "depot".

    Returns:
        The list of places ordered such that visiting them in that order minimizes the total trip
        time.
    """

    # pre-compute a matrix of distances between each destination
    distance_matrix = np.zeros((len(destinations),)*2)
    for idx, dest in enumerate(destinations):
        for jdx in range(idx + 1, len(destinations)):
            distance_matrix[idx, jdx] = dest.distance_to(destinations[jdx])
    distance_matrix += np.transpose(distance_matrix)

    manager = pywrapcp.RoutingIndexManager(len(destinations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    route = [destinations[manager.IndexToNode(index)]]
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        route.append(destinations[manager.IndexToNode(index)])

    return route
