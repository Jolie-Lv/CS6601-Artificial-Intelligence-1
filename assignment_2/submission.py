# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math

import itertools
import pdb
import copy

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.counter = itertools.count()

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        while self.queue:
            priority, count, node_id = heapq.heappop(self.queue)
            return (priority, node_id)
        raise KeyError('pop from an empty priority queue')

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        for qidx, (qcost, _, qnode) in enumerate(self.queue):
            if qnode is node_id:
                del self.queue[qidx]
                heapq.heapify(self.queue)
                return (qcost, qnode)
        # raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        priority, node_id = node
        count = next(self.counter)
        entry = [priority, count, node_id]
        heapq.heappush(self.queue, entry)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """
        if len(self.queue) == 0: return None
        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    path = []
    parent = {}
    if start is goal:
        return path
    frontier = PriorityQueue()
    frontier.append((0,start))
    parent[start] = None
    explored = set()
    found = False
    while frontier.size() > 0:
        _, curr_node = frontier.pop()
        explored.add(curr_node)
        neighbors = sorted(graph[curr_node])
        for neighbor in neighbors:
            if neighbor not in frontier and neighbor not in explored:
                parent[neighbor] = curr_node
                if neighbor is goal:
                    found = True
                    break
                frontier.append((0, neighbor))
        if found:
            break
    # Backtracking
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]

def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path = []
    parent = {}
    if start is goal:
        return path
    frontier = PriorityQueue()
    explored = set()
    frontier.append((0, start))
    parent[start] = None
    while frontier.size() > 0:
        cost, curr_node = frontier.pop()
        if curr_node is goal:
            break
        explored.add(curr_node)
        for neighbor in graph[curr_node]:
            weight = graph.get_edge_weight(curr_node, neighbor)
            if neighbor not in frontier and neighbor not in explored:
                ncost = cost+weight
                frontier.append((ncost, neighbor))
                parent[neighbor] = curr_node
            elif neighbor in frontier:
                frontier_cost, _ = frontier.remove(neighbor)
                if frontier_cost > cost+weight:
                    frontier.append((cost+weight, neighbor))
                    parent[neighbor] = curr_node
                else:
                    frontier.append((frontier_cost, neighbor))
    # Backtracking
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    v_x, v_y = graph.node[v]['pos']
    goal_x, goal_y = graph.node[goal]['pos']
    return math.sqrt((v_x-goal_x)**2 + (v_y-goal_y)**2)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path = []
    parent = {}
    if start is goal:
        return path
    frontier = PriorityQueue()
    explored = set()
    frontier.append((0+heuristic(graph, start, goal), start))
    parent[start] = None
    while frontier.size() > 0:
        cost, curr_node = frontier.pop()
        cost -= heuristic(graph, curr_node, goal)
        if curr_node is goal:
            break
        explored.add(curr_node)
        for neighbor in graph[curr_node]:
            weight = graph.get_edge_weight(curr_node, neighbor)
            if neighbor not in frontier and neighbor not in explored:
                ncost = cost+weight+heuristic(graph, neighbor, goal)
                frontier.append((ncost, neighbor))
                parent[neighbor] = curr_node
            elif neighbor in frontier:
                frontier_cost, _ = frontier.remove(neighbor)
                if frontier_cost > cost+weight+heuristic(graph, neighbor, goal):
                    frontier.append((cost+weight+heuristic(graph, neighbor, goal), neighbor))
                    parent[neighbor] = curr_node
                else:
                    frontier.append((frontier_cost, neighbor))
    # Backtracking
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]

def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # Initialize
    # pdb.set_trace()
    frontier_f, frontier_b = PriorityQueue(), PriorityQueue()
    explored_f, explored_b = set(), set()
    parents_f, parents_b = {}, {}
    best_path = []
    mu = float("inf") # least path length from start to goal
    if start is goal:
        return best_path
    frontier_f.append((0,start))
    frontier_b.append((0,goal))
    parents_f[start] = None
    parents_b[goal] = None
    while frontier_f.size() > 0 and frontier_b.size() > 0:
        top_f, top_b = frontier_f.top()[0], frontier_b.top()[0]
        if top_f + top_b >= mu:
            return best_path

        if top_f <= top_b:
            cost_f, curr_node_f = frontier_f.pop()
            # cost_f -= heuristic(graph, curr_node_f, goal)
            explored_f.add(curr_node_f)

            # check if goal is reached just using forward path
            if curr_node_f is goal:
                # cost is less than current best cost
                if cost_f < mu:
                    node = curr_node_f
                    path = []
                    while node is not None:
                        path.append(node)
                        node = parents_f[node]
                    path = path[::-1]
                    best_path = path
            # current node is in frontier of backward path
            elif curr_node_f in frontier_b:
                mu_new, best_path_new = _find_path(graph, curr_node_f, parents_f, parents_b)
                if mu_new < mu:
                    mu = mu_new
                    best_path = best_path_new

            # Expand forward path
            for neighbor in graph[curr_node_f]:
                # Check if neighbor is in backward search explored set
                if neighbor not in frontier_f and neighbor not in explored_f:
                    frontier_f.append((cost_f+graph.get_edge_weight(curr_node_f, neighbor), neighbor))
                    parents_f[neighbor] = curr_node_f
                elif neighbor in frontier_f:
                    frontier_cost_f, _ = frontier_f.remove(neighbor)
                    if frontier_cost_f > cost_f+graph.get_edge_weight(curr_node_f, neighbor):
                        frontier_f.append((cost_f+graph.get_edge_weight(curr_node_f, neighbor),neighbor))
                        parents_f[neighbor] = curr_node_f
                    else:
                        frontier_f.append((frontier_cost_f, neighbor))
        else:
            cost_b, curr_node_b = frontier_b.pop()
            # cost_b -= heuristic(graph, curr_node_b, start)
            explored_b.add(curr_node_b)

            # check if start is reached just using backward path
            if curr_node_b is start:
                # cost is less than current best cost
                if cost_b < mu:
                    node = curr_node_b
                    path = []
                    while node is not None:
                        path.append(node)
                        node = parents_b[node]
                    path = path[::-1]
                    best_path = path
            # current node is in frontier of forward path
            elif curr_node_b in frontier_f:
                mu_new, best_path_new = _find_path(graph,curr_node_b, parents_b, parents_f)
                if mu_new < mu:
                    mu = mu_new
                    best_path = best_path_new

            # Expand backward path
            for neighbor in graph[curr_node_b]:
                if neighbor not in frontier_b and neighbor not in explored_b:
                    frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor), neighbor))
                    parents_b[neighbor] = curr_node_b
                    # Check if neigh is in forward search frontier

                elif neighbor in frontier_b:
                    frontier_cost_b, _ = frontier_b.remove(neighbor)
                    if frontier_cost_b > cost_b+graph.get_edge_weight(curr_node_b, neighbor):
                        frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor),neighbor))
                        parents_b[neighbor] = curr_node_b
                    else:
                        frontier_b.append((frontier_cost_b, neighbor))
    return best_path

def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # Initialize
    # pdb.set_trace()
    frontier_f, frontier_b = PriorityQueue(), PriorityQueue()
    explored_f, explored_b = set(), set()
    parents_f, parents_b = {}, {}
    best_path = []
    mu = float("inf") # least path length from start to goal
    if start is goal:
        return best_path
    frontier_f.append((0+heuristic(graph,start,goal),start))
    frontier_b.append((0+heuristic(graph,goal,start),goal))
    parents_f[start] = None
    parents_b[goal] = None
    while frontier_f.size() > 0 and frontier_b.size() > 0:
        top_f, top_b = frontier_f.top()[0], frontier_b.top()[0]
        top_node_f, top_node_b = frontier_f.top()[2], frontier_b.top()[2]
        top_f -= heuristic(graph, top_node_f, goal)
        top_b -= heuristic(graph, top_node_b, start)
        if top_f + top_b >= mu:
            return best_path

        if top_f <= top_b:
            cost_f, curr_node_f = frontier_f.pop()
            cost_f -= heuristic(graph, curr_node_f, goal)
            explored_f.add(curr_node_f)

            # check if goal is reached just using forward path
            if curr_node_f is goal:
                # cost is less than current best cost
                if cost_f < mu:
                    node = curr_node_f
                    path = []
                    while node is not None:
                        path.append(node)
                        node = parents_f[node]
                    path = path[::-1]
                    best_path = path
            # current node is in frontier of backward path
            elif curr_node_f in frontier_b:
                mu_new, best_path_new = _find_path(graph, curr_node_f, parents_f, parents_b)
                if mu_new < mu:
                    mu = mu_new
                    best_path = best_path_new

            # Expand forward path
            for neighbor in graph[curr_node_f]:
                # Check if neighbor is in backward search explored set
                if neighbor not in frontier_f and neighbor not in explored_f:
                    frontier_f.append((cost_f+graph.get_edge_weight(curr_node_f, neighbor)+heuristic(graph, neighbor, goal), neighbor))
                    parents_f[neighbor] = curr_node_f
                elif neighbor in frontier_f:
                    frontier_cost_f, _ = frontier_f.remove(neighbor)
                    if frontier_cost_f > cost_f+graph.get_edge_weight(curr_node_f, neighbor)+heuristic(graph, neighbor, goal):
                        frontier_f.append((cost_f+graph.get_edge_weight(curr_node_f, neighbor)+heuristic(graph, neighbor, goal),neighbor))
                        parents_f[neighbor] = curr_node_f
                    else:
                        frontier_f.append((frontier_cost_f, neighbor))
        else:
            cost_b, curr_node_b = frontier_b.pop()
            cost_b -= heuristic(graph, curr_node_b, start)
            explored_b.add(curr_node_b)

            # check if start is reached just using backward path
            if curr_node_b is start:
                # cost is less than current best cost
                if cost_b < mu:
                    node = curr_node_b
                    path = []
                    while node is not None:
                        path.append(node)
                        node = parents_b[node]
                    path = path[::-1]
                    best_path = path
            # current node is in frontier of forward path
            elif curr_node_b in frontier_f:
                mu_new, best_path_new = _find_path(graph,curr_node_b, parents_b, parents_f)
                if mu_new < mu:
                    mu = mu_new
                    best_path = best_path_new

            # Expand backward path
            for neighbor in graph[curr_node_b]:
                if neighbor not in frontier_b and neighbor not in explored_b:
                    frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor)+heuristic(graph, neighbor, start), neighbor))
                    parents_b[neighbor] = curr_node_b
                    # Check if neigh is in forward search frontier

                elif neighbor in frontier_b:
                    frontier_cost_b, _ = frontier_b.remove(neighbor)
                    if frontier_cost_b > cost_b+graph.get_edge_weight(curr_node_b, neighbor)+heuristic(graph, neighbor, start):
                        frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor)+heuristic(graph, neighbor, start),neighbor))
                        parents_b[neighbor] = curr_node_b
                    else:
                        frontier_b.append((frontier_cost_b, neighbor))
    return best_path

def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # pdb.set_trace()
    a, b, c = goals
    if a is b and b is c:
        return []
    if a is b:
        return _bidirectional_ucs(graph, a, c)
    if a is c:
        return _bidirectional_ucs(graph, a, b)
    if b is c:
        return _bidirectional_ucs(graph, a, b)
    frontier_a, frontier_b, frontier_c = PriorityQueue(), PriorityQueue(), PriorityQueue()
    explored_a, explored_b, explored_c = set(), set(), set()
    found_ab, found_bc, found_ac = False, False, False
    mu_ab, mu_bc, mu_ac = float("inf"), float("inf"), float("inf")
    best_path_ab, best_path_bc, best_path_ac = [], [], []
    parents_a, parents_b, parents_c = {}, {}, {}
    parents_a[a], parents_b[b], parents_c[c] = None, None, None
    frontier_a.append((0,a))
    frontier_b.append((0,b))
    frontier_c.append((0,c))
    while not (found_ab and found_bc and found_ac):
        top_a, top_b, top_c = frontier_a.top(), frontier_b.top(), frontier_c.top()
        if top_a is None: top_a = [float("inf"), 0, '<invalid>']
        if top_b is None: top_b = [float("inf"), 0, '<invalid>']
        if top_c is None: top_c = [float("inf"), 0, '<invalid>']

        if top_a[0] + top_b[0] >= mu_ab:
            found_ab = True

        if top_b[0] + top_c[0] >= mu_bc:
            found_bc = True

        if top_c[0] + top_a[0] >= mu_ac:
            found_ac = True

        # Explore a
        if top_a is not None and top_b is not None and top_c is not None and top_a[0] <= top_b[0] and top_a[0] <= top_c[0]:
            cost_a, curr_node_a = frontier_a.pop()
            explored_a.add(curr_node_a)

            if found_ab and found_ac:
                continue

            if not found_ab and curr_node_a is b:
                # if current cost is less than best cost path
                if cost_a < mu_ab:
                    path_ab = _backtrack_path(curr_node_a, parents_a)
                    mu_ab, best_path_ab = cost_a, path_ab

            if not found_ab and curr_node_a in frontier_b:
                mu_ab_new, best_path_ab_new = _find_path(graph, curr_node_a, parents_a, parents_b)
                if mu_ab_new < mu_ab:
                    mu_ab, best_path_ab = mu_ab_new, best_path_ab_new

            if not found_ac and curr_node_a is c:
                # if current cost is less than the best cost path
                if cost_a < mu_ac:
                    path_ac = _backtrack_path(curr_node_a, parents_a)
                    mu_ac, best_path_ac = cost_a, path_ac

            if not found_ac and curr_node_a in frontier_c:
                mu_ac_new, best_path_ac_new = _find_path(graph, curr_node_a, parents_a, parents_c)
                if mu_ac_new < mu_ac:
                    mu_ac, best_path_ac = mu_ac_new, best_path_ac_new

            for neighbor in graph[curr_node_a]:
                if neighbor not in frontier_a and neighbor not in explored_a:
                    frontier_a.append((cost_a+graph.get_edge_weight(curr_node_a,neighbor),neighbor))
                    parents_a[neighbor] = curr_node_a

                elif neighbor in frontier_a:
                    frontier_cost_a, _ = frontier_a.remove(neighbor)
                    if frontier_cost_a > cost_a+graph.get_edge_weight(curr_node_a, neighbor):
                        frontier_a.append((cost_a+graph.get_edge_weight(curr_node_a, neighbor), neighbor))
                        parents_a[neighbor] = curr_node_a
                    else:
                        frontier_a.append((frontier_cost_a, neighbor))

        # Explore b
        elif top_b[0] < top_a[0] and top_b[0] < top_c[0]:
            cost_b, curr_node_b = frontier_b.pop()
            explored_b.add(curr_node_b)

            if found_bc and found_ab:
                continue

            if not found_ab and curr_node_b is a:
                if cost_b < mu_ab:
                    path_ba = _backtrack_path(curr_node_b, parents_b)
                    path_ab = path_ba[::-1]
                    mu_ab, best_path_ab = cost_b, path_ab

            if not found_ab and curr_node_b in frontier_a:
                mu_ab_new, best_path_ab_new = _find_path(graph, curr_node_b, parents_a, parents_b)
                if mu_ab_new < mu_ab:
                    mu_ab, best_path_ab = mu_ab_new, best_path_ab_new

            if not found_bc and curr_node_b is c:
                if cost_b < mu_bc:
                    path_bc = _backtrack_path(curr_node_b, parents_b)
                    mu_bc, best_path_bc = cost_b, path_bc

            if not found_bc and curr_node_b in frontier_c:
                mu_bc_new, best_path_bc_new = _find_path(graph, curr_node_b, parents_b, parents_c)
                if mu_bc_new < mu_bc:
                    mu_bc, best_path_bc = mu_bc_new, best_path_bc_new

            for neighbor in graph[curr_node_b]:
                if neighbor not in frontier_b and neighbor not in explored_b:
                    frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor), neighbor))
                    parents_b[neighbor] = curr_node_b
                elif neighbor in frontier_b:
                    frontier_cost_b, _ = frontier_b.remove(neighbor)
                    if frontier_cost_b > cost_b+graph.get_edge_weight(curr_node_b, neighbor):
                        frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor), neighbor))
                        parents_b[neighbor] = curr_node_b
                    else:
                        frontier_b.append((frontier_cost_b, neighbor))

        # Explore c
        else:
            cost_c, curr_node_c = frontier_c.pop()
            explored_c.add(curr_node_c)

            if found_ac and found_bc:
                continue

            if not found_ac and curr_node_c is a:
                if cost_c < mu_ac:
                    path_ca = _backtrack_path(curr_node_c, parents_c)
                    path_ac = path_ca[::-1]
                    mu_ac, best_path_ac = cost_c, path_ac

            if not found_ac and curr_node_c in frontier_a:
                mu_ac_new, best_path_ac_new = _find_path(graph, curr_node_c, parents_a, parents_c)
                if mu_ac_new < mu_ac:
                    mu_ac, best_path_ac = mu_ac_new, best_path_ac_new

            if not found_bc and curr_node_c is b:
                if cost_c < mu_bc:
                    path_cb = _backtrack_path(curr_node_c, parents_c)
                    path_bc = path_cb[::-1]
                    mu_bc, best_path_bc = cost_c, path_bc

            if not found_bc and curr_node_c in frontier_b:
                mu_bc_new, best_path_bc_new = _find_path(graph, curr_node_c, parents_b, parents_c)
                if mu_bc_new < mu_bc:
                    mu_bc, best_path_bc = mu_bc_new, best_path_bc_new

            for neighbor in graph[curr_node_c]:
                if neighbor not in frontier_c and neighbor not in explored_c:
                    frontier_c.append((cost_c+graph.get_edge_weight(curr_node_c,neighbor), neighbor))
                    parents_c[neighbor] = curr_node_c
                elif neighbor in frontier_c:
                    frontier_cost_c, _ = frontier_c.remove(neighbor)
                    if frontier_cost_c > cost_c+graph.get_edge_weight(curr_node_c,neighbor):
                        frontier_c.append((cost_c+graph.get_edge_weight(curr_node_c,neighbor), neighbor))
                        parents_c[neighbor] = curr_node_c
                    else:
                        frontier_c.append((frontier_cost_c,neighbor))

    # construct single path from 3 paths
    # pdb.set_trace()
    if mu_ab+mu_ac <= mu_ab+mu_bc and mu_ab + mu_ac <= mu_ac + mu_bc:
        if all([ac in best_path_bc for ac in best_path_ac]):
            best_path = best_path_bc
        elif all([ab in best_path_ac for ab in best_path_ab]):
            best_path = best_path_ac
        else:
            best_path = (best_path_ac[::-1]) + best_path_ab[1:]
    elif mu_ab + mu_bc <= mu_ac+mu_bc:
        if all([ab in best_path_bc for ab in best_path_ab]):
            best_path = best_path_bc
        elif all([bc in best_path_ab for bc in best_path_bc]):
            best_path = best_path_ab
        else:
            best_path = best_path_ab + (best_path_bc[1:])
    else:
        if all([ac in best_path_bc for ac in best_path_ac]):
            best_path = best_path_bc
        elif all([bc in best_path_ac for bc in best_path_bc]):
            best_path = best_path_ac
        else:
            best_path = best_path_bc + (best_path_ac[::-1][1:])

    return best_path

def _path_len(graph, path):
    path_len = 0
    for i in range(len(path)-1):
        path_len += graph.get_edge_weight(path[i], path[i+1])
    return path_len

def _bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # Initialize
    # pdb.set_trace()
    frontier_f, frontier_b = PriorityQueue(), PriorityQueue()
    explored_f, explored_b = set(), set()
    parents_f, parents_b = {}, {}
    best_path = []
    mu = float("inf") # least path length from start to goal
    if start is goal:
        return best_path
    frontier_f.append((0,start))
    frontier_b.append((0,goal))
    parents_f[start] = None
    parents_b[goal] = None
    while frontier_f.size() > 0 and frontier_b.size() > 0:
        top_f, top_b = frontier_f.top()[0], frontier_b.top()[0]
        if top_f + top_b >= mu:
            return best_path

        if top_f <= top_b:
            cost_f, curr_node_f = frontier_f.pop()
            # cost_f -= heuristic(graph, curr_node_f, goal)
            explored_f.add(curr_node_f)

            # check if goal is reached just using forward path
            if curr_node_f is goal:
                # cost is less than current best cost
                if cost_f < mu:
                    node = curr_node_f
                    path = []
                    while node is not None:
                        path.append(node)
                        node = parents_f[node]
                    path = path[::-1]
                    best_path = path
            # current node is in frontier of backward path
            elif curr_node_f in frontier_b:
                mu_new, best_path_new = _find_path(graph, curr_node_f, parents_f, parents_b)
                if mu_new < mu:
                    mu = mu_new
                    best_path = best_path_new

            # Expand forward path
            for neighbor in graph[curr_node_f]:
                # Check if neighbor is in backward search explored set
                if neighbor not in frontier_f and neighbor not in explored_f:
                    frontier_f.append((cost_f+graph.get_edge_weight(curr_node_f, neighbor), neighbor))
                    parents_f[neighbor] = curr_node_f
                elif neighbor in frontier_f:
                    frontier_cost_f, _ = frontier_f.remove(neighbor)
                    if frontier_cost_f > cost_f+graph.get_edge_weight(curr_node_f, neighbor):
                        frontier_f.append((cost_f+graph.get_edge_weight(curr_node_f, neighbor),neighbor))
                        parents_f[neighbor] = curr_node_f
                    else:
                        frontier_f.append((frontier_cost_f, neighbor))
        else:
            cost_b, curr_node_b = frontier_b.pop()
            # cost_b -= heuristic(graph, curr_node_b, start)
            explored_b.add(curr_node_b)

            # check if start is reached just using backward path
            if curr_node_b is start:
                # cost is less than current best cost
                if cost_b < mu:
                    node = curr_node_b
                    path = []
                    while node is not None:
                        path.append(node)
                        node = parents_b[node]
                    path = path[::-1]
                    best_path = path
            # current node is in frontier of forward path
            elif curr_node_b in frontier_f:
                mu_new, best_path_new = _find_path(graph,curr_node_b, parents_b, parents_f)
                if mu_new < mu:
                    mu = mu_new
                    best_path = best_path_new

            # Expand backward path
            for neighbor in graph[curr_node_b]:
                if neighbor not in frontier_b and neighbor not in explored_b:
                    frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor), neighbor))
                    parents_b[neighbor] = curr_node_b
                    # Check if neigh is in forward search frontier

                elif neighbor in frontier_b:
                    frontier_cost_b, _ = frontier_b.remove(neighbor)
                    if frontier_cost_b > cost_b+graph.get_edge_weight(curr_node_b, neighbor):
                        frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor),neighbor))
                        parents_b[neighbor] = curr_node_b
                    else:
                        frontier_b.append((frontier_cost_b, neighbor))
    return best_path

def _backtrack_path(end, parents):
    node = end
    path = []
    while node is not None:
        path.append(node)
        node = parents[node]
    return path

def _find_path(graph, common_node, parents_f, parents_b):
    forward_path_length, backward_path_length = 0, 0
    node_f = node_b = common_node
    path_f, path_b = [], []
    # trace path along front
    while node_f is not None:
        path_f.append(node_f)
        if parents_f[node_f] is not None:
            forward_path_length += graph.get_edge_weight(node_f, parents_f[node_f])
        node_f = parents_f[node_f]

    while node_b is not None:
        path_b.append(node_b)
        if parents_b[node_b] is not None:
            backward_path_length += graph.get_edge_weight(node_b, parents_b[node_b])
        node_b = parents_b[node_b]

    path_f = path_f[::-1]
    del path_b[0]
    best_path = path_f + path_b
    mu = forward_path_length+backward_path_length
    return mu, best_path

def _tri_heuristic(graph, heuristic, src, goal1, goal2):
    return min(heuristic(graph, src, goal1), heuristic(graph, src, goal2))

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    a, b, c = goals
    if a is b and b is c:
        return []
    if a is b:
        return _bidirectional_ucs(graph, a, c)
    if a is c:
        return _bidirectional_ucs(graph, a, b)
    if b is c:
        return _bidirectional_ucs(graph, a, b)
    frontier_a, frontier_b, frontier_c = PriorityQueue(), PriorityQueue(), PriorityQueue()
    explored_a, explored_b, explored_c = set(), set(), set()
    found_ab, found_bc, found_ac = False, False, False
    mu_ab, mu_bc, mu_ac = float("inf"), float("inf"), float("inf")
    best_path_ab, best_path_bc, best_path_ac = [], [], []
    parents_a, parents_b, parents_c = {}, {}, {}
    parents_a[a], parents_b[b], parents_c[c] = None, None, None
    frontier_a.append((0+_tri_heuristic(graph,heuristic,a,b,c),a))
    frontier_b.append((0+_tri_heuristic(graph,heuristic,b,a,c),b))
    frontier_c.append((0+_tri_heuristic(graph,heuristic,c,a,b),c))
    while not (found_ab and found_bc and found_ac):
        top_a, top_b, top_c = frontier_a.top(), frontier_b.top(), frontier_c.top()
        if top_a is None: top_a = [float("inf"), 0, '<invalid>']
        if top_b is None: top_b = [float("inf"), 0, '<invalid>']
        if top_c is None: top_c = [float("inf"), 0, '<invalid>']

        top_a[0] -= _tri_heuristic(graph,heuristic,a,b,c)
        top_b[0] -= _tri_heuristic(graph,heuristic,b,a,c)
        top_c[0] -= _tri_heuristic(graph,heuristic,c,a,b)

        if top_a[0] + top_b[0] >= mu_ab:
            found_ab = True

        if top_b[0] + top_c[0] >= mu_bc:
            found_bc = True

        if top_c[0] + top_a[0] >= mu_ac:
            found_ac = True

        # Explore a
        if top_a is not None and top_b is not None and top_c is not None and top_a[0] <= top_b[0] and top_a[0] <= top_c[0]:
            cost_a, curr_node_a = frontier_a.pop()
            cost_a -= _tri_heuristic(graph,heuristic,a,b,c)
            explored_a.add(curr_node_a)

            if found_ab and found_ac:
                continue

            if not found_ab and curr_node_a is b:
                # if current cost is less than best cost path
                if cost_a < mu_ab:
                    path_ab = _backtrack_path(curr_node_a, parents_a)
                    mu_ab, best_path_ab = cost_a, path_ab

            if not found_ab and curr_node_a in frontier_b:
                mu_ab_new, best_path_ab_new = _find_path(graph, curr_node_a, parents_a, parents_b)
                if mu_ab_new < mu_ab:
                    mu_ab, best_path_ab = mu_ab_new, best_path_ab_new

            if not found_ac and curr_node_a is c:
                # if current cost is less than the best cost path
                if cost_a < mu_ac:
                    path_ac = _backtrack_path(curr_node_a, parents_a)
                    mu_ac, best_path_ac = cost_a, path_ac

            if not found_ac and curr_node_a in frontier_c:
                mu_ac_new, best_path_ac_new = _find_path(graph, curr_node_a, parents_a, parents_c)
                if mu_ac_new < mu_ac:
                    mu_ac, best_path_ac = mu_ac_new, best_path_ac_new

            for neighbor in graph[curr_node_a]:
                if neighbor not in frontier_a and neighbor not in explored_a:
                    frontier_a.append((cost_a+graph.get_edge_weight(curr_node_a,neighbor)+_tri_heuristic(graph, heuristic, neighbor, b, c),neighbor))
                    parents_a[neighbor] = curr_node_a

                elif neighbor in frontier_a:
                    frontier_cost_a, _ = frontier_a.remove(neighbor)
                    if frontier_cost_a > cost_a+graph.get_edge_weight(curr_node_a, neighbor)+_tri_heuristic(graph, heuristic, neighbor, b, c):
                        frontier_a.append((cost_a+graph.get_edge_weight(curr_node_a, neighbor)+_tri_heuristic(graph, heuristic, neighbor, b, c), neighbor))
                        parents_a[neighbor] = curr_node_a
                    else:
                        frontier_a.append((frontier_cost_a, neighbor))

        # Explore b
        elif top_b[0] < top_a[0] and top_b[0] < top_c[0]:
            cost_b, curr_node_b = frontier_b.pop()
            cost_b -= _tri_heuristic(graph, heuristic, b, a, c)
            explored_b.add(curr_node_b)

            if found_bc and found_ab:
                continue

            if not found_ab and curr_node_b is a:
                if cost_b < mu_ab:
                    # pdb.set_trace()
                    path_ba = _backtrack_path(curr_node_b, parents_b)
                    path_ab = path_ba[::-1]
                    mu_ab, best_path_ab = cost_b, path_ab

            if not found_ab and curr_node_b in frontier_a:
                mu_ab_new, best_path_ab_new = _find_path(graph, curr_node_b, parents_a, parents_b)
                if mu_ab_new < mu_ab:
                    mu_ab, best_path_ab = mu_ab_new, best_path_ab_new

            if not found_bc and curr_node_b is c:
                if cost_b < mu_bc:
                    path_bc = _backtrack_path(curr_node_b, parents_b)
                    mu_bc, best_path_bc = cost_b, path_bc

            if not found_bc and curr_node_b in frontier_c:
                mu_bc_new, best_path_bc_new = _find_path(graph, curr_node_b, parents_b, parents_c)
                if mu_bc_new < mu_bc:
                    mu_bc, best_path_bc = mu_bc_new, best_path_bc_new

            for neighbor in graph[curr_node_b]:
                if neighbor not in frontier_b and neighbor not in explored_b:
                    frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor)+_tri_heuristic(graph,heuristic,neighbor,a,c), neighbor))
                    parents_b[neighbor] = curr_node_b
                elif neighbor in frontier_b:
                    frontier_cost_b, _ = frontier_b.remove(neighbor)
                    if frontier_cost_b > cost_b+graph.get_edge_weight(curr_node_b, neighbor)+_tri_heuristic(graph,heuristic,neighbor,a,c):
                        frontier_b.append((cost_b+graph.get_edge_weight(curr_node_b, neighbor)+_tri_heuristic(graph,heuristic,neighbor,a,c), neighbor))
                        parents_b[neighbor] = curr_node_b
                    else:
                        frontier_b.append((frontier_cost_b, neighbor))

        # Explore c
        else:
            cost_c, curr_node_c = frontier_c.pop()
            # pdb.set_trace()
            cost_c -= _tri_heuristic(graph,heuristic,c,a,b)
            explored_c.add(curr_node_c)

            if found_ac and found_bc:
                continue

            if not found_ac and curr_node_c is a:
                if cost_c < mu_ac:
                    path_ca = _backtrack_path(curr_node_c, parents_c)
                    path_ac = path_ca[::-1]
                    mu_ac, best_path_ac = cost_c, path_ac

            if not found_ac and curr_node_c in frontier_a:
                mu_ac_new, best_path_ac_new = _find_path(graph, curr_node_c, parents_a, parents_c)
                if mu_ac_new < mu_ac:
                    mu_ac, best_path_ac = mu_ac_new, best_path_ac_new

            if not found_bc and curr_node_c is b:
                if cost_c < mu_bc:
                    path_cb = _backtrack_path(curr_node_c, parents_c)
                    path_bc = path_cb[::-1]
                    mu_bc, best_path_bc = cost_c, path_bc

            if not found_bc and curr_node_c in frontier_b:
                mu_bc_new, best_path_bc_new = _find_path(graph, curr_node_c, parents_b, parents_c)
                if mu_bc_new < mu_bc:
                    mu_bc, best_path_bc = mu_bc_new, best_path_bc_new

            for neighbor in graph[curr_node_c]:
                if neighbor not in frontier_c and neighbor not in explored_c:
                    frontier_c.append((cost_c+graph.get_edge_weight(curr_node_c,neighbor)+_tri_heuristic(graph,heuristic,neighbor,a,b), neighbor))
                    parents_c[neighbor] = curr_node_c
                elif neighbor in frontier_c:
                    frontier_cost_c, _ = frontier_c.remove(neighbor)
                    if frontier_cost_c > cost_c+graph.get_edge_weight(curr_node_c,neighbor)+_tri_heuristic(graph,heuristic,neighbor,a,b):
                        frontier_c.append((cost_c+graph.get_edge_weight(curr_node_c,neighbor)+_tri_heuristic(graph,heuristic,neighbor,a,b), neighbor))
                        parents_c[neighbor] = curr_node_c
                    else:
                        frontier_c.append((frontier_cost_c,neighbor))

    # construct single path from 3 paths
    # pdb.set_trace()
    if mu_ab+mu_ac <= mu_ab+mu_bc and mu_ab + mu_ac <= mu_ac + mu_bc:
        if all([ac in best_path_bc for ac in best_path_ac]):
            best_path = best_path_bc
        elif all([ab in best_path_ac for ab in best_path_ab]):
            best_path = best_path_ac
        else:
            best_path = (best_path_ac[::-1]) + best_path_ab[1:]
    elif mu_ab + mu_bc <= mu_ac+mu_bc:
        if all([ab in best_path_bc for ab in best_path_ab]):
            best_path = best_path_bc
        elif all([bc in best_path_ab for bc in best_path_bc]):
            best_path = best_path_ab
        else:
            best_path = best_path_ab + (best_path_bc[1:])
    else:
        if all([ac in best_path_bc for ac in best_path_ac]):
            best_path = best_path_bc
        elif all([bc in best_path_ac for bc in best_path_bc]):
            best_path = best_path_ac
        else:
            best_path = best_path_bc + (best_path_ac[::-1][1:])

    return best_path


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Advait Koparkar"


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """

pass

# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to bonnie, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError



def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None

def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.node[v]["pos"][0]), math.radians(graph.node[v]["pos"][1]))
    goalLatLong = (math.radians(graph.node[goal]["pos"][0]), math.radians(graph.node[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
