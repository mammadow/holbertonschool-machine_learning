#!/usr/bin/env python3
"""Module for building a decision tree from scratch."""
import numpy as np


class Node:
    """Represents an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node with its feature, threshold, children, and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Returns the maximum depth found in the subtree rooted at this node."""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts all nodes (or only leaves) in the subtree rooted at this node."""
        self_count = 0 if only_leaves else 1
        return (self_count
                + self.left_child.count_nodes_below(only_leaves=only_leaves)
                + self.right_child.count_nodes_below(only_leaves=only_leaves))


class Leaf(Node):
    """Represents a leaf (terminal) node in a decision tree."""

    def __init__(self, value, depth=None):
        """Initializes a Leaf with its prediction value and depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 since a leaf always counts as one node."""
        return 1


class Decision_Tree():
    """Represents a decision tree classifier."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a Decision_Tree with its hyperparameters and root node."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the number of nodes (or only leaves) in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)
