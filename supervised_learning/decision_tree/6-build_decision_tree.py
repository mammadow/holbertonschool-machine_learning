#!/usr/bin/env python3
"""Module for building a decision tree from scratch."""
import numpy as np


class Node:
    """Represents an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node with feature, threshold, children, depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Returns the maximum depth in the subtree rooted at this node."""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts all nodes (or only leaves) in the subtree."""
        self_count = 0 if only_leaves else 1
        return (self_count
                + self.left_child.count_nodes_below(
                    only_leaves=only_leaves)
                + self.right_child.count_nodes_below(
                    only_leaves=only_leaves))

    def left_child_add_prefix(self, text):
        """Adds the left-child prefix with vertical bar to each line."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds the right-child prefix with spaces to each line."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns a string representation of the subtree."""
        if self.is_root:
            node_str = (f"root [feature={self.feature},"
                        f" threshold={self.threshold}]\n")
        else:
            node_str = (f"-> node [feature={self.feature},"
                        f" threshold={self.threshold}]\n")
        left_str = self.left_child_add_prefix(str(self.left_child))
        right_str = self.right_child_add_prefix(str(self.right_child))
        return node_str + left_str + right_str

    def get_leaves_below(self):
        """Returns all leaves in the subtree rooted at this node."""
        return (self.left_child.get_leaves_below()
                + self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively computes and attaches bounds to all nodes below."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            if child is self.left_child:
                child.lower[self.feature] = max(
                    self.lower.get(self.feature, -np.inf),
                    self.threshold
                )
            else:
                child.upper[self.feature] = min(
                    self.upper.get(self.feature, np.inf),
                    self.threshold
                )

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Builds the indicator function from lower and upper bounds."""
        def is_large_enough(x):
            """Returns True for each individual above all lower bounds."""
            return np.all(
                np.array([
                    np.greater(x[:, key], self.lower[key])
                    for key in self.lower
                ]),
                axis=0
            )

        def is_small_enough(x):
            """Returns True for each individual below all upper bounds."""
            return np.all(
                np.array([
                    np.less_equal(x[:, key], self.upper[key])
                    for key in self.upper
                ]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Returns prediction for one individual by traversing the tree."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def __str__(self):
        """Returns a string representation of this leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Returns a list containing only this leaf."""
        return [self]

    def update_bounds_below(self):
        """Does nothing since leaves have no children to update."""
        pass

    def pred(self, x):
        """Returns the leaf value as prediction for one individual."""
        return self.value


class Decision_Tree():
    """Represents a decision tree classifier."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a Decision_Tree with hyperparameters and root."""
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

    def __str__(self):
        """Returns a string representation of the full decision tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Returns the list of all leaves in the decision tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Triggers recursive bounds computation from the root."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Builds the vectorized predict function using leaf indicators."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )

    def pred(self, x):
        """Returns prediction for one individual using tree traversal."""
        return self.root.pred(x)
