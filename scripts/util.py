import numpy as np


def compare(query, feature, threshold, sign=True):
    if sign:
        return query[feature] > threshold
    else:
        return not query[feature] > threshold


class LocalDecisionStump:

    def __init__(self, feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs):

        self.feature = feature
        self.threshold = threshold
        self.left_val = left_val
        self.right_val = right_val
        self.a_features = a_features
        self.a_thresholds = a_thresholds
        self.a_signs = a_signs

    def __call__(self, query):

        in_node = all([compare(query, f, t, g) for f, t, g in zip(self.a_features,
                                                                  self.a_thresholds,
                                                                  self.a_signs)])
        if not in_node:
            return 0
        else:
            is_right = compare(query, self.feature, self.threshold)
            if is_right:
                return self.right_val
            else:
                return self.left_val

    def __repr__(self):

        return f"LocalDecisionStump(feature={self.feature}, threshold={self.threshold}, left_val={self.left_val}, " \
               f"right_val={self.right_val}, a_features={self.a_features}, a_thresholds={self.a_thresholds}, " \
               f"a_signs={self.a_signs})"



def make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize=False):
    """
    Create a single localized decision stump corresponding to a node

    :param node_no:
    :param tree_struct:
    :param parent_stump:
    :param is_right_child:
    :param normalize:
    :return:
    """
    if parent_stump is None:  # If root node
        a_features = []
        a_thresholds = []
        a_signs = []
    else:
        a_features = parent_stump.a_features + [parent_stump.feature]
        a_thresholds = parent_stump.a_thresholds + [parent_stump.threshold]
        a_signs = parent_stump.a_signs + [is_right_child]

    feature = tree_struct.feature[node_no]
    threshold = tree_struct.threshold[node_no]

    if not normalize:
        return LocalDecisionStump(feature, threshold, -1, 1, a_features, a_thresholds, a_signs)
    else:
        # parent_size = tree_struct.n_node_samples[node_no]
        left_child = tree_struct.children_left[node_no]
        right_child = tree_struct.children_right[node_no]
        left_size = tree_struct.n_node_samples[left_child]
        right_size = tree_struct.n_node_samples[right_child]
        left_val = - np.sqrt(right_size / left_size)
        right_val = np.sqrt(left_size / right_size)
        return LocalDecisionStump(feature, threshold, left_val, right_val, a_features, a_thresholds, a_signs)


def make_stumps(tree_struct, normalize=False):
    """
    Take sklearn decision tree structure and create a collection of local
    decision stump objects

    :param tree_struct:
    :param normalize:
    :return: list of stumps
    """
    stumps = []

    def make_stump_iter(node_no, tree_struct, parent_stump, is_right_child, normalize, stumps):

        new_stump = make_stump(node_no, tree_struct, parent_stump, is_right_child, normalize)
        stumps.append(new_stump)
        left_child = tree_struct.children_left[node_no]
        right_child = tree_struct.children_right[node_no]
        if tree_struct.feature[left_child] != -2:  # is not leaf
            make_stump_iter(left_child, tree_struct, new_stump, False, normalize, stumps)
        if tree_struct.feature[right_child] != -2:  # is not leaf
            make_stump_iter(right_child, tree_struct, new_stump, True, normalize, stumps)

    make_stump_iter(0, tree_struct, None, None, normalize, stumps)

    return stumps