import numpy as np
import pdb
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = DecisionNode(None, None, lambda x : x[0] == 1)
    n2 = DecisionNode(None, None, lambda x: x[1] == 0)
    n3 = DecisionNode(None, None, lambda x: x[3] == 1)
    n4 = DecisionNode(None, None, lambda x: x[2] == x[3])
    one = DecisionNode(None, None, None, 1)
    zero = DecisionNode(None, None, None, 0)

    decision_tree_root.left = one
    decision_tree_root.right = n2
    n2.left = n3
    n2.right = n4
    n3.left = one
    n3.right = zero
    n4.left = one
    n4.right = zero

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    # pdb.set_trace()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    tn = np.sum((classifier_output == 0) * (true_labels == 0))
    tp = np.sum((classifier_output == 1) * (true_labels == 1))
    fn = np.sum((classifier_output == 0) * (true_labels == 1))
    fp = np.sum((classifier_output == 1) * (true_labels == 0))

    return [[tp, fn], [fp, tn]]

def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    tp = np.sum((classifier_output == 1) * (true_labels == 1))
    fp = np.sum((classifier_output == 1) * (true_labels == 0))

    return tp / (tp+fp)


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    tp = np.sum((classifier_output == 1) * (true_labels == 1))
    fn = np.sum((classifier_output == 0) * (true_labels == 1))

    return tp / (tp+fn)

def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    return np.sum(classifier_output == true_labels) / true_labels.shape[0]

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    class_vector = np.array(class_vector)
    p0 = np.sum(class_vector == 0) / class_vector.shape[0]
    p1 = np.sum(class_vector == 1) / class_vector.shape[0]
    return 1.0 - p0**2 - p1**2

def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    # def entropy(vec):
    #     c0, c1 = np.sum(vec == 0), np.sum(vec == 1)
    #     p = c0/(c0+c1)
    #     if p == 1 or p == 0: return 0.0
    #     return -p*np.log2(p)-(1-p)*np.log2(1-p)

    # pdb.set_trace()
    previous_classes = np.array(previous_classes)
    # p_entropy = entropy(previous_classes)
    p_entropy = gini_impurity(previous_classes)
    rem = 0
    for c in current_classes:
        c = np.array(c)
        if c.size == 0: continue
        rem += gini_impurity(c)*(c.shape[0]/previous_classes.shape[0])
    return p_entropy - rem

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float("inf")):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        def mode(classes):
            hmap = {}
            mode, max_freq = None, -1
            for c in classes:
                if c not in hmap: hmap[c] = 0
                hmap[c] += 1
            for c in hmap:
                if hmap[c] > max_freq:
                    max_freq = hmap[c]
                    mode = c
            return mode

        # pdb.set_trace()
        # Check base cases:
        if classes.size == 0:
            return None

        if np.unique(classes).size == 1 or depth == self.depth_limit:
            return DecisionNode(None, None, None, mode(classes))

        # Find best feature to split data:
        alpha_best, alpha_best_g, alpha_best_split = -1, float("-inf"), float("-inf")
        for alpha_idx, alpha in enumerate(features.T):
            # pdb.set_trace()
            alpha_min_val, alpha_max_val = np.min(alpha), np.max(alpha)
            if alpha_min_val == alpha_max_val: continue
            best_g, best_split = float("-inf"), None
            splits = np.linspace(alpha_min_val+0.001, alpha_max_val, num=100)
            for split in splits:
                n_idx, p_idx = np.where(alpha <= split), np.where(alpha > split)
                # pos_samples, neg_samples = alpha[p_idx], alpha[n_idx]
                pos_classes, neg_classes = classes[p_idx], classes[n_idx]
                g = gini_gain(classes, [pos_classes, neg_classes])
                if g > best_g:
                    best_g = g
                    best_split = split
            if best_g > alpha_best_g:
                alpha_best_g = best_g
                alpha_best = alpha_idx
                alpha_best_split = best_split
        # Split on feature alpha_best, with threshold alpha_best_split
        n_idx, p_idx = np.where(features[:, alpha_best] <= alpha_best_split), np.where(features[:, alpha_best] > alpha_best_split)
        n_features, n_classes = features[n_idx], classes[n_idx]
        p_features, p_classes = features[p_idx], classes[p_idx]
        # Build children
        n_node = self.__build_tree__(n_features, n_classes, depth+1)
        p_node = self.__build_tree__(p_features, p_classes, depth+1)
        # if p_node is None or n_node is None: pdb.set_trace()
        # Return root
        return DecisionNode(n_node, p_node, lambda feature: feature[alpha_best] < alpha_best_split)

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = []

        # TODO: finish this.
        for idx, feature in enumerate(features):
            class_labels.append(self.root.decide(feature))
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    f, c = dataset
    N, D = f.shape
    idx = np.random.permutation(N)
    f, c = f[idx], c[idx]
    # fs, cs = np.array_split(f, k), np.array_split(c, k)
    folds = []
    test_size = N // k

    for fold in range(k):
        # pdb.set_trace()
        test_idx = np.arange(start=(fold + (k-1))%k * test_size, stop=(fold + (k-1))%k * test_size+test_size)
        test_feats, test_class = f[test_idx, :], c[test_idx]
        training_feats, training_class = np.delete(f, test_idx, axis=0), np.delete(c, test_idx, axis=0)
        # for i in range(k-1):
        #     if tr_idx >= k: tr_idx = 0
        #     training_feats = np.append(training_feats, fs[tr_idx], axis=0)
        #     training_class = np.append(training_class, cs[tr_idx], axis=0)
        # test_feats, test_class = fs[(fold+(k-1))%k], cs[(fold+(k-1))%k]
        folds.append(((training_feats, training_class),(test_feats,test_class)))

    return folds




class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.feat_map = {}

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # pdb.set_trace()
        # pdb.set_trace()
        for tree_idx in range(self.num_trees):
            tree = DecisionTree(self.depth_limit)
            N, D = features.shape
            ex_idx = np.random.choice(N, size=int(N * self.example_subsample_rate), replace=False)
            f_idx = np.random.choice(D, size=int(D * self.attr_subsample_rate), replace=False)
            self.feat_map[tree_idx] = f_idx
            feats, labels = features[ex_idx,:][:,f_idx], classes[ex_idx]
            tree.fit(feats, labels)
            self.trees.append(tree)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """
        def mode(classes):
            hmap = {}
            mode, max_freq = None, -1
            for c in classes:
                if c not in hmap: hmap[c] = 0
                hmap[c] += 1
            for c in hmap:
                if hmap[c] > max_freq:
                    max_freq = hmap[c]
                    mode = c
            return mode

        N, D = features.shape
        classifications = np.zeros((N, self.num_trees))
        # pdb.set_trace()
        ret = np.ones((N,))
        for t_idx, tree in enumerate(self.trees):
            feat = features[:, self.feat_map[t_idx]]
            classifications[:, t_idx] = np.array(tree.classify(feat))
        for n in range(N):
            ret[n] = mode(classifications[n, :])
        return ret

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees=15, depth_limit=15, example_subsample_rate=0.7, attr_subsample_rate=0.7):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.mean_feat, self.std_feat = None, None
        self.feat_map = {}
        # print(num_trees, depth_limit, example_subsample_rate, attr_subsample_rate)

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # pdb.set_trace()
        # pdb.set_trace()
        self.mean_feat = np.mean(features, axis=0)
        self.std_feat = np.std(features, axis=0)
        features = (features - self.mean_feat) / self.std_feat

        for tree_idx in range(self.num_trees):
            tree = DecisionTree(self.depth_limit)
            N, D = features.shape
            ex_idx = np.random.choice(N, size=int(N * self.example_subsample_rate), replace=False)
            f_idx = np.random.choice(D, size=int(D * self.attr_subsample_rate), replace=False)
            self.feat_map[tree_idx] = f_idx
            feats, labels = features[ex_idx,:][:,f_idx], classes[ex_idx]
            tree.fit(feats, labels)
            self.trees.append(tree)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """
        def mode(classes):
            hmap = {}
            mode, max_freq = None, -1
            for c in classes:
                if c not in hmap: hmap[c] = 0
                hmap[c] += 1
            for c in hmap:
                if hmap[c] > max_freq:
                    max_freq = hmap[c]
                    mode = c
            return mode

        features = (features - self.mean_feat) / self.std_feat
        N, D = features.shape
        classifications = np.zeros((N, self.num_trees))
        # pdb.set_trace()
        ret = np.ones((N,))
        for t_idx, tree in enumerate(self.trees):
            feat = features[:, self.feat_map[t_idx]]
            classifications[:, t_idx] = np.array(tree.classify(feat))
        for n in range(N):
            ret[n] = mode(classifications[n, :])
        return ret


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        return data*data + data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        row_sum = np.sum(data[0:100, :], axis=1)
        max_idx = np.argmax(row_sum)
        return (row_sum[max_idx], max_idx)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        data = data.flatten()
        data = data[data > 0]
        un, ct = np.unique(data, return_counts=True)
        return list(zip(un,ct))

def return_your_name():
    # return your name
    # TODO: finish this
    return "Advait Koparkar"
