import numpy as np
from sklearn.ensemble import RandomForestClassifier

from transcend.classifiers.ncm_classifier import \
    NCMClassifier


class RandomForestNCMClassifier(NCMClassifier, RandomForestClassifier):

    def __init__(self, **kwargs):
        RandomForestClassifier.__init__(self, **kwargs)
        NCMClassifier.__init__(self)
        self.proximity = None
        self.train_leaves = None
        self.X_train = None
        self.y_train = None

    def __compute_proximity(self, leaves, step_size=100):
        """
        Computes the proximity between each pair of examples.
        :param leaves: A matrix of shape [num_example, num_tree] where the value [i,j] is
          the index of the leaf reached by example "i" in the tree "j".
        :param step_size: Size of the block of examples for the computation of the
          proximity. Does not impact the results.
        :return: The example pair-wise proximity matrix of shape [n,n] with "n" the number of
        examples.
        """
        example_idx = 0
        num_examples = len(self.X_train)

        t_leaves = np.transpose(self.train_leaves)
        proximities = []

        # Instead of computing the proximity in between all the examples at the same
        # time, we compute the similarity in blocks of "step_size" examples. This
        # makes the code more efficient with the the numpy broadcast.
        while example_idx < num_examples:
            end_idx = min(example_idx + step_size, num_examples)
            proximities.append(
                np.mean(
                    leaves[..., np.newaxis] == t_leaves[:,
                                               example_idx:end_idx][np.newaxis,
                    ...],
                    axis=1)
            )
            example_idx = end_idx
        return np.concatenate(proximities, axis=1)

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.train_leaves = self.apply(X_train)
        self.proximity = self.__compute_proximity(self.train_leaves)

    def ncm(self, X, y):
        if np.array_equal(X, self.X_train):
            proximity = self.proximity
        else:
            leaves = self.apply(X)
            proximity = self.__compute_proximity(leaves)

        return np.array([np.mean(np.sort(proximity[i, self.y_train != y[i]])[-10:]) /
                         np.mean(np.sort(proximity[i, self.y_train == y[i]])[-10:])
                         for i in range(len(y))])

    # def per_label_ncm(self, args):
    #     args = yy, idx
    #     return np.array([np.mean(np.sort(self.proximity_conf[idx, self.y_train != yy[i]])[-10:]) /
    #                      np.mean(np.sort(self.proximity_conf[idx, self.y_train == yy[i]])[-10:])
    #                      for i in range(len(yy))])
    #
    # def ncm_conf(self, X):
    #     labels_excluded = self.predict(X)
    #     unique_labels = np.unique(self.y_train)
    #
    #     y = np.array([unique_labels[unique_labels != labels_excluded[i]]
    #                   for i in range(len(labels_excluded))])
    #
    #     if X.equals(self.X_train):
    #         self.proximity_conf = self.proximity_matrix
    #     else:
    #         leaves = self.apply(X)
    #         self.proximity_conf = self.prox_ncm(leaves)
    #
    #     with Pool(16) as p:
    #         ncms = p.map(self.per_label_ncm, [(yy, i) for i, yy in enumerate(y)])
    #
    #     return ncms
