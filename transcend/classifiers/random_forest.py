import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from transcend.classifiers.ncm_classifier import NCMClassifier

eps = np.finfo(np.float64).eps


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
                    leaves[..., np.newaxis]
                    == t_leaves[:, example_idx:end_idx][np.newaxis, ...],
                    axis=1,
                )
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

        def single_ncm(i):
            avg_prox_diff = np.mean(np.sort(proximity[i, self.y_train != y[i]])[-10:])
            avg_prox_eq = np.mean(np.sort(proximity[i, self.y_train == y[i]])[-10:])
            return avg_prox_diff / (avg_prox_eq + eps)

        return np.array([single_ncm(i) for i in tqdm(range(len(y)))])
