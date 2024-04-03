import gc
import numpy as np
import scipy
import sys
from typing import List

np.random.seed(10)


class MPMethod:
    def __init__(self, task, directions):
        self.task = task
        self.W = directions

    def get_rowspace_projection(self, W: np.ndarray) -> np.ndarray:
        """
        :param W: the matrix over its nullspace to project
        :return: the projection matrix over the rowspace
        """

        if np.allclose(W, 0):
            w_basis = np.zeros_like(W.T)
        else:
            w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

        P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

        del w_basis
        del W
        gc.collect()
        return P_W

    def debias_by_specific_directions(self, directions: List[np.ndarray], input_dim: int):
        """
        the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
        :param directions: list of vectors, as numpy arrays.
        :param input_dim: dimensionality of the vectors.
        """

        rowspace_projections = []

        print("Prepare W")
        W = np.empty((0, input_dim), dtype="float64")

        print("Stack W")
        for v in directions:
            W = np.vstack((W, v))

        print("Get rowspace projection Q")
        Q = self.get_rowspace_projection(W)
        rowspace_projections.append(Q)

        print("Calculate P")
        print("Size of q: ", sys.getsizeof(Q))
        print("Input dim: ", input_dim)
        P = np.eye(input_dim, dtype="float64") - Q

        del Q
        del W
        del directions
        gc.collect()
        return P, rowspace_projections

    def mean_projection_method(self):
        P, rowspace_projs = self.debias_by_specific_directions(self.W, input_dim=768)
        return P


def assign_dep(x_train, y_train, value):
    dep_mask = y_train == value
    dep_x_train = x_train[dep_mask]
    return dep_x_train


def get_labels(x_train, y_train):
    labels_list = []
    unique_labels = list(set(y_train))
    for i in unique_labels:
        labels_list.append(assign_dep(x_train, y_train, i))

    return labels_list


def get_directions(input_x_train, input_y_train):
    input_labels = get_labels(input_x_train, input_y_train)

    weights = []

    for i, value in enumerate(input_labels):
        tmp_list = input_labels.copy()
        tmp_list.pop(i)

        target_sum = np.mean(value, axis=0)
        rest_of_sums = [np.mean(x, axis=0) for x in tmp_list]
        v_means = target_sum - np.mean(rest_of_sums, axis=0)
        v_means = v_means / np.linalg.norm(v_means)
        v_means = v_means.reshape(1, -1)

        weights.append(v_means)

    return weights
