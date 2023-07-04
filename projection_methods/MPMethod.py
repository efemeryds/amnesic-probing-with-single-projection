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


def get_direction(a, b):
    v_means = a - b
    v_means = v_means / np.linalg.norm(v_means)
    v_means = v_means.reshape(1, -1)
    return v_means


def assign_dep(x_train, y_train, value):
    dep_mask = y_train == value
    dep_x_train = x_train[dep_mask]
    return dep_x_train


def get_directions_for_fpos(x_train, y_train):
    labels_dictionary = {"label_0": assign_dep(x_train, y_train, 0), "label_1": assign_dep(x_train, y_train, 1),
                         "label_2": assign_dep(x_train, y_train, 2), "label_3": assign_dep(x_train, y_train, 3),
                         "label_4": assign_dep(x_train, y_train, 4), "label_5": assign_dep(x_train, y_train, 5),
                         "label_6": assign_dep(x_train, y_train, 6), "label_7": assign_dep(x_train, y_train, 7),
                         "label_8": assign_dep(x_train, y_train, 8), "label_9": assign_dep(x_train, y_train, 9),
                         "label_10": assign_dep(x_train, y_train, 10), "label_11": assign_dep(x_train, y_train, 11),
                         "label_12": assign_dep(x_train, y_train, 12), "label_13": assign_dep(x_train, y_train, 13),
                         "label_14": assign_dep(x_train, y_train, 14), "label_15": assign_dep(x_train, y_train, 15),
                         "label_16": assign_dep(x_train, y_train, 16), "label_17": assign_dep(x_train, y_train, 17),
                         "label_18": assign_dep(x_train, y_train, 18), "label_19": assign_dep(x_train, y_train, 19),
                         "label_20": assign_dep(x_train, y_train, 20), "label_21": assign_dep(x_train, y_train, 21),
                         "label_22": assign_dep(x_train, y_train, 22), "label_23": assign_dep(x_train, y_train, 23),
                         "label_24": assign_dep(x_train, y_train, 24), "label_25": assign_dep(x_train, y_train, 25),
                         "label_26": assign_dep(x_train, y_train, 26), "label_27": assign_dep(x_train, y_train, 27),
                         "label_28": assign_dep(x_train, y_train, 28), "label_29": assign_dep(x_train, y_train, 29),
                         "label_30": assign_dep(x_train, y_train, 30), "label_31": assign_dep(x_train, y_train, 31),
                         "label_32": assign_dep(x_train, y_train, 32), "label_33": assign_dep(x_train, y_train, 33),
                         "label_34": assign_dep(x_train, y_train, 34), "label_35": assign_dep(x_train, y_train, 35),
                         "label_36": assign_dep(x_train, y_train, 36), "label_37": assign_dep(x_train, y_train, 37),
                         "label_38": assign_dep(x_train, y_train, 38), "label_39": assign_dep(x_train, y_train, 39),
                         "label_40": assign_dep(x_train, y_train, 40), "label_41": assign_dep(x_train, y_train, 41),
                         "label_42": assign_dep(x_train, y_train, 42), "label_43": assign_dep(x_train, y_train, 43),
                         "label_44": assign_dep(x_train, y_train, 44)}

    v_means_1 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_1"], axis=0))
    v_means_2 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_2"], axis=0))
    v_means_3 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_3"], axis=0))
    v_means_4 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_4"], axis=0))
    v_means_5 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_5"], axis=0))
    v_means_6 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_6"], axis=0))
    v_means_7 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_7"], axis=0))
    v_means_8 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_8"], axis=0))
    v_means_9 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_9"], axis=0))
    v_means_10 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_10"], axis=0))
    v_means_11 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_11"], axis=0))
    v_means_12 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_12"], axis=0))
    v_means_13 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_13"], axis=0))
    v_means_14 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_14"], axis=0))
    v_means_15 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_15"], axis=0))
    v_means_16 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_16"], axis=0))
    v_means_17 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_17"], axis=0))
    v_means_18 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_18"], axis=0))
    v_means_19 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_19"], axis=0))
    v_means_20 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_20"], axis=0))
    v_means_21 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_21"], axis=0))
    v_means_22 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_22"], axis=0))
    v_means_23 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_23"], axis=0))
    v_means_24 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_24"], axis=0))
    v_means_25 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_25"], axis=0))
    v_means_26 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_26"], axis=0))
    v_means_27 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_27"], axis=0))
    v_means_28 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_28"], axis=0))
    v_means_29 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_29"], axis=0))
    v_means_30 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_30"], axis=0))
    v_means_31 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_31"], axis=0))
    v_means_32 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_32"], axis=0))
    v_means_33 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_33"], axis=0))
    v_means_34 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_34"], axis=0))
    v_means_35 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_35"], axis=0))
    v_means_36 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_36"], axis=0))
    v_means_37 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_37"], axis=0))
    v_means_38 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_38"], axis=0))
    v_means_39 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_39"], axis=0))
    v_means_40 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_40"], axis=0))
    v_means_41 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_41"], axis=0))
    v_means_42 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_42"], axis=0))
    v_means_43 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_43"], axis=0))
    v_means_44 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_44"], axis=0))

    W = [v_means_1, v_means_2,
         v_means_3, v_means_4,
         v_means_5, v_means_6,
         v_means_7, v_means_8,
         v_means_9, v_means_10,
         v_means_11, v_means_12,
         v_means_13, v_means_14,
         v_means_15, v_means_16,
         v_means_17, v_means_18,
         v_means_19, v_means_20,
         v_means_21, v_means_22,
         v_means_23, v_means_24,
         v_means_25, v_means_26,
         v_means_27, v_means_28,
         v_means_29, v_means_30,
         v_means_31, v_means_32,
         v_means_33, v_means_34,
         v_means_35, v_means_36,
         v_means_37, v_means_38,
         v_means_39, v_means_40,
         v_means_41, v_means_42,
         v_means_43, v_means_44]

    return W


def get_directions_for_dependency(x_train, y_train):
    labels_dictionary = {"label_0": assign_dep(x_train, y_train, 0), "label_1": assign_dep(x_train, y_train, 1),
                         "label_2": assign_dep(x_train, y_train, 2), "label_3": assign_dep(x_train, y_train, 3),
                         "label_4": assign_dep(x_train, y_train, 4), "label_5": assign_dep(x_train, y_train, 5),
                         "label_6": assign_dep(x_train, y_train, 6), "label_7": assign_dep(x_train, y_train, 7),
                         "label_8": assign_dep(x_train, y_train, 8), "label_9": assign_dep(x_train, y_train, 9),
                         "label_10": assign_dep(x_train, y_train, 10), "label_11": assign_dep(x_train, y_train, 11),
                         "label_12": assign_dep(x_train, y_train, 12), "label_13": assign_dep(x_train, y_train, 13),
                         "label_14": assign_dep(x_train, y_train, 14), "label_15": assign_dep(x_train, y_train, 15),
                         "label_16": assign_dep(x_train, y_train, 16), "label_17": assign_dep(x_train, y_train, 17),
                         "label_18": assign_dep(x_train, y_train, 18), "label_19": assign_dep(x_train, y_train, 19),
                         "label_20": assign_dep(x_train, y_train, 20), "label_21": assign_dep(x_train, y_train, 21),
                         "label_22": assign_dep(x_train, y_train, 22), "label_23": assign_dep(x_train, y_train, 23),
                         "label_24": assign_dep(x_train, y_train, 24), "label_25": assign_dep(x_train, y_train, 25),
                         "label_26": assign_dep(x_train, y_train, 26), "label_27": assign_dep(x_train, y_train, 27),
                         "label_28": assign_dep(x_train, y_train, 28), "label_29": assign_dep(x_train, y_train, 29),
                         "label_30": assign_dep(x_train, y_train, 30), "label_31": assign_dep(x_train, y_train, 31),
                         "label_32": assign_dep(x_train, y_train, 32), "label_33": assign_dep(x_train, y_train, 33),
                         "label_34": assign_dep(x_train, y_train, 34), "label_35": assign_dep(x_train, y_train, 35),
                         "label_36": assign_dep(x_train, y_train, 36), "label_37": assign_dep(x_train, y_train, 37),
                         "label_38": assign_dep(x_train, y_train, 38), "label_39": assign_dep(x_train, y_train, 39),
                         "label_40": assign_dep(x_train, y_train, 40)}

    v_means_1 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_1"], axis=0))
    v_means_2 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_2"], axis=0))
    v_means_3 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_3"], axis=0))
    v_means_4 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_4"], axis=0))
    v_means_5 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_5"], axis=0))
    v_means_6 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_6"], axis=0))
    v_means_7 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_7"], axis=0))
    v_means_8 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_8"], axis=0))
    v_means_9 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_9"], axis=0))
    v_means_10 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_10"], axis=0))
    v_means_11 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_11"], axis=0))
    v_means_12 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_12"], axis=0))
    v_means_13 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_13"], axis=0))
    v_means_14 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_14"], axis=0))
    v_means_15 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_15"], axis=0))
    v_means_16 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_16"], axis=0))
    v_means_17 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_17"], axis=0))
    v_means_18 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_18"], axis=0))
    v_means_19 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_19"], axis=0))
    v_means_20 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_20"], axis=0))
    v_means_21 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_21"], axis=0))
    v_means_22 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_22"], axis=0))
    v_means_23 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_23"], axis=0))
    v_means_24 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_24"], axis=0))
    v_means_25 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_25"], axis=0))
    v_means_26 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_26"], axis=0))
    v_means_27 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_27"], axis=0))
    v_means_28 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_28"], axis=0))
    v_means_29 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_29"], axis=0))
    v_means_30 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_30"], axis=0))
    v_means_31 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_31"], axis=0))
    v_means_32 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_32"], axis=0))
    v_means_33 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_33"], axis=0))
    v_means_34 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_34"], axis=0))
    v_means_35 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_35"], axis=0))
    v_means_36 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_36"], axis=0))
    v_means_37 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_37"], axis=0))
    v_means_38 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_38"], axis=0))
    v_means_39 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_39"], axis=0))
    v_means_40 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_40"], axis=0))

    W = [v_means_1, v_means_2,
         v_means_3, v_means_4,
         v_means_5, v_means_6,
         v_means_7, v_means_8,
         v_means_9, v_means_10,
         v_means_11, v_means_12,
         v_means_13, v_means_14,
         v_means_15, v_means_16,
         v_means_17, v_means_18,
         v_means_19, v_means_20,
         v_means_21, v_means_22,
         v_means_23, v_means_24,
         v_means_25, v_means_26,
         v_means_27, v_means_28,
         v_means_29, v_means_30,
         v_means_31, v_means_32,
         v_means_33, v_means_34,
         v_means_35, v_means_36,
         v_means_37, v_means_38,
         v_means_39, v_means_40]
    return W


def get_directions_for_cpos(x_train, y_train):
    labels_dictionary = {"label_0": assign_dep(x_train, y_train, 0), "label_1": assign_dep(x_train, y_train, 1),
                         "label_2": assign_dep(x_train, y_train, 2), "label_3": assign_dep(x_train, y_train, 3),
                         "label_4": assign_dep(x_train, y_train, 4), "label_5": assign_dep(x_train, y_train, 5),
                         "label_6": assign_dep(x_train, y_train, 6), "label_7": assign_dep(x_train, y_train, 7),
                         "label_8": assign_dep(x_train, y_train, 8), "label_9": assign_dep(x_train, y_train, 9),
                         "label_10": assign_dep(x_train, y_train, 10), "label_11": assign_dep(x_train, y_train, 11)}

    v_means_1 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_1"], axis=0))
    v_means_2 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_2"], axis=0))
    v_means_3 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_3"], axis=0))
    v_means_4 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_4"], axis=0))
    v_means_5 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_5"], axis=0))
    v_means_6 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_6"], axis=0))
    v_means_7 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_7"], axis=0))
    v_means_8 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_8"], axis=0))
    v_means_9 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                              np.mean(labels_dictionary["label_9"], axis=0))
    v_means_10 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_10"], axis=0))
    v_means_11 = get_direction(np.mean(labels_dictionary["label_0"], axis=0),
                               np.mean(labels_dictionary["label_11"], axis=0))

    W = [v_means_1, v_means_2,
         v_means_3, v_means_4,
         v_means_5, v_means_6,
         v_means_7, v_means_8,
         v_means_9, v_means_10,
         v_means_11]
    return W


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

