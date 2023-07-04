# Code from https://github.com/yanaiela/amnesic_probing/tree/a96e3067a9f9918099015a7173c94830584bbe61
import gc
from evaluation.PytorchClassifier import PytorchClassifier
from torch.nn.functional import kl_div
from tqdm import tqdm
import scipy
import torch
import numpy as np
from typing import List
from collections import defaultdict, Counter
from transformers import BertTokenizer, BertForMaskedLM
import pickle

np.random.seed(10)


def load_data(path):
    vecs = np.load(f"{path}/last_vec.npy", allow_pickle=True)
    vecs = np.array([x[1:-1] for x in vecs])

    with open(f"{path}/tokens.pickle", 'rb') as f:
        labels = pickle.load(f)

    return vecs, labels


def load_labels(labels_file):
    with open(labels_file, 'rb') as f:
        rebias_labels = pickle.load(f)

    return rebias_labels


def flatten_list(input_list):
    return [x for x_list in input_list for x in x_list]


def flatten_label_list(input_list, labels_list):
    flat_list = flatten_list(input_list)
    return np.array([labels_list.index(y) for y in flat_list]).flatten()


def flatten_tokens(all_vectors, all_labels, lm_tokenizer):
    x = np.array(flatten_list(all_vectors))
    y = np.array(
        [label for sentence_y in all_labels for label in
         lm_tokenizer.convert_tokens_to_ids(sentence_y)]).flatten()
    return x, y


def get_lm_vals(model_name):
    lm_model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    out_embed = lm_model.cls.predictions.decoder.weight.detach().cpu().numpy()
    bias = lm_model.cls.predictions.decoder.bias.detach().cpu().numpy()
    return lm_model, tokenizer, out_embed, bias


def data_projection(x, projection_matrix):
    return x.dot(projection_matrix)


def most_probable_label(words, labels):
    words_labels = defaultdict(list)

    for word, label in zip(words, labels):
        words_labels[word].append(label)

    most_probable_label_per_word = {}
    for word, label_list in words_labels.items():
        most_probable_label_per_word[word] = Counter(label_list).most_common(1)[0][0]
    return most_probable_label_per_word


def define_network(W: np.ndarray, b: np.ndarray, projection_mat: np.ndarray = None, device: str = 'cpu'):
    embedding_net = torch.nn.Linear(in_features=W.shape[1], out_features=W.shape[0])
    embedding_net.weight.data = torch.tensor(W)
    embedding_net.bias.data = torch.tensor(b)

    if projection_mat is not None:
        projection_net = torch.nn.Linear(in_features=projection_mat.shape[1],
                                         out_features=projection_mat.shape[0],
                                         bias=False)
        projection_net.weight.data = torch.tensor(projection_mat, dtype=torch.float)
        for p in projection_net.parameters():
            p.requires_grad = False
        word_prediction_net = torch.nn.Sequential(projection_net, embedding_net)

    else:
        word_prediction_net = torch.nn.Sequential(embedding_net)

    net = PytorchClassifier(word_prediction_net, device=device)

    del embedding_net
    del word_prediction_net
    gc.collect()
    return net


def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

    w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

    return P_W


def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param input_dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)

    return P


def dropout_control(x, n_coord):
    all_indices = np.array(range(x.shape[1]))
    # shuffle - this function only shuffles the array along the first axis of a multi-dimensional array.
    # The order of sub-arrays is changed but their contents remains the same.
    np.random.shuffle(all_indices)
    # shuffled column indexes
    random_indices = all_indices[:n_coord]
    x_rand_dropout = x.copy()
    x_rand_dropout[:, random_indices] = 0
    # x_rand_dropout[:, :] = 0
    return x_rand_dropout


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directions
    (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P


def rand_direction_control(x, n_coord):
    dim = x.shape[1]
    # creating random directions (vectors) within the range of -0.5 : 0.5
    rand_directions = [np.random.rand(1, dim) - 0.5 for _ in range(n_coord)]

    # finding the null-space of random directions
    rand_direction_p = debias_by_specific_directions(rand_directions, dim)

    # and projecting the original data into that space (to remove random directions)
    x_rand_direction = rand_direction_p.dot(x.T).T
    return x_rand_direction


def get_lm_softmax_gpu(w, b, x, y, device: str):
    network = define_network(w, b, device=device)
    distribution = network.get_probs(x, y)
    del network
    gc.collect()
    return distribution


def dkl_gpu(w, b, x_orig, x_diff, y, plain_probs: np.ndarray = None, device: str = 'cpu'):
    print("dkl_gpu - get probs")
    if plain_probs is None:
        probs = get_lm_softmax_gpu(w, b, x_orig, y, device=device)
    else:
        probs = plain_probs
    probs_diff = get_lm_softmax_gpu(w, b, x_diff, y, device=device)

    print("dkl_gpu - get all distributions")
    all_dkl = []
    for batch_prob, batch_prob_diff in tqdm(zip(probs, probs_diff)):
        batch_dkl = kl_div(torch.tensor(batch_prob_diff).float().log(),
                           torch.tensor(batch_prob).float(), reduction='none').sum(axis=1).cpu().numpy()

        # batch_dkl = kl_div(torch.tensor(batch_prob_diff).float().to(device).log(),
        #                    torch.tensor(batch_prob).float().to(device), reduction='none').sum(axis=1).cpu().numpy()

        all_dkl.extend(batch_dkl)

        del batch_dkl
        gc.collect()

    dkl_mean = np.mean(all_dkl)

    return dkl_mean, probs
