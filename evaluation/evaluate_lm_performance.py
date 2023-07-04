import gc
import numpy as np
from evaluation.helper_functions import define_network, dropout_control, \
    rand_direction_control, dkl_gpu
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(10)


def data_projection(x, projection_matrix):
    return x.dot(projection_matrix)


def get_lm_predictions_gpu(w, b, x, y, projection: np.ndarray = None, device: str = 'cpu'):
    network = define_network(w, b, projection_mat=projection, device=device)
    accuracy = network.eval(x, y)
    del network
    gc.collect()
    return accuracy


def cosine_similarity_measure(original_data, modified_data):
    # Calculate cosine similarity row by row
    similarity_scores = []
    for row1, row2 in zip(original_data, modified_data):
        similarity = cosine_similarity([row1], [row2])[0][0]
        similarity_scores.append(similarity)

    # Calculate the average cosine similarity
    average_similarity = np.mean(similarity_scores)

    return average_similarity


def eval_lm_performance(tokenizer, out_embed, bias, x, words, projection, n_coords, device='cpu'):
    # n_coords -> removed directions

    y_ids = tokenizer.convert_tokens_to_ids(words)

    lm_results = {}
    # Accuracy without the removal of the information
    print("Accuracy without the removal of the information")
    base_acc = get_lm_predictions_gpu(out_embed, bias, x, y_ids, device=device)

    # Accuracy after the removal of the information
    print("Accuracy after the removal of the information")
    x_p = data_projection(x, projection)
    p_acc = get_lm_predictions_gpu(out_embed, bias, x_p, y_ids, device=device)

    # Dropout control
    print("Dropout control")
    x_dropout = dropout_control(x, n_coords)
    dropout_acc = get_lm_predictions_gpu(out_embed, bias, x_dropout, y_ids, device=device)

    # Random directions control
    print("Random directions control")
    x_rand_dir = rand_direction_control(x, n_coords)
    rand_dir_acc = get_lm_predictions_gpu(out_embed, bias, x_rand_dir, y_ids, device=device)

    cosine_similarity_projected = cosine_similarity_measure(x, x_p)
    cosine_similarity_random = cosine_similarity_measure(x, x_rand_dir)

    print("Projected cosine", cosine_similarity_projected)
    print("Random cosine", cosine_similarity_random)

    lm_results['cosine_sim_projected'] = cosine_similarity_projected
    lm_results['cosine_sim_random'] = cosine_similarity_random

    dkl_p, x_probs = dkl_gpu(out_embed, bias, x, x_p, y_ids, device=device)
    dkl_drop, _ = dkl_gpu(out_embed, bias, x, x_dropout, y_ids, x_probs, device=device)
    dkl_rand, _ = dkl_gpu(out_embed, bias, x, x_rand_dir, y_ids, x_probs, device=device)

    lm_results['dkl_p'] = dkl_p
    lm_results['dkl_dropout'] = dkl_drop
    lm_results['dkl_rand_dir'] = dkl_rand
    lm_results['lm_acc_vanilla'] = base_acc
    lm_results['lm_acc_p'] = p_acc
    lm_results['lm_acc_dropout'] = dropout_acc
    lm_results['lm_acc_rand_dir'] = rand_dir_acc

    del tokenizer
    del out_embed
    del bias
    del x
    del words
    del projection
    del base_acc
    del p_acc
    del dropout_acc
    del rand_dir_acc
    gc.collect()

    return lm_results
