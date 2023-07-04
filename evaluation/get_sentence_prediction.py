import numpy as np
from evaluation.helper_functions import data_projection
from evaluation.helper_functions import define_network, dropout_control, \
    rand_direction_control, dkl_gpu

np.random.seed(10)
import gc


def get_lm_logits(x, w, b):
    logits = np.dot(w, x.T) + np.array([b]).repeat(x.shape[0], axis=0).T
    return logits


def get_lm_predictions(tokenizer, out_embed, bias, x):
    logits = get_lm_logits(x, out_embed, bias)
    y = logits.argmax(axis=0)
    return tokenizer.convert_ids_to_tokens(y)


def get_srl_predictions_gpu(tokenizer, w, b, x, y, projection: np.ndarray = None, device: str = 'cpu'):
    network = define_network(w, b, projection_mat=projection, device=device)
    accuracy = network.eval(x, y)

    probs = network.get_probs(x, y)
    labels = probs.argmax(axis=0)
    # final_labels = tokenizer.convert_ids_to_tokens(y)
    del network
    gc.collect()
    return accuracy, labels


def sentence_prediction_example(tokenizer, out_embed, bias, text_tokens, text_vecs, task_labels,
                                projection_matrix):
    # print('sentence: ', ' '.join(text_tokens))
    # print('token', 'lm predicted token', 'lm task-less predicted token', 'task label')
    outputs = []
    _, predicted_tokens = get_srl_predictions_gpu(tokenizer, out_embed, bias, text_vecs, task_labels)
    _, predicted_tokens_p = get_srl_predictions_gpu(tokenizer, out_embed, bias, text_vecs, task_labels,
                                                    projection_matrix)
    for true_word, y_hat, y_hat_p, y_task in zip(text_tokens, predicted_tokens, predicted_tokens_p, task_labels):
        # print(true_word, y_hat, y_hat_p, y_task)
        outputs.append([true_word, y_hat, y_hat_p, y_task])
    return outputs

# Old
# def sentence_prediction_example(tokenizer, out_embed, bias, text_tokens, text_vecs, task_labels,
#                                 projection_matrix):
#     # print('sentence: ', ' '.join(text_tokens))
#     # print('token', 'lm predicted token', 'lm task-less predicted token', 'task label')
#     outputs = []
#     predicted_tokens = get_lm_predictions(tokenizer, out_embed, bias, text_vecs)
#     predicted_tokens_p = get_lm_predictions(tokenizer, out_embed, bias,
#                                             data_projection(text_vecs, projection_matrix))
#     for true_word, y_hat, y_hat_p, y_task in zip(text_tokens, predicted_tokens, predicted_tokens_p, task_labels):
#         # print(true_word, y_hat, y_hat_p, y_task)
#         outputs.append([true_word, y_hat, y_hat_p, y_task])
#     return outputs
