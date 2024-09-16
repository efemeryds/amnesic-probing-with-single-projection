# Code with some modifications from https://github.com/yanaiela/amnesic_probing/tree/a96e3067a9f9918099015a7173c94830584bbe61

import gc
from sklearn.linear_model import SGDClassifier
import json
import os
from collections import Counter
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.utils import shuffle
from evaluation.helper_functions import data_projection
from transformers import BertTokenizer, BertForMaskedLM
from evaluation.EvalTaskPerformance import EvalTaskPerformance
from evaluation.evaluate_lm_performance import eval_lm_performance
from evaluation.get_sentence_prediction import sentence_prediction_example

np.random.seed(10)


def read_files(vec_f, label_f, text_f=None, ignore_special_tokens=False):
    vecs = np.load(vec_f, allow_pickle=True)

    if ignore_special_tokens:
        vecs = np.array([x[1:-1] for x in vecs])

    with open(label_f, 'rb') as f:
        labels = pickle.load(f)

    if text_f:
        with open(text_f, 'rb') as f:
            sentences = pickle.load(f)
    else:
        sentences = None

    return vecs, labels, sentences


def create_labeled_data(vecs, labels_seq, pos2i=None):
    x = []
    y = []

    if not pos2i:
        # using `sorted` function to make this process deterministic
        pos2i = {p: i for i, p in enumerate(sorted(set([item for sublist in labels_seq for item in sublist])))}

    for label, vec in zip(labels_seq, vecs):
        for l, v in zip(label, vec):
            x.append(v)
            y.append(pos2i[l])

    return np.array(x), np.array(y), pos2i


def get_appropriate_data(vecs_train, labels_train, sentences_train, vecs_dev, labels_dev, sentences_dev):
    x_train, y_train, label2ind = create_labeled_data(vecs_train, labels_train)
    x_dev, y_dev, _ = create_labeled_data(vecs_dev, labels_dev, label2ind)

    words_train = [w for sen in sentences_train for w in sen]
    words_dev = [w for sen in sentences_dev for w in sen]
    return x_train, y_train, words_train, x_dev, y_dev, words_dev


def load_deprobing_params(in_file):
    with open(in_file, 'r') as f:
        meta = json.load(f)
    return meta


def get_lm_vals(model_name):
    lm_model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    out_embed = lm_model.cls.predictions.decoder.weight.detach().cpu().numpy()
    bias = lm_model.cls.predictions.decoder.bias.detach().cpu().numpy()
    return lm_model, tokenizer, out_embed, bias


def prepare_data_per_task(task, original_data_path, method_folder_path):
    dir_path_train = original_data_path + "/train/"
    dir_path_dev = original_data_path + "/dev/"

    vecs_train, labels_train, sentences_train = read_files(dir_path_train + "last_vec.npy",
                                                           dir_path_train + f"{task}.pickle",
                                                           dir_path_train + "tokens.pickle",
                                                           ignore_special_tokens=True)

    vecs_dev, labels_dev, sentences_dev = read_files(dir_path_dev + "last_vec.npy",
                                                     dir_path_dev + f"{task}.pickle",
                                                     dir_path_dev + "tokens.pickle",
                                                     ignore_special_tokens=True)

    x_train, y_train, words_train, x_dev, y_dev, words_dev = get_appropriate_data(vecs_train, labels_train,
                                                                                  sentences_train,
                                                                                  vecs_dev, labels_dev,
                                                                                  sentences_dev)

    pos2ind = {p: i for i, p in enumerate(sorted(set([item for sublist in labels_train for item in sublist])))}

    print('number of classes', len(pos2ind))
    print('most common class', Counter(y_dev).most_common(1)[0][1] / float(len(y_dev)))

    meta = load_deprobing_params(method_folder_path + '/meta.json')
    n_coordinates = int(meta['removed_directions'])

    print("Number of coordinates: ", n_coordinates)
    # Load learnt projection
    proj_file = method_folder_path + '/P.npy'

    if os.path.isfile(proj_file):
        P = np.load(proj_file)
    else:
        raise FileNotFoundError('projection file does not exists...')

    del vecs_train

    gc.collect()

    return x_train, y_train, words_train, x_dev, y_dev, words_dev, sentences_dev, vecs_dev, labels_dev, \
        n_coordinates, P


def evaluation_and_control_per_task(task: str, method_type: str, method_folder_path: str, original_data_path: str,
                                    bert_tokenizer: str):
    print(f"Task: {task}, method: {method_type}, method folder: {method_folder_path}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)

    _, tokenizer, out_embed, bias = get_lm_vals(bert_tokenizer)

    model = SGDClassifier
    loss = 'log'
    warm_start = True
    early_stopping = False
    max_iter = 10000

    parameters = {'warm_start': warm_start, 'loss': loss, 'random_state': 0, 'early_stopping': early_stopping,
                  'max_iter': max_iter}

    # Prepare data
    x_train, y_train, words_train, x_dev, y_dev, words_dev, sentences_dev, vecs_dev, labels_dev, n_coordinates, P = \
        prepare_data_per_task(
            task, original_data_path, method_folder_path)

    print("Evaluating the results - basic, dropout, random directions")
    lm_results = eval_lm_performance(tokenizer, out_embed, bias, x_dev, words_dev, P,
                                     n_coords=n_coordinates, device=device)


    print('Task Control')
    x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train, random_state=0, n_samples=min(len(y_train), 100000))
    x_train_no_label = data_projection(x_train_shuffled, P)
    x_dev_no_label = data_projection(x_dev, P)

    # Run batches 100k
    eval_task_perf = EvalTaskPerformance(model, parameters)
    task_results = eval_task_perf.eval_task_performance(x_train_shuffled, y_train_shuffled, x_dev, y_dev,
                                                        x_train_no_label,
                                                        x_dev_no_label)

    all_results = {**lm_results, **task_results}

    with open(f'{method_folder_path}/all_results.pkl', 'wb') as file:
        pickle.dump(all_results, file)

    print("Sentence Prediction results")
    # how the standard and after information removal model behaves with token prediction task
    table_data = []
    ind = 0
    for i in range(int(10)):
        for w, orig_y, P_y, y_label in sentence_prediction_example(tokenizer, out_embed, bias,
                                                                   sentences_dev[i],
                                                                   vecs_dev[i],
                                                                   labels_dev[i], P):
            table_data.append([w, orig_y, P_y, y_label, ind])
            ind += 1
        table_data.append(['-', '-', '-', '-', ind])
        ind += 1

    df = pd.DataFrame(table_data, columns=["word", "lm_word", "-p_word", "label", "index"])
    df.to_csv(method_folder_path + '/sentence_prediction_results.tsv', sep='\t', index=False)

    del x_train
    del y_train
    del x_dev
    del y_dev
    del tokenizer
    del bias
    del out_embed
    gc.collect()

    return


if __name__ == "__main__":
    ########## Universal Dependency dataset #############

    ######### MASKED ################
    evaluation_and_control_per_task("dep", "inlp",
                                    "results/100k_batches_SGD_stable/masked/dep/removed_inlp",
                                    "datasets/ud_data_masked",
                                    'bert-base-uncased')

    evaluation_and_control_per_task("dep", "mp", "results/100k_batches_SGD_stable/masked/dep/removed_mp",
                                    "datasets/ud_data_masked",
                                    'bert-base-uncased')

    # pos.pickle -> f-pos
    evaluation_and_control_per_task("pos", "inlp", "results/100k_batches_SGD_stable/masked/fpos/removed_inlp",
                                    "datasets/ud_data_masked", 'bert-base-uncased')
    evaluation_and_control_per_task("pos", "mp", "results/100k_batches_SGD_stable/masked/fpos/removed_mp",
                                    "datasets/ud_data_masked",
                                    'bert-base-uncased')

    # tag.pickle -> c-pos
    evaluation_and_control_per_task("tag", "inlp",
                                    "results/100k_batches_SGD_stable/masked/cpos/removed_inlp",
                                    "datasets/ud_data_masked",
                                    'bert-base-uncased')
    evaluation_and_control_per_task("tag", "mp", "results/100k_batches_SGD_stable/masked/cpos/removed_mp",
                                    "datasets/ud_data_masked",
                                    'bert-base-uncased')

    ############# NON-MASKED -> NORMAL #################

    evaluation_and_control_per_task("dep", "inlp",
                                    "results/100k_batches_SGD_stable/normal/dep/removed_inlp",
                                    "datasets/ud_data_normal",
                                    'bert-base-uncased')
    evaluation_and_control_per_task("dep", "mp", "results/100k_batches_SGD_stable/normal/dep/removed_mp",
                                    "datasets/ud_data_normal",
                                    'bert-base-uncased')

    # pos.pickle -> f-pos
    evaluation_and_control_per_task("pos", "inlp", "results/100k_batches_SGD_stable/normal/fpos/removed_inlp",
                                    "datasets/ud_data_normal", 'bert-base-uncased')
    evaluation_and_control_per_task("pos", "mp", "results/100k_batches_SGD_stable/normal/fpos/removed_mp",
                                    "datasets/ud_data_normal",
                                    'bert-base-uncased')

    # tag.pickle -> c-pos
    evaluation_and_control_per_task("tag", "inlp",
                                    "results/100k_batches_SGD_stable/normal/cpos/removed_inlp",
                                    "datasets/ud_data_normal",
                                    'bert-base-uncased')
    evaluation_and_control_per_task("tag", "mp", "results/100k_batches_SGD_stable/normal/cpos/removed_mp",
                                    "datasets/ud_data_normal",
                                    'bert-base-uncased')

