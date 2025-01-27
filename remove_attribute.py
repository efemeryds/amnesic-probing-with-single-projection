# Code from https://github.com/yanaiela/amnesic_probing, https://github.com/shauli-ravfogel/nullspace_projection
# and https://github.com/tue-alga/debias-mean-projection with some modifications

import numpy as np
import os
from sklearn.linear_model import SGDClassifier
import json
import gc
from sklearn.utils import shuffle
from collections import Counter
from projection_methods.INLPMethod import INLPMethod
from projection_methods.MPMethod import MPMethod, get_directions
from projection_methods.LEACE import LeaceEraser, LeaceFitter
import torch
import time
from data_processing.UDDataProcessing import UDDataProcessing

np.random.seed(10)


def save_leace_projection(eraser, file_path):
    """
    Save LEACE projection matrices and bias to a file.
    """
    data = {
        'proj_left': eraser.proj_left.cpu().numpy(),
        'proj_right': eraser.proj_right.cpu().numpy(),
        'bias': eraser.bias.cpu().numpy() if eraser.bias is not None else None
    }

    if eraser.bias is None:
        print("Bias is None, ensure this is intentional.")

    torch.save(data, file_path)  # Use PyTorch serialization for consistency
    print(f"LEACE projection saved to {file_path}")


def load_leace_projection(file_path, device="cpu"):
    """
    Load LEACE projection matrices and bias from a file and reconstruct the eraser.
    """
    data = torch.load(file_path)
    proj_left = torch.tensor(data['proj_left'], device=device)
    proj_right = torch.tensor(data['proj_right'], device=device)
    bias = torch.tensor(data['bias'], device=device) if data['bias'] is not None else None
    if bias is None:
        print("Bias is None, ensure this is intentional.")

    print(f"LEACE projection loaded from {file_path}")
    return LeaceEraser(proj_left=proj_left, proj_right=proj_right, bias=bias)


def remove_attribute_leace(task, x_train, y_train, x_dev, y_dev, out_dir_leace):
    folder_exists = os.path.exists(out_dir_leace)
    if not folder_exists:
        os.makedirs(out_dir_leace)

    print(f"Running LEACE method for task: {task}..")
    n_classes = len(set(y_train))
    majority = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))

    # Convert data to torch tensors for LEACE
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

    n_classes = len(set(y_train))  # Total number of unique labels in y_train
    # special way for leace
    y_train_tensor = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=n_classes).float()

    x_dev_tensor = torch.tensor(x_dev, dtype=torch.float32)

    # Fit the LEACE model

    eraser = LeaceEraser.fit(x_train_tensor, y_train_tensor)

    final_path = out_dir_leace + "/trained_parameters.npz"
    # Save the projection
    save_leace_projection(eraser, final_path)

    # Load the projection
    loaded_eraser = load_leace_projection(final_path)

    x_dev_cleaned_leace = loaded_eraser(x_dev_tensor).numpy()
    x_train_cleaned_leace = loaded_eraser(x_train_tensor).numpy()

    # Test accuracy before and after removal
    model = SGDClassifier
    loss = 'log'
    warm_start = True
    early_stopping = False
    max_iter = 10000

    params = {'warm_start': warm_start, 'loss': loss, 'random_state': 0, 'early_stopping': early_stopping,
              'max_iter': max_iter}

    test_network_before = INLPMethod(model, params)
    score_before, weights = test_network_before.train_network(x_train, y_train, x_dev, y_dev)
    print("Accuracy before removal: ", score_before)


    del x_train
    del test_network_before
    gc.collect()

    p_rank = np.linalg.matrix_rank(x_dev_cleaned_leace).tolist()
    rank_before = np.linalg.matrix_rank(x_dev).tolist()

    del x_dev
    gc.collect()

    test_network_after = INLPMethod(model, params)
    score_after, weights = test_network_after.train_network(x_train_cleaned_leace, y_train, x_dev_cleaned_leace,
                                                            y_dev)
    print("Accuracy after removal: ", score_after)

    # Save metadata
    meta_dic = {'best_i': 0,
                'n_classes': n_classes,
                'majority': majority,
                'removed_directions': n_classes,
                'probing_accuracy': score_before,
                'final_accuracy': score_after,
                'p_rank': p_rank,
                'p_train_rank': rank_before,
                'accuracy_per_iteration': score_after,
                'loss': loss,
                'warm_start': warm_start,
                'early_stopping': early_stopping,
                'max_iter': max_iter}

    json.dump(meta_dic, open(out_dir_leace + '/meta.json', 'w'))

    del model
    del x_dev_cleaned_leace
    del x_train_cleaned_leace
    del test_network_after
    del meta_dic
    gc.collect()
    print('done iterations. exiting................')
    return


def remove_attribute_inlp(x_train, y_train, x_dev, y_dev, out_dir_inlp, num_clfs=20):
    start = time.time()
    model = SGDClassifier

    loss = 'log'
    warm_start = True
    early_stopping = False
    max_iter = 10000

    params = {'warm_start': warm_start, 'loss': loss, 'random_state': 0, 'early_stopping': early_stopping,
              'max_iter': max_iter}

    folder_exists = os.path.exists(out_dir_inlp)
    if not folder_exists:
        os.makedirs(out_dir_inlp)

    n_classes = len(set(y_train))
    majority = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))

    print('The number of classes:', n_classes)
    print('The most common class (dev):', majority)

    x_train, y_train = shuffle(x_train, y_train, random_state=0, n_samples=len(y_train))
    # Run INLP method
    inlp_object = INLPMethod(model, params)

    # Main function for INLP
    output_p, all_projections, best_projection, accuracy_per_iteration = inlp_object.get_projection_matrix(num_clfs,
                                                                                                           x_train,
                                                                                                           y_train,
                                                                                                           x_dev, y_dev,
                                                                                                           majority_acc=majority)
    for i, projection in enumerate(all_projections):
        np.save(out_dir_inlp + '/P_{}.npy'.format(i), projection)

    # Save final projection P
    np.save(out_dir_inlp + '/P.npy', best_projection[0])

    removed_directions = int((best_projection[1]) * n_classes)
    print("The number of removed_directions: ", removed_directions)

    # In case of 2 classes, each inlp iteration we remove a single direction
    if out_dir_inlp == 2:
        removed_directions /= 2

    x_dev_projected = np.dot(x_dev, output_p)  # (projection_mp.dot(x_dev.T)).T
    x_train_projected = np.dot(x_train, output_p)

    print("The accuracy before removal: ", accuracy_per_iteration[0])

    del all_projections
    del inlp_object
    del x_train

    gc.collect()

    rank_before = np.linalg.matrix_rank(x_dev).tolist()

    test_network_after = INLPMethod(model, params)
    score_after, weights = test_network_after.train_network(x_train_projected, y_train, x_dev_projected, y_dev)
    print("The accuracy after removal: ", score_after)

    del output_p
    del model
    del test_network_after
    del x_train_projected
    del y_train
    del y_dev

    gc.collect()

    # Save metadata
    meta_dic = {'best_i': best_projection[1],
                'n_classes': n_classes,
                'majority': majority,
                'removed_directions': removed_directions,
                'probing_accuracy': accuracy_per_iteration[0],
                'final_accuracy': score_after,
                'p_rank': x_dev_projected,
                'rank_before': rank_before,
                'accuracy_per_iteration': accuracy_per_iteration,
                'loss': loss,
                'warm_start': warm_start,
                'early_stopping': early_stopping,
                'max_iter': max_iter
                }

    json.dump(meta_dic, open(out_dir_inlp + '/meta.json', 'w'))

    end = time.time()
    print(f"The time it takes to remove attribute using INLP: ",
          (end - start) / 60, " minutes")

    print('The processing is finished. Exiting......')
    del meta_dic
    del x_dev
    del accuracy_per_iteration
    gc.collect()
    return


def remove_attribute_mp(task, x_train, y_train, x_dev, y_dev, out_dir_mp, directions):
    folder_exists = os.path.exists(out_dir_mp)
    if not folder_exists:
        os.makedirs(out_dir_mp)

    print(f"Running MP method for task: {task}..")
    n_classes = len(set(y_train))
    majority = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))

    # Run Single Mean Projection method
    mp_object = MPMethod(task, directions)
    projection_mp = mp_object.mean_projection_method()

    # Save final projection P
    np.save(out_dir_mp + '/P.npy', projection_mp)

    x_dev_cleaned_mp = np.dot(x_dev, projection_mp)  # (projection_mp.dot(x_dev.T)).T
    x_train_cleaned_mean_mp = np.dot(x_train, projection_mp)  # projection_mp.dot(x_train.T).T

    model = SGDClassifier
    loss = 'log'
    warm_start = True
    early_stopping = False
    max_iter = 10000

    params = {'warm_start': warm_start, 'loss': loss, 'random_state': 0, 'early_stopping': early_stopping,
              'max_iter': max_iter}

    test_network_before = INLPMethod(model, params)
    score_before, weights = test_network_before.train_network(x_train, y_train, x_dev, y_dev)
    print("Accuracy before removal: ", score_before)

    del x_train
    del test_network_before
    gc.collect()

    p_rank = np.linalg.matrix_rank(x_dev_cleaned_mp).tolist()
    rank_before = np.linalg.matrix_rank(x_dev).tolist()

    del x_dev
    gc.collect()

    test_network_after = INLPMethod(model, params)
    score_after, weights = test_network_after.train_network(x_train_cleaned_mean_mp, y_train, x_dev_cleaned_mp, y_dev)
    print("Accuracy after removal: ", score_after)

    # Save metadata
    meta_dic = {'best_i': 0,
                'n_classes': n_classes,
                'majority': majority,
                'removed_directions': n_classes,
                'probing_accuracy': score_before,
                'final_accuracy': score_after,
                'p_rank': p_rank,
                'p_train_rank': rank_before,
                'accuracy_per_iteration': score_after,
                'loss': loss,
                'warm_start': warm_start,
                'early_stopping': early_stopping,
                'max_iter': max_iter}

    json.dump(meta_dic, open(out_dir_mp + '/meta.json', 'w'))

    del model
    del directions
    del projection_mp
    del mp_object
    del x_dev_cleaned_mp
    del x_train_cleaned_mean_mp
    del test_network_after
    del meta_dic
    gc.collect()
    print('done iterations. exiting................')
    return


def run_pipeline(task, dir_path, out_dir_inlp_dep, out_dir_mp_dep, out_dir_leace_dep):
    print(f"Removing {task} attribute.............")

    process_data = UDDataProcessing(dir_path + "/train/", dir_path + "/dev/", task)
    x_train, y_train, _, x_dev, y_dev, _ = process_data.return_data_files()

    print("Rank train", np.linalg.matrix_rank(x_train))
    print("Rank dev", np.linalg.matrix_rank(x_dev))

    print("Applying INLP..............")
    # remove_attribute_inlp(x_train, y_train, x_dev, y_dev, out_dir_inlp_dep)
    directions_mp = get_directions(x_train, y_train)

    print("Applying Mean Projection..............")
    # remove_attribute_mp(task, x_train, y_train, x_dev, y_dev, out_dir_mp_dep, directions_mp)

    print("Applying LEACE..............")
    remove_attribute_leace(task, x_train, y_train, x_dev, y_dev, out_dir_leace_dep)

    del x_train
    del y_train
    del x_dev
    del y_dev
    gc.collect()
    return


if __name__ == "__main__":
    # Universal Dependency dataset
    # MASKED
    run_pipeline("dep", "datasets/ud_data_masked", "results/100k_batches_SGD_stable/masked/dep/removed_inlp",
                 "results/100k_batches_SGD_stable/masked/dep/removed_mp",
                 "results/100k_batches_SGD_stable/masked/dep/removed_leace")

    run_pipeline("pos", "datasets/ud_data_masked", "results/100k_batches_SGD_stable/masked/fpos/removed_inlp",
                 "results/100k_batches_SGD_stable/masked/fpos/removed_mp",
                 "results/100k_batches_SGD_stable/masked/fpos/removed_leace")

    run_pipeline("tag", "datasets/ud_data_masked", "results/100k_batches_SGD_stable/masked/cpos/removed_inlp",
                 "results/100k_batches_SGD_stable/masked/cpos/removed_mp",
                 "results/100k_batches_SGD_stable/masked/cpos/removed_leace")

    # NON-MASKED -> NORMAL
    run_pipeline("dep", "datasets/ud_data_normal", "results/100k_batches_SGD_stable/normal/dep/removed_inlp",
                 "results/100k_batches_SGD_stable/normal/dep/removed_mp",
                 "results/100k_batches_SGD_stable/normal/dep/removed_leace")

    run_pipeline("pos", "datasets/ud_data_normal", "results/100k_batches_SGD_stable/normal/fpos/removed_inlp",
                 "results/100k_batches_SGD_stable/normal/fpos/removed_mp",
                 "results/100k_batches_SGD_stable/normal/fpos/removed_leace")

    run_pipeline("tag", "datasets/ud_data_normal", "results/100k_batches_SGD_stable/normal/cpos/removed_inlp",
                 "results/100k_batches_SGD_stable/normal/cpos/removed_mp",
                 "results/100k_batches_SGD_stable/normal/cpos/removed_leace")
