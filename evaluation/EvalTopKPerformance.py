from evaluation.helper_functions import dropout_control, rand_direction_control, \
    define_network, data_projection
import torch
import numpy as np
from collections import Counter

np.random.seed(10)


class EvalTopKPerformance:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def convert_words2labels(self, words, probable_labels, label2ind, most_common_label):
        labels_freqeuency = np.zeros(len(label2ind))
        for word in words:
            labels_freqeuency[label2ind[probable_labels.get(word, most_common_label)]] += 1
        return labels_freqeuency

    def get_top_k_lm_predictions_gpu(self, tokenizer, out_embed, bias, x, y, projection: np.ndarray = None, k=100,
                                     device: str = 'cpu'):
        network = define_network(out_embed, bias, projection_mat=projection, device=device)
        distribution = network.get_probs(x, y)[0]
        top_y = torch.tensor(distribution).to(device).topk(k=k, dim=1, largest=True, sorted=True).indices.cpu().numpy()
        top_words = []
        for top_k_per_word in top_y:
            top_k = tokenizer.convert_ids_to_tokens(top_k_per_word)
            top_words.append(top_k)
        return top_words

    def calc_entropy(self, x, y, y_labels, probable_labels, k, device):
        all_labels = list(set(y_labels))
        ind2label = dict(enumerate(all_labels))
        label2ind = {v: k for k, v in ind2label.items()}
        most_common_label = Counter(y_labels).most_common(1)[0][0]

        top_words = self.get_top_k_lm_predictions_gpu(x, y, None, k=k, device=device)
        all_dists = torch.tensor(
            [self.convert_words2labels(top_words[i], probable_labels, label2ind, most_common_label) for i in
             range(len(top_words))]).to(device)
        # this will be normalized to 2
        entropy_score = torch.distributions.Categorical(logits=all_dists).entropy().mean().cpu().numpy()
        return entropy_score

    def eval_topk_performance(self, tokenizer, out_embed, bias, x, words, projection, probable_labels, n_coords,
                              y_train_labels, k=100,
                              device='cpu'):
        y_ids = tokenizer.convert_tokens_to_ids(words)

        lm_labels_results = {}
        entropy_vanilla = self.calc_entropy(x, y_ids, y_train_labels, probable_labels, k, device)

        x_p = data_projection(x, projection)
        entropy_p = self.calc_entropy(x_p, y_ids, y_train_labels, probable_labels, k, device)

        x_dropout = dropout_control(x, n_coords)
        entropy_dropout = self.calc_entropy(x_dropout, y_ids, y_train_labels, probable_labels, k, device)

        x_rand_dir = rand_direction_control(x, n_coords)
        entropy_rand_dir = self.calc_entropy(x_rand_dir, y_ids, y_train_labels, probable_labels, k, device)

        lm_labels_results['top_k_entropy_vanilla'] = entropy_vanilla
        lm_labels_results['top_k_entropy_p'] = entropy_p
        lm_labels_results['top_k_entropy_dropout'] = entropy_dropout
        lm_labels_results['top_k_entropy_rand_dir'] = entropy_rand_dir
        return lm_labels_results
