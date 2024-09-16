# Code with some modifications from https://github.com/yanaiela/amnesic_probing/tree/a96e3067a9f9918099015a7173c94830584bbe61

import time
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import torch
import pickle
import gc
from evaluation.PytorchClassifier import PytorchClassifier
from evaluation.helper_functions import get_lm_vals, load_data, load_labels, flatten_tokens, \
    flatten_label_list

np.random.seed(10)


class RebiasClassifier(torch.nn.Module):
    def __init__(self, debias, num_bias_labels, classifier, bias):
        super(RebiasClassifier, self).__init__()
        net = torch.nn.Linear(in_features=classifier.shape[1], out_features=classifier.shape[0])
        net.weight.data = torch.tensor(classifier)
        net.bias.data = torch.tensor(bias)

        rebias_embedding_size = 32
        encode_rebias = torch.nn.Embedding(num_bias_labels, rebias_embedding_size)
        rebias_net = torch.nn.Linear(in_features=classifier.shape[1] + rebias_embedding_size,
                                     out_features=classifier.shape[1])

        # self.debias_net = debias_net
        self.encode_rebias = encode_rebias
        self.rebias_net = rebias_net
        self.classifier_net = net

    def forward(self, input: torch.Tensor):
        rebias_labels = input[:, -1].long()
        input = input[:, :-1].float()

        debiased_input = self.debias_net(input)
        rebias = self.encode_rebias(rebias_labels)
        rebiased_input = torch.cat([debiased_input, rebias], dim=1)
        rebiased_input = self.rebias_net(rebiased_input)

        return self.classifier_net(rebiased_input)


class SelectivityControl:
    def __init__(self, tokenizer, word_embeddings, bias):
        self.tokenizer = tokenizer
        self.word_embeddings = word_embeddings
        self.bias = bias
        self.net = torch.nn.Linear(in_features=self.word_embeddings.shape[1],
                                   out_features=self.word_embeddings.shape[0])
        self.net.weight.data = torch.tensor(self.word_embeddings)
        self.net.bias.data = torch.tensor(bias)

    def run_selectivity_control(self, input_dir, projection, rebias_labels_name, out_dir):
        train_dir = input_dir + "/train"
        dev_dir = train_dir.replace('train', 'dev')
        debias = np.load(projection)

        train_vecs, train_words = load_data(train_dir)
        train_labels = load_labels(f'{train_dir}/{rebias_labels_name}.pickle')

        dev_vecs, dev_words = load_data(train_dir.replace('train', 'dev'))
        dev_labels = load_labels(f'{dev_dir}/{rebias_labels_name}.pickle')

        all_labels = list(set([y for sen_y in train_labels for y in sen_y]))

        x_train, y_words_train = flatten_tokens(train_vecs, train_words, self.tokenizer)
        y_labels_train = flatten_label_list(train_labels, all_labels)

        x_dev, y_words_dev = flatten_tokens(dev_vecs, dev_words, self.tokenizer)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        net_pytorch = PytorchClassifier(self.net, device=device)
        selectivity_results = net_pytorch.train(x_train, y_words_train, x_dev, y_words_dev,
                                                epochs=1,
                                                save_path=f"{out_dir}/finetuned_with_rebias.pt")

        with open(f'{out_dir}/selectivity_results.pkl', 'wb') as file:
            pickle.dump(selectivity_results, file)

        del train_vecs
        del train_words
        del train_labels
        del dev_vecs
        del dev_words
        del dev_labels
        del x_train
        del y_words_train
        del y_labels_train
        del x_dev
        del y_words_dev
        del selectivity_results
        gc.collect()
        return


def get_lm_vals(model_name):
    lm_model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    out_embed = lm_model.cls.predictions.decoder.weight.detach().cpu().numpy()
    bias = lm_model.cls.predictions.decoder.bias.detach().cpu().numpy()
    return lm_model, tokenizer, out_embed, bias


def run_selectivity_control(task: str, method_type: str, method_folder_path: str, original_data_path: str,
                            bert_tokenizer: str):
    lm_model, tokenizer, out_embed, bias = get_lm_vals(bert_tokenizer)

    start = time.time()
    # Run selectivity control
    selectivity_obj = SelectivityControl(tokenizer, out_embed, bias)
    selectivity_obj.run_selectivity_control(input_dir=original_data_path, projection=method_folder_path + '/P.npy',
                                            rebias_labels_name=task,
                                            out_dir=method_folder_path)
    end = time.time()
    print("The time it takes to run selectivity control equal: ",
          (end - start) / 60, " minutes")

    del selectivity_obj
    del tokenizer
    del out_embed
    del bias
    gc.collect()
    return


if __name__ == "__main__":
    ########## Universal Dependency dataset ##############
    ########### MASKED ###################

    run_selectivity_control("dep", "inlp",
                            "results/100k_batches_SGD_stable/masked/dep/removed_inlp", "datasets/ud_data_masked",
                            'bert-base-uncased')
    run_selectivity_control("dep", "mp", "results/100k_batches_SGD_stable/masked/dep/removed_mp",
                            "datasets/ud_data_masked",
                            'bert-base-uncased')

    # pos.pickle -> f-pos
    run_selectivity_control("pos", "inlp", "results/100k_batches_SGD_stable/masked/fpos/removed_inlp",
                            "datasets/ud_data_masked", 'bert-base-uncased')
    run_selectivity_control("pos", "mp", "results/100k_batches_SGD_stable/masked/fpos/removed_mp",
                            "datasets/ud_data_masked",
                            'bert-base-uncased')

    # tag.pickle -> c-pos
    run_selectivity_control("tag", "inlp",
                            "results/100k_batches_SGD_stable/masked/cpos/removed_inlp", "datasets/ud_data_masked",
                            'bert-base-uncased')
    run_selectivity_control("tag", "mp", "results/100k_batches_SGD_stable/masked/cpos/removed_mp",
                            "datasets/ud_data_masked",
                            'bert-base-uncased')

    ######### NORMAL NON-MASKED ###############
    run_selectivity_control("dep", "inlp",
                            "results/100k_batches_SGD_stable/normal/dep/removed_inlp", "datasets/ud_data_normal",
                            'bert-base-uncased')
    run_selectivity_control("dep", "mp", "results/100k_batches_SGD_stable/normal/dep/removed_mp",
                            "datasets/ud_data_normal",
                            'bert-base-uncased')

    # pos.pickle -> f-pos
    run_selectivity_control("pos", "inlp", "results/100k_batches_SGD_stable/normal/fpos/removed_inlp",
                            "datasets/ud_data_normal", 'bert-base-uncased')
    run_selectivity_control("pos", "mp", "results/100k_batches_SGD_stable/normal/fpos/removed_mp",
                            "datasets/ud_data_normal",
                            'bert-base-uncased')

    # tag.pickle -> c-pos
    run_selectivity_control("tag", "inlp",
                            "results/100k_batches_SGD_stable/normal/cpos/removed_inlp", "datasets/ud_data_normal",
                            'bert-base-uncased')
    run_selectivity_control("tag", "mp", "results/100k_batches_SGD_stable/normal/cpos/removed_mp",
                            "datasets/ud_data_normal",
                            'bert-base-uncased')
