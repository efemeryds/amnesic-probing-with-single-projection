# Code from https://github.com/yanaiela/amnesic_probing,
# https://github.com/shauli-ravfogel/nullspace_projection  with some modifications

import pickle
import numpy as np


class UDDataProcessing:

    def __init__(self, train_path, dev_path, task):
        self.train_path = train_path
        self.dev_path = dev_path
        self.task = task

    def read_files(self, vec_f, label_f, text_f=None, ignore_special_tokens=False):
        vecs = np.load(vec_f, allow_pickle=True)
        if ignore_special_tokens:
            vecs = np.array([x[1:-1] for x in vecs], dtype=object)
        with open(label_f, 'rb') as f:
            labels = pickle.load(f)
        if text_f:
            with open(text_f, 'rb') as f:
                sentences = pickle.load(f)
        else:
            sentences = None
        return vecs, labels, sentences

    def create_labeled_data(self, vecs, labels_seq, pos2i=None):
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

    def get_appropriate_data(self, vecs_train, labels_train, sentences_train, vecs_dev, labels_dev, sentences_dev):
        x_train, y_train, label2ind = self.create_labeled_data(vecs_train, labels_train)
        x_dev, y_dev, _ = self.create_labeled_data(vecs_dev, labels_dev, label2ind)

        words_train = [w for sen in sentences_train for w in sen]
        words_dev = [w for sen in sentences_dev for w in sen]
        return (x_train, y_train, words_train), (x_dev, y_dev, words_dev)

    def return_data_files(self):

        vecs_train, labels_train, sentences_train = self.read_files(self.train_path + "last_vec.npy",
                                                                    self.train_path + f"{self.task}.pickle",
                                                                    self.train_path + "tokens.pickle",
                                                                    ignore_special_tokens=True)

        vecs_dev, labels_dev, sentences_dev = self.read_files(self.dev_path + "last_vec.npy",
                                                              self.dev_path + f"{self.task}.pickle",
                                                              self.dev_path + "tokens.pickle",
                                                              ignore_special_tokens=True)

        (x_train, y_train, words_train), (x_dev, y_dev, words_dev) = self.get_appropriate_data(vecs_train, labels_train,
                                                                                               sentences_train,
                                                                                               vecs_dev, labels_dev,
                                                                                               sentences_dev)

        return x_train, y_train, words_train, x_dev, y_dev, words_dev
