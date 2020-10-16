import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class POSTDataset(Dataset):

    def __init__(self, data_root, indexes, max_len):
        super(POSTDataset, self).__init__()
        self.data_root = data_root
        self.indexes = indexes
        self.max_len = max_len
        self.X, self.Y, self.label_names = self._load_data()
        self.label_names = sorted(self.label_names)
        print('labels: ', self.label_names)
        # self.label_names.insert(0, 'BLANK')
        self.missing_embedding_idx = 1

    def _load_data(self):

        MAX_LEN = 0
        data = []
        gt = []
        tag_names = []
        for fn in os.listdir(self.data_root):
            path = os.path.join(self.data_root, fn)
            with open(path, 'r') as f:
                lines = [line.strip().split(' ') for line in f.readlines()]
                for line in lines:
                    line = [t for i, t in enumerate(line) if len(t) > 0]
                    if len(line) > 0:
                        words = [word_tag.split('/')[0] for word_tag in line]
                        MAX_LEN = max(MAX_LEN, len(words))
                        tags = [word_tag.split('/')[1] for word_tag in line]
                        tag_names.extend(tags)
                        data.append(words)
                        gt.append(tags)

        return data, gt, list(set(tag_names))

    @property
    def n_class(self):
        return len(self.label_names)

    def preprocess(self, input, label=None):

        def pad(input, output=None):
            pad_len = self.max_len - len(input)
            input += [0 for i in range(pad_len)]
            if output:
                output += [-1 for i in range(pad_len)]

            return input, output

        input = list(map(lambda x: self.indexes.get(x, self.missing_embedding_idx), input))
        if label:
            label = [self.label_names.index(tag) for i, tag in enumerate(label)]
        padded_input, padded_label = pad(input, label)
        padded_input = torch.tensor(padded_input)
        if padded_label:
            padded_label = torch.tensor(padded_label)

        return padded_input, padded_label

    def postprocess(self, pred, n_words):

        pred = np.argmax(pred, axis=1)
        tags = [self.label_names[pred[i]] for i in range(n_words)]

        return tags

    def sample(self, sample_size):

        sample_inds = np.random.choice(list(range(len(self.X))), sample_size, replace=False)
        sample_sentences = []
        sample_input = torch.zeros(sample_size, self.max_len)
        sample_label = torch.zeros(sample_size, self.max_len)
        for i, ind in enumerate(sample_inds):
            sample_sentences.append(self.X[ind])
            tmp_sentence, tmp_tag = self.preprocess(self.X[ind], self.Y[ind])
            sample_input[i] = tmp_sentence
            sample_label[i] = tmp_tag

        #sample_sentences.append(self.X[ind])
        #tmp_sentence, tmp_tag = self.preprocess(self.X[ind], self.Y[ind])
        #sample_input[i] = tmp_sentence
        #sample_label[i] = tmp_tag

        sample_input = sample_input.long()
        sample_label = sample_label.long()

        return sample_sentences, sample_input, sample_label

    def __getitem__(self, item):

        sentence = self.X[item]
        tags = self.Y[item]
        sentence, tags = self.preprocess(sentence, tags)

        return sentence, tags

    def __len__(self):
        return len(self.X)
