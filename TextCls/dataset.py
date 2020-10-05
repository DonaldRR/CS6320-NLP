import os
from collections import defaultdict
import numpy as np

class TextDataset(object):

    def __init__(self, ds_dir, split='train', stopwords_path=None, method=None, n_features=None):

        self.ds_dir = ds_dir
        self.stopwords = self._load_stopwords(stopwords_path)
        self.method = method
        self.n_features = n_features
        self.data = self._load_ds()
        self.data_priors = self._compute_priors()
        self.data_table = None
        self.vocabulary = None
        self.N_words_doc = None

        if split.lower() == 'train':
            self.data_table, self.vocabulary, self.N_words_doc =  self._collect_words(self.data)
            self.N = len(self.vocabulary)
            self._feature_selection()

    def _compute_priors(self):

        priors = defaultdict(lambda : 0)
        N = 0
        for label, docs in self.data.items():
            N += len(docs)
        for label, docs in self.data.items():
            priors[label] = len(docs) + 1 / (N + len(self.data))

        return priors

    def _load_stopwords(self, path):
        if path:
            with open(path, 'r') as f:
                stopwords = f.readlines()
                stopwords = [line.strip() for line in stopwords]
                print('* Use %d stopwords' % len(stopwords))
                return stopwords
        return []

    def _feature_selection(self):

        """
        Choose a subset of words to be evaluated during inference

        :return:
        """

        idx = 0
        cls2idx = {}
        for text_label, text_content in self.data.items():
            cls2idx[text_label] = idx
            cls2idx['_' + text_label] = idx + 1
            idx += 2

        if self.method:
            if self.method.lower() == 'mi':

                # counting the number of documents each word occurring in each class
                N_mat = {word:[0 for i in range(len(cls2idx))] for word, _ in self.vocabulary.items()}
                count_c = {}
                for text_label, text_content in self.data.items():
                    for text in text_content:
                        text = set(text)
                        for word in text:
                            N_mat[word][cls2idx[text_label]] += 1
                    count_c[text_label] = len(text_content)
                    for word, count_dict in N_mat.items():
                        N_mat[word][cls2idx['_' + text_label]] = count_c[text_label] - N_mat[word][cls2idx[text_label]]

                word_list = [] # vocabulary list
                N_list = [] # [n11, n01, n10, n00]
                for word, counts in N_mat.items():
                    word_list.append(word)
                    N_list.append(counts)
                N_list = np.array(N_list) + 1

                # calculate mutual information
                n11, n01, n10, n00 = N_list[:, 0], N_list[:, 1], N_list[:, 2], N_list[:, 3]
                I = n11 * np.log2(n11 / (n11 + n01) / (n11 + n10)) + \
                    n01 * np.log2(n01 / (n00 + n01) / (n11 + n01)) + \
                    n10 * np.log2(n10 / (n11 + n10) / (n00 + n10)) + \
                    n00 * np.log2(n00 / (n00 + n01) / (n00 + n10))
                #word_mi = [[word_list[i], I[i]] for i in range(len(word_list))]
                #word_mi = sorted(word_mi, key=lambda x: x[1])

                # rank and select words of high mutual information
                word_list = sorted(word_list, key=lambda x: I[word_list.index(x)])
                selected_words = word_list[-self.n_features:]
                tmp_data_table = {k:defaultdict(lambda : 0) for k, _ in self.data_table.items()}
                for text_label, _ in self.data_table.items():
                    for word in selected_words:
                        tmp_data_table[text_label][word] = self.data_table[text_label][word]
                self.data_table = tmp_data_table
            else:
                raise NotImplementedError('* ERROR:%s feature method is not implemented' % self.method)

    def get_data(self):
        return self.data

    def get_data_table(self):
        return self.data_table

    def p_w_c(self, w, c):

        """
        Compute posterior probability P(w|c), where c is the class of the text and w the word

        :param w:
        :param c:
        :return:
        """

        if self.data_table:
            n_c = self.N_words_doc[c] + self.N
            n_w_c = self.data_table[c][w] + 1
            return n_w_c / n_c
        else:
            raise NotImplementedError('This is not training set')

    def _load_ds(self):

        """
        Load dataset from .txt files
        :return:
        """

        ham_path = os.path.join(self.ds_dir, 'ham')
        spam_path = os.path.join(self.ds_dir, 'spam')

        def _line_filter(x):
            x = x.strip().split(' ')
            x = [w for _, w in enumerate(x) if len(w) > 0 and w not in self.stopwords]
            return x

        def extract_text(text_path):

            with open(text_path, 'r', errors='ignore') as f:

                ret = []
                lines = f.readlines()
                lines = list(map(_line_filter, lines))
                for line in lines:
                    ret += line
                ret = ['<START>'] + ret + ['<END>']

                return ret

        ham_fns = os.listdir(ham_path)
        hams = []
        for fn in ham_fns:
            hams.append(extract_text(os.path.join(ham_path, fn)))
        print('* #ham=%d' % len(hams))

        spam_fns = os.listdir(spam_path)
        spams = []
        for fn in spam_fns:
            spams.append(extract_text(os.path.join(spam_path, fn)))
        print('* #spam=%d' % len(spams))

        return {
            'ham': hams,
            'spam': spams
        }

    def _collect_words(self, data_dict):

        """
        :param data_dict: dict, the keys represent classes and the corresponding values represent texts in such classes
        :return: (counting_dict, vocabulary), (dict, dict)
        """

        vocabulary = defaultdict(lambda : 0)
        ret = defaultdict(lambda: defaultdict(lambda: 0))
        n_words_doc = defaultdict(lambda : 0)
        for text_label, text_content in data_dict.items():
            for text in text_content:
                for word in text:
                    ret[text_label][word] += 1
                    vocabulary[word] += 1
            n_words_doc[text_label] = sum(list(ret[text_label].values()))

        return ret, vocabulary, n_words_doc
