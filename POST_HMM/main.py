import numpy as np
from collections import defaultdict
import os
import math
import argparse

class Dataset:

    def __init__(self, root_dir):

        self.dataset_root = root_dir
        self.corpus = self.load_corpus(self.dataset_root)
        self.tags = None
        self.voc = None
        self.PI, self.T, self.E = self.initialize_probabilities(self.corpus)

    def load_corpus(self, root_dir):

        data = []
        for fn in os.listdir(root_dir):
            text_path = os.path.join(root_dir, fn)
            with open(text_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                for line in lines:
                    if len(line) > 0:
                        data.append(list(map(lambda x: tuple(x.split('/')) , line.split(' '))) + [("<END>","END")])
        print('* Load %d sentences' % len(data))

        return data

    def initialize_probabilities(self, sentences):

        # initial prob.
        PI = defaultdict(lambda :0)
        # transition prob.
        T = defaultdict(lambda : defaultdict(lambda : 0))
        # emission prob.
        E = defaultdict(lambda : defaultdict(lambda : 0))
        # tag set
        TAGS = set()
        # vocabulary set
        VOC = set()

        # count tags/words
        for sentence in sentences:
            for i, word_tag in enumerate(sentence):
                word, tag = word_tag
                if i == 0:
                    PI[tag] += 1
                if i < len(sentence) - 1:
                    next_tag = sentence[i + 1][1]
                    T[tag][next_tag] += 1
                E[tag][word] += 1
                TAGS.add(tag)
                VOC.add(word)

        # calculate initial prob.
        N_tags = sum(list(PI.values()))
        for tag in TAGS:
            PI[tag] = (PI[tag] + 1) / (N_tags + len(TAGS))

        # calculate emission prob.
        for tag in TAGS:
            words_dict = E[tag]
            N_sub_words = sum(list(words_dict.values()))
            tmp_dict = defaultdict(lambda : 1 / (N_sub_words + len(VOC)))
            for word, count in words_dict.items():
                tmp_dict[word] = (count + 1) / (N_sub_words + len(VOC))
            E[tag] = tmp_dict

        # calculate transition prob.
        for tag in TAGS:
            next_tag_dict = T[tag]
            N_sub_tags = sum(list(next_tag_dict.values()))
            tmp_dict = defaultdict(lambda : 1 / (N_sub_tags + len(TAGS)))
            for next_tag, count in next_tag_dict.items():
                tmp_dict[next_tag] = (1 + count) / (N_sub_tags + len(TAGS))
            T[tag] = tmp_dict

        self.tags = list(TAGS)
        self.voc = list(VOC)

        return PI, T, E


    def viterbi_decode(self, sentence):

        sentence = sentence.split(' ') + ["<END>"]
        V = []
        B = []
        for t, word in enumerate(sentence):
            v = []
            b = []
            for i, tag_i in enumerate(self.tags):
                if t == 0:
                    v.append(math.log10(self.PI[tag_i] * self.E[tag_i][word]))
                else:
                    max_p = float('-inf')
                    max_j = 0
                    for j, v_j in enumerate(V[-1]):
                        p_t = math.log10(self.T[self.tags[j]][tag_i])
                        p_e = math.log10(self.E[tag_i][word])
                        if v_j + p_t + p_e > max_p:
                            max_p = v_j + p_t + p_e
                            max_j = j
                    v.append(max_p)
                    b.append(max_j)
            V.append(v)
            B.append(b)

        ret = []
        ret.append(V[-1].index(max(V[-1])))
        for i in range(len(sentence) - 1):
            ret.insert(0, B[-1][ret[0]])
            B.pop(-1)
        ret = [self.tags[t] for t in ret]

        return ret[:-1]

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='train')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    dataset = Dataset(args.train)

    test_sentences = [
        'the secretariat is expected to race tomorrow',
        'people continue to enquire the reason for the race for outer space'
    ]
    for sentence in test_sentences:
        print("* input sentence: %s" % sentence)
        print("* output tags:", dataset.viterbi_decode(sentence))