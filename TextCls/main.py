import argparse
import os
from collections import defaultdict
import math
import tabulate
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

from dataset import TextDataset
from classifier import TextClassifier

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='train')
    parser.add_argument('--test', type=str, default='test')
    parser.add_argument('--method', type=str, default='MI')
    parser.add_argument('--stopwords', type=str, default=None)
    args = parser.parse_args()

    return args
args = parse_args()





if __name__ == '__main__':

    print('-- Loading test set -- ')
    test_ds = TextDataset(args.test, split='test', stopwords_path=args.stopwords)
    print('-- Loading training set --')
    train_ds = TextDataset(args.train, split='train', stopwords_path=args.stopwords, method=args.method, n_features=8800)
    clf = TextClassifier(train_ds, test_ds, args.method)
    print('-- Evaluating --')
    f1 = clf.evaluate()
    print('F1 score:', f1)
    #ret = []
    #for n in tqdm(range(6000, 10000, 200)):
    #    print('-- Loading training set --')
    #    train_ds = TextDataset(args.train, split='train', stopwords_path=args.stopwords, method=args.method, n_features=n)
    #    clf = TextClassifier(train_ds, test_ds, args.method)
    #    print('-- Evaluating --')
    #    f1 = clf.evaluate()
    #    ret.append(f1)
    #    print('-- f1=%.4f, n=%d --' % (f1, n))
    #print('* Max F1=%.4f with %d features selected' % (max(ret), 10 + 5 * ret.index(max(ret))))
