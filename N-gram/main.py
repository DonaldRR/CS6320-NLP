import numpy as np
from collections import defaultdict
import argparse
import tqdm
from tabulate import tabulate

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='./train.txt')
    parser.add_argument('--test', type=str, default='./test.txt')
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--smooth', action='store_true', default=False)
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    n_tokens = 0
    n_grams = {}
    for n in range(args.n):
        n_grams[n] = defaultdict(lambda : 0)

    print('* Preprocessing corpus ... ', end='')
    with open(args.train, 'r') as f:

        lines = f.readlines()
        lines = list(map(lambda x: x.strip().split(' '), lines))
        for line in tqdm.tqdm(lines):
            n_tokens += len(line)
            for t, word in enumerate(line):

                for n in n_grams.keys():
                    if t + n < len(line):
                        if n == 0:
                            n_grams[n][line[t]] += 1
                        else:
                            n_grams[n][tuple(line[t:t+n+1])] += 1
    print('Finished.')

    print('#tokens=%d' % n_tokens)
    for k, v in n_grams.items():
        print('#%d-gram=%d' % (k, len(v)))

    n_types = len(n_grams[0])
    print('* Testing')
    with open(args.test, 'r') as f:

        lines = f.readlines()
        lines = list(map(lambda x: x.strip().split(' '), lines))

        for line in lines:
            p = 1
            table = []
            for t in range(len(line)):
                prev_len = min(t, args.n - 1)
                while prev_len >= 0:
                    if prev_len > 0:
                        if prev_len == 1:
                            prev_n_gram = line[t - 1]
                        else:
                            prev_n_gram = tuple(line[t - prev_len:t])
                        n_gram = tuple(line[t - prev_len:t + 1])
                        if n_grams[prev_len - 1][prev_n_gram] > 0:
                            #print('p(%s|%s)=%d/%d=%f' %
                            #      (
                            #          n_gram[-1],
                            #          n_gram[:-1],
                            #          n_grams[prev_len][n_gram],
                            #          n_grams[prev_len - 1][prev_n_gram],
                            #          n_grams[prev_len][n_gram] / n_grams[prev_len - 1][prev_n_gram])
                            #      )
                            if args.smooth:
                                p_cond = (n_grams[prev_len][n_gram] + 1) / (n_grams[prev_len - 1][prev_n_gram] + n_types)
                            else:
                                p_cond = n_grams[prev_len][n_gram]/ n_grams[prev_len - 1][prev_n_gram]
                            table.append([n_gram, n_grams[prev_len][n_gram], p_cond])
                            p *= p_cond
                            break
                    else:
                        #print('p(%s)=%d/%d=%f' %
                        #      (
                        #          line[t],
                        #          n_grams[0][line[t]],
                        #          n_tokens,
                        #          n_grams[0][line[t]] / n_tokens)
                        #      )
                        p_prior = n_grams[0][line[t]] / n_tokens
                        table.append([line[t], n_grams[0][line[t]], p_prior])
                        p *=p_prior
                    prev_len -= 1
            print(tabulate(table, headers=['N-gram', 'Count', 'Probability']))
            print('p=', p)