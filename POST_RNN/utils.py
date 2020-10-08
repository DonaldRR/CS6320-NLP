import numpy as np
import torch
from torch import nn


def evaluate(model, input, dataset):
    model.eval()
    with torch.no_grad():
        input, _ = dataset.preprocess(input)
        input = input.cuda()
        pred = model(input).detach().cpu().numpy()[0]

        return dataset.postprocess(pred)


def compute_confusion_matrix(pred, label, n_class):
    pred = np.argmax(pred, axis=1)

    m = np.zeros((n_class, n_class))
    pred = pred * n_class + label
    pred = pred[label >= 0]
    count = np.bincount(pred)
    for i, c in enumerate(count):
        m[i // n_class][i % n_class] = c

    return m

def load_pretrained_embeddings(path):

    words = []
    embeddings = []
    # Read from embeddings text file
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        for line in lines:
            words.append(line[0])
            embeddings.append(list(map(lambda x: float(x), line[1:])))
    words = {
        word:i + 2
        for i, word in enumerate(words)
    }

    # Add dummy embedding
    words[' '] = 0
    embeddings.insert(0, np.zeros(len(embeddings[0])))
    # Add out-of-vocabulary embedding
    words['*'] = 1
    embeddings.insert(1, np.mean(embeddings[1:], axis=0))

    embeddings = np.array(embeddings)

    return {
        'embeddings': embeddings,
        'word2idx': words,
        'n_voc': embeddings.shape[0],
        'n_dim': embeddings.shape[1]
    }
