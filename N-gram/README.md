# CS6320: Natural Language Processing -- Homework1: N-grams
A simple program for building n-gram language model.

## Prerequisite
Run under `Python3` with packages: `tabulate`, `tqdm`

## Run
```
python main.py --train TRAIN_FILE --test TEST_FILE --n N [--smooth]
```
###Arguments
* train: location of training file
* test: location of test file
* n: n for n-gram, e.g. 1 for unigram, 2 for bi-gram, so on and so forth
* smooth: use add-one smooth or not
