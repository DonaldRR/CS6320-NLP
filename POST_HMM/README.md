# POS tagging with HMM

POS tagging task implemented with first-order HMM assumption.

## Environment
Python3.6+

## Run
```
python main.py --train $TRAINING_SET_DIR
```

## Test
For the following two test cases
* `the secretariat is expected to race tomorrow`
* `people continue to enquire the reason for the race for outer space`

The expected predicted taggings are respectively:
* `['DETERMINER', 'NOUN', 'VERB', 'VERB', 'PREPOSITION', 'NOUN', 'PUNCT']`
* `['NOUN', 'VERB', 'X', 'VERB', 'DETERMINER', 'NOUN', 'PREPOSITION', 'DETERMINER', 'NOUN', 'PREPOSITION', 'ADJECTIVE', 'NOUN']`
