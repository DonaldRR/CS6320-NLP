# POS Tagging with RNN
This project implements the task of POS Tagging with biLSTM.

## Environment
Run the code with `Python3.5` or later version, refer to `requirements.txt` for required packages.

## Run
### Prequisite
Please download the the GloVe pre-trained word embeddings(from https://github.com/stanfordnlp/GloVe). And put the embeddings txt files in directory `pretrained`.
### Train
```
python main.py --train $TRAIN_DATASET --embd $WORD_EMBEDDINGS_FILE --epoch $EPOCH --batch-size $BATCH_SIZE
```
### Test
```
python main.py --eval --ckpt $MODEL_CHECKPOINT_PATH --embd $WORD_EMBEDDINGS_FILE
```
#### Expected
Test cases are:
* `people continue to enquire the reason for the race for outer space`
* `the secretariat is expected to race tomorrow`
Expected outputs are:
* `['NOUN', 'VERB', 'X', 'VERB', 'DETERMINER', 'NOUN', 'PREPOSITION', 'DETERMINER', 'NOUN', 'PREPOSITION', 'ADJECTIVE', 'NOUN']`
* `['DETERMINER', 'NOUN', 'VERB', 'VERB', 'X', 'NOUN', 'NOUN']`
