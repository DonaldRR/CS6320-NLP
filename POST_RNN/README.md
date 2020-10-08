# POS Tagging with RNN
This project implements the task of POS Tagging with biLSTM.

## Environment
Run the code with `Python3.5` or later version, refer to `requirements.txt` for required packages.

## Run
### Prequisite
Please download the the GloVe pre-trained word embeddings(from https://github.com/stanfordnlp/GloVe). And push embeddings txt files in directory `pretrained`.
### Train
```
python main.py --train $TRAIN_DATASET --embd $WORD_EMBEDDINGS_FILE --epoch $EPOCH --batch-size $BATCH_SIZE
```
