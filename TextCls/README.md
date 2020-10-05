# Spam/Ham mail Text Classifier
A Text Classifier implemented with Multinomial Naive Bayes basis and the mutual information feature selection strategy. Train/Test data and stopwords is also included.

## Environment
Run with `Python3.6` and required packages can be shown in `requirement.txt`

## Run
```
python main.py --train $TRAINING_SET_PATH --test $TEST_SET_PATH --stopwords $STOPWORDS_PATH [--method $FEATURE_SELECTION_METHOD] 
```bash
