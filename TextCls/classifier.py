import tabulate
import math


class TextClassifier(object):

    def __init__(self, train_ds, test_ds, method='normal'):

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.method = method

    def _predict(self, x):

        p_ham = 0
        p_spam = 0
        for word in x:
            p_ham += math.log10(self.train_ds.p_w_c(word, 'ham'))
            p_spam += math.log10(self.train_ds.p_w_c(word, 'spam'))

        return 'ham' if p_ham > p_spam else 'spam'

    def evaluate(self):

        confusion_matrix = {
            'ham':{'ham':0, 'spam':0},
            'spam':{'ham':0, 'spam':0}} # rows-groundtruth, cols-predictions
        for text_label, text_content in self.test_ds.get_data().items():
            for text in text_content:
                confusion_matrix[text_label][self._predict(text)] += 1

        # vis
        headers = ['/', 'ham', 'spam', 'precision_ham', 'precision_spam']
        tp = confusion_matrix['ham']['ham']
        fn = confusion_matrix['spam']['spam']
        tn = confusion_matrix['ham']['spam']
        fp = confusion_matrix['spam']['ham']
        table = [
            ['ham', tp, tn, tp / (tp + tn), '/'],
            ['spam', fp, fn, '/', fn / (fp + fn)],
            ['recall_ham', tp / (tp + fp), '/', '/', '/'],
            ['recall_spam', '/', fn / (fn + tn) ,'/', '/'],
        ]
        tab = tabulate.tabulate(table, headers)
        print(tab)

        precision = tp / (tp + tn)
        recall = tp / (tp + fp)
        f1 = 2 / (1/precision + 1/recall)
        return f1