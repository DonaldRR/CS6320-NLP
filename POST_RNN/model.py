import torch
from torch import nn


class POSTModel(torch.nn.Module):

    def __init__(self, pretrained_embeddings, num_class, hidden_dim=32):
        super(POSTModel, self).__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.N_voc, self.input_dim = pretrained_embeddings.shape

        self.embedding = self._load_embeddings(pretrained_embeddings)
        self.biLSTM = nn.LSTM(input_size=self.input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.bn = nn.BatchNorm1d(2 * self.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=False),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_dim, self.num_class, bias=False),
        )

    def _load_embeddings(self, pretrained_embeddings):

        embedding = nn.Embedding(num_embeddings=self.N_voc,
                                 embedding_dim=self.input_dim)
        embedding.from_pretrained(torch.tensor(pretrained_embeddings))

        return embedding

    def forward(self, x):

        """
        :param x: (B, L, C)
        :return:
        """

        # B, L, C
        x = self.embedding(x)
        # B, L, 2*C
        x, _ = self.biLSTM(x)
        x = self.bn(x.permute(0, 2, 1))
        # B, L, n_class
        x = self.classifier(x.permute(0, 2, 1))

        return x
