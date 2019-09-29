import numpy as np
import sacremoses
import torch
import torch.nn as nn

from nlpr.tasks.core import FeaturizationSpec

from pyutils.display import maybe_tqdm


class GloVeEmbeddings:
    UNK = "<UNK>"
    CLS = "<CLS>"
    SEP = "<SEP>"
    PAD = "<PAD"
    SPECIAL_LIST = [PAD, CLS, SEP, UNK]

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.word_list = list(embeddings)
        self.special_list = self.SPECIAL_LIST
        for special_token in self.special_list:
            assert special_token not in self.embeddings
        self.full_token_list = self.special_list + self.word_list
        self.id_token_map = self.full_token_list
        self.token_id_map = {token: i for i, token in enumerate(self.id_token_map)}
        self.tokenizer = sacremoses.MosesTokenizer(lang='en')

    def tokenize(self, string):
        tokens = self.tokenizer.tokenize(string)
        return [token if token in self.embeddings else self.UNK for token in tokens]

    @property
    def sep_token(self):
        return self.SEP

    @property
    def cls_token(self):
        return self.CLS

    @property
    def pad_id(self):
        return 0

    @classmethod
    def read_glove(cls, path, vocab_size=None, verbose=False):
        embeddings = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in maybe_tqdm(f, total=vocab_size, verbose=verbose):
                if vocab_size is not None and len(embeddings) == vocab_size:
                    break
                word, vec = line.split(" ", 1)
                embeddings[word] = np.array(list(map(float, vec.split())))
        return cls(embeddings)

    @staticmethod
    def get_feat_spec(max_seq_length):
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
            sep_token_extra=False,
        )

    def convert_tokens_to_ids(self, tokens):
        return [self.token_id_map[token] for token in tokens]

    def convert_ids_to_tokens(self, idxs):
        return [self.id_token_map[idx] for idx in idxs]


class GloVeEmbeddingModule(nn.Module):
    def __init__(self, glove):
        super().__init__()
        self.glove = glove
        self.embedding_dim = len(self.glove.embeddings[self.glove.word_list[0]])
        self.word_embeddings = nn.Embedding(
            num_embeddings=len(self.glove.word_list) + 1,
            embedding_dim=self.embedding_dim,
        )
        self.special_embeddings = nn.Embedding(
            num_embeddings=len(self.glove.special_list) + 1,
            embedding_dim=self.embedding_dim,
        )
        self.word_embeddings.weight.requires_grad = False
        self.special_embeddings.weight.requires_grad = True
        self.word_embeddings.weight[0].zero_()
        self.word_embeddings.weight[1:] = torch.tensor(np.array([
            vector for vector in glove.embeddings.values()
        ]))
        self.special_embeddings.weight.data[0].zero_()
        self.num_special = len(self.glove.special_list)

    def forward(self, token_ids):
        is_special = (token_ids < self.num_special).long()
        is_word = 1 - is_special
        special_embedding_ids = (token_ids + 1) * is_special
        word_embedding_ids = (token_ids + 1 - self.num_special) * is_word

        embedded_specials = self.special_embeddings(special_embedding_ids)
        embedded_words = self.word_embeddings(word_embedding_ids)
        return embedded_words + embedded_specials


class GloveLSTMModelBase(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_classes, glove_embedding, drop_prob=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.glove_embedding = glove_embedding
        self.drop_prob = drop_prob

        self.lstm = nn.LSTM(
            input_size=glove_embedding.embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # bidirectional, so *2
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids):
        embeddings = self.glove_embedding(input_ids)
        lstm_out = self.lstm(embeddings)[0]  # get hidden out
        pooled = lstm_out.mean(dim=1)
        # pooled = lstm_out[:, -1]
        pooled = self.fc1(self.relu(pooled))
        pooled = self.dropout(pooled)
        logits = self.fc2(self.relu(pooled))
        return logits, None  # Follow PTT format of returning tuple


class GloveLSTMModel(nn.Module):
    def __init__(self, model: GloveLSTMModelBase):
        super().__init__()
        self.model = model

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        # Ignore token_type_ids, attention_mask, labels
        return self.model(self.cut_padding(input_ids))

    def cut_padding(self, input_ids):
        not_padding = (input_ids.detach() != self.model.glove_embedding.glove.pad_id)
        match = np.where(not_padding.sum(0).cpu().numpy() == 0)
        if len(match[0]) == 0:
            input_ids = input_ids
        else:
            input_ids = input_ids[:, :match[0][0]]
        return input_ids


class GloveLSTMForSequenceClassification(GloveLSTMModel):
    pass
