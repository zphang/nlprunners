import numpy as np
import sacremoses
import torch
import torch.nn as nn
from dataclasses import dataclass

from nlpr.tasks.core import FeaturizationSpec

from pyutils.display import maybe_tqdm
import pyutils.io as io


@dataclass
class GloveLSTMConfig:
    hidden_dim: int
    num_layers: int
    vocab_size: int
    drop_prob: float

    @classmethod
    def from_json(cls, path):
        return cls(**io.read_json(path))


class GloVeEmbeddings:
    UNK = "<UNK>"
    CLS = "<CLS>"
    SEP = "<SEP>"
    PAD = "<PAD>"
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
            for line in maybe_tqdm(f, total=vocab_size, verbose=verbose, desc="GloVe"):
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
            padding_idx=0,
        )
        self.special_embeddings = nn.Embedding(
            num_embeddings=len(self.glove.special_list) + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
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
    def __init__(self, hidden_dim, num_layers, num_classes, num_inputs,
                 glove_embedding, drop_prob=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_inputs = num_inputs
        self.glove_embedding = glove_embedding
        self.drop_prob = drop_prob

        self.has_lstm = self.num_layers > 0

        # Kind of hack-ish to handle BOW/LSTM, and single vs double input
        if self.has_lstm:
            self.lstm1 = nn.GRU(
                input_size=glove_embedding.embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=True,
            )
            if self.num_inputs == 2:
                self.lstm2 = nn.GRU(
                    input_size=glove_embedding.embedding_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    num_layers=num_layers,
                    bidirectional=True,
                )
                self.fc1 = nn.Linear(hidden_dim * 2 * 4, hidden_dim)
            else:
                self.lstm2 = None
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.lstm1 = None
            self.lstm2 = None
            if self.num_inputs == 2:
                self.fc1 = nn.Linear(300 * 4, hidden_dim)
            else:
                self.fc1 = nn.Linear(300, hidden_dim)
        # two inputs, and bidirectional
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input1, input2, lengths1, lengths2):
        if input2 is None and lengths2 is None:
            pooled = self.pool_input(
                input_ids=input1,
                lengths=lengths1,
            )
        else:
            pooled = self.pool_double_input(
                input1=input1,
                input2=input2,
                lengths1=lengths1,
                lengths2=lengths2,
            )

        pooled = self.fc1(self.relu(pooled))
        pooled = self.dropout(pooled)
        logits = self.fc2(self.relu(pooled))
        return logits, pooled  # Follow PTT format of returning tuple

    def pool_double_input(self, input1, input2, lengths1, lengths2):
        embeddings1 = self.glove_embedding(input1)
        lstm_out1 = self.get_lstm_output_v2(self.lstm1, embeddings1, lengths1)
        pooled1 = self.masked_average(lstm_out1, lengths1)

        embeddings2 = self.glove_embedding(input2)
        lstm_out2 = self.get_lstm_output_v2(self.lstm2, embeddings2, lengths2)
        pooled2 = self.masked_average(lstm_out2, lengths2)

        pooled = torch.cat([pooled1, pooled2, pooled1-pooled2, pooled1*pooled2], dim=1)
        return pooled

    def pool_input(self, input_ids, lengths):
        embeddings = self.glove_embedding(input_ids)
        lstm_out = self.get_lstm_output_v2(self.lstm1, embeddings, lengths)
        pooled = self.masked_average(lstm_out, lengths)
        return pooled

    @classmethod
    def get_lstm_output(cls, lstm, embeddings, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths,
            batch_first=True, enforce_sorted=False,
        )
        batch_size = embeddings.shape[0]
        lstm_out = lstm(packed)[1][0]
        lstm_out = lstm_out.transpose(0, 1).reshape(batch_size, -1)
        return lstm_out

    def get_lstm_output_v2(self, lstm, embeddings, lengths):
        if not self.has_lstm:
            return embeddings

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_lstm_out = lstm(packed)
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out[0],
            batch_first=True,
            total_length=embeddings.shape[1],
        )
        return lstm_out

    @classmethod
    def masked_average(cls, inputs, lengths):
        mask = torch.zeros(inputs.shape[:2]).unsqueeze(-1).long()
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        mask = mask.to(inputs.device).float()
        numerator = (inputs * mask).sum(1)
        denominator = mask.sum(1).sum(1).clamp(1).unsqueeze(1)
        return numerator / denominator


class GloveLSTMModel(nn.Module):
    def __init__(self, model: GloveLSTMModelBase):
        super().__init__()
        self.model = model

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        input1_tensor, input2_tensor, lengths1, lengths2 = self.modify_input(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return self.model(
            input1=input1_tensor,
            input2=input2_tensor,
            lengths1=lengths1,
            lengths2=lengths2,
        )

    @classmethod
    def modify_input(cls, input_ids, token_type_ids, attention_mask):
        input_ids_arr = input_ids.cpu().numpy()
        segment_ids_arr = (attention_mask + token_type_ids).cpu().numpy()
        num_segments = segment_ids_arr.max()
        if num_segments == 1:
            return cls.modify_input_single(
                input_ids_arr=input_ids_arr,
                segment_ids_arr=segment_ids_arr,
                device=input_ids.device,
            )
        elif num_segments == 2:
            return cls.modify_input_double(
                input_ids_arr=input_ids_arr,
                segment_ids_arr=segment_ids_arr,
                device=input_ids.device,
            )
        else:
            raise RuntimeError()

    @classmethod
    def modify_input_single(cls, input_ids_arr, segment_ids_arr, device):
        batch_size = input_ids_arr.shape[0]

        input_ls = []
        for i in range(batch_size):
            input_ls.append(input_ids_arr[i][segment_ids_arr[i] == 1])

        padded, lengths = get_padded_ids(input_ls)
        input_tensor = torch.tensor(padded).long().to(device)
        return input_tensor, None, lengths, None

    @classmethod
    def modify_input_double(cls, input_ids_arr, segment_ids_arr, device):
        batch_size = input_ids_arr.shape[0]

        input1_ls = []
        input2_ls = []
        for i in range(batch_size):
            input1_ls.append(input_ids_arr[i][segment_ids_arr[i] == 1])
            input2_ls.append(input_ids_arr[i][segment_ids_arr[i] == 2])

        padded1, lengths1 = get_padded_ids(input1_ls)
        padded2, lengths2 = get_padded_ids(input2_ls)
        input1_tensor = torch.tensor(padded1).long().to(device)
        input2_tensor = torch.tensor(padded2).long().to(device)
        return input1_tensor, input2_tensor, lengths1, lengths2


class GloveLSTMForSequenceClassification(GloveLSTMModel):
    pass


class GloveLSTMForSequenceRegression(GloveLSTMModel):
    pass


def get_padded_ids(token_id_ls_ls):
    zeros = np.zeros((len(token_id_ls_ls), max(map(len, token_id_ls_ls))), dtype=int)
    lengths = []
    for i, token_ids in enumerate(token_id_ls_ls):
        length = len(token_ids)
        zeros[i, :length] = token_ids
        lengths.append(length)
    return zeros, lengths


def get_bilstm_last(lstm_out, hidden_dim):
    fwd = lstm_out[:, -1, :hidden_dim]
    bwd = lstm_out[:, 0, hidden_dim:]
    return torch.cat([fwd, bwd], dim=1)
