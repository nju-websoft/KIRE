import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from knowledge_injection_layer.modules import SpGAT


class Attr_triple_enc(nn.Module):
    def __init__(self, ent_hidden_dim):
        super(Attr_triple_enc, self).__init__()
        self.autoenc = AutoEncoderModel(50)
        self.autoenc.load_state_dict(torch.load('./knowledge_injection_layer/ae_results/re.model'), strict=False)
        for child in self.autoenc.children():
            for param in child.parameters():
                param.requires_grad = False

        self.kernel_size = 3
        self.stride = 1
        self._1dcnn = nn.Conv1d(in_channels=ent_hidden_dim, out_channels=ent_hidden_dim, kernel_size=self.kernel_size, stride=self.stride)#, padding=self.padding)
        nn.init.xavier_normal_(self._1dcnn.weight.data, gain=1.414)

        self.dropout = nn.Dropout(0.5)
        self.ent_hidden_dim = ent_hidden_dim

    def forward(self, entity_attrs, attrs_nums, entity_attr_lens):
        """
        :param entity_attrs: <batch_attr_nums, max_attr_len>
        :param attrs_nums: [[2,8], [3,3,3],[4,4]]
        :param entity_attr_lens: <batch_attr_nums, 1>
        :return: output_attr_hiddens <batch_size, entity_size, -1>
        """
        entity_nums = [len(attrs_num) for attrs_num in attrs_nums]
        attrs_nums = [attr for attrs_num in attrs_nums for attr in attrs_num]
        encoded_attr_hiddens = self.autoenc.encode(entity_attrs, entity_attr_lens)

        encoded_attr_hiddens = pad_sequence(torch.split(encoded_attr_hiddens, attrs_nums), batch_first=True)
        _encoded_attr_hiddens = encoded_attr_hiddens.permute(0, 2, 1)  # entity_nums, -1, attr_size
        _encoded_attr_hiddens = F.relu(self._1dcnn(_encoded_attr_hiddens))
        output_attr_hiddens = torch.max(_encoded_attr_hiddens, dim=2)[0]

        output_attr_hiddens = pad_sequence(torch.split(output_attr_hiddens, entity_nums),
                                           batch_first=True)  # batch_size, entity_size, -1

        return output_attr_hiddens


class Rela_triple_enc(nn.Module):

    def __init__(self, ent_hidden_dim, rel_hidden_dim, gcn_layer_nums, gcn_head_nums, gcn_in_drop):
        super(Rela_triple_enc, self).__init__()
        self.attr_encoder = Attr_triple_enc(ent_hidden_dim)
        self.gcn_layer = SpGAT(ent_hidden_dim, ent_hidden_dim, gcn_layer_nums, gcn_in_drop, 0.1, rel_hidden_dim, gcn_head_nums)

    def forward(self, entity_attrs, attrs_nums, entity_attr_lens, kg_adj, entity_doc_nodeid, kg_adj_edges=None):
        batch_size = entity_doc_nodeid.shape[0]

        attr_hiddens = self.attr_encoder(entity_attrs, attrs_nums, entity_attr_lens)  # batch_size * node_size * -1 <16,3344,100>
        ent_kg_hiddens = self.gcn_layer(attr_hiddens, kg_adj, attr_hiddens, kg_adj_edges)

        entity_doc_nodeid = entity_doc_nodeid + 1
        entity_doc_nodeid[entity_doc_nodeid.eq(-99)] = 0
        ent_kg_hiddens_padding = torch.zeros(ent_kg_hiddens.size(0), 1, ent_kg_hiddens.size(2)).to(entity_attrs.device)
        new_ent_kg_hiddens = torch.cat([ent_kg_hiddens_padding, ent_kg_hiddens], dim=1)
        kg_input_ent = new_ent_kg_hiddens[torch.arange(batch_size).unsqueeze(-1), entity_doc_nodeid]
        return kg_input_ent, ent_kg_hiddens


class AutoEncoderModel(nn.Module):
    def __init__(self, lstm_units, go=194784, train_embeddings=False, bidirectional=True, gpuid=1):
        """
            Initialize the encoder/decoder and creates Tensor objects
            :param lstm_units: number of LSTM units
            :param embeddings: numpy array with initial embeddings
            :param go: index of the GO symbol in the embedding matrix
            :param train_embeddings: whether to adjust embeddings during training
            :param bidirectional: whether to create a bidirectional autoencoder
            """
        # EOS and GO share the same symbol. Only GO needs to be embedded, and
        # only EOS exists as a possible network output
        super(AutoEncoderModel, self).__init__()

        self.embedding_size = 100
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.go = go
        self._init_encoder()


    def _init_encoder(self):
        self.embeddings = nn.Embedding(194785, self.embedding_size)

        self.lstm_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.lstm_units, num_layers=1,
                                    batch_first=True, bidirectional=self.bidirectional)

    def encode(self, sentences, lens):
        """
        :param sentences:
        :param lens:
        :return:
        """
        embedded = self.embeddings(sentences)
        word_vec_packed = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        encoded_sent_outputs, (encoded_sent_hiddens, encoded_sent_cells) = self.lstm_encoder(word_vec_packed)  # batch， max_len, -1
        encoded_sent_outputs, _ = pad_packed_sequence(encoded_sent_outputs, batch_first=True)
        encoded_sent_hiddens = encoded_sent_hiddens.permute(1, 0, 2).reshape(embedded.size(0), -1)
        return encoded_sent_hiddens

    def _generate_batch_go(self, like):
        """
        Generate a 1-d tensor with copies of EOS as big as the batch size,
        :param like: a tensor whose shape the returned embeddings should match
        :return: a tensor with shape as `like`
        """
        ones = torch.ones_like(like).cuda()
        return ones * self.go
