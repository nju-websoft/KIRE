"""
输入每篇文档的kg 子图， 每个节点的attr_encoder输出作为各节点初始化信息，输出作为实体kg表示

r ==> 所有h-t 的平均
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from knowledge_injection_layer.config import Config as kgconfig

from knowledge_injection_layer.modules import GCN, DSGCN, SpGAT, RGCN

# todo 属性编码器换成自编码器
"""
当attr_encoder中存在可训练变量和 模型一起训练时，导致模型过多关注attr_encoder层，导致dev效果还行，但是test上效果很差
"""
class Attr_encoder(nn.Module):
    def __init__(self, pembeds, kg_freeze_words, ent_hidden_dim, gpuid):
        """
        :param pre_embed: 预训练的embedding矩阵 numpy
        :param kg_freeze_words: 是否训练词向量
        :param ent_hidden_dim： kg实体向量维度
        :param gpuid
        """
        super(Attr_encoder, self).__init__()
        self.device = torch.device("cuda" if gpuid != -1 else "cpu")
        if not kgconfig.attr_module or kgconfig.attr_encode_type == 'max':
            self.word_embed = nn.Embedding.from_pretrained(pembeds, freeze=True)
            # todo 加入字符编码，缓解大量未登录词
            self.lstm_encoder = nn.LSTM(input_size=pembeds.shape[1], hidden_size=ent_hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        else:
            # eos_index = pembeds.size(0)
            # new_pembeds = torch.cat([pembeds, torch.zeros((1,100))], dim=0)
            self.autoenc = AutoEncoderModel(50)
            self.autoenc.load_state_dict(torch.load('./knowledge_injection_layer/ae_results/re.model'), strict=False)
            #self.autoenc.load_state_dict(torch.load('./remodel/pytorch_model.bin'), strict=False)
            print("load something here kg_encoder")
            for child in self.autoenc.children():  # 不参与训练好
                for param in child.parameters():
                    param.requires_grad = False
        self.kernel_size = 3
        self.stride = 1
        # self.padding = int((self.kernel_size - 1) / 2)  # 保证输出长度不变
        self._1dcnn = nn.Conv1d(in_channels=ent_hidden_dim, out_channels=ent_hidden_dim, kernel_size=self.kernel_size, stride=self.stride)#, padding=self.padding)
        nn.init.xavier_normal_(self._1dcnn.weight.data, gain=1.414)
        # 使用2dcnn
        # _conv = tf.nn.l2_normalize(_conv, 2)

        self.dropout = nn.Dropout(0.5)
        self.ent_hidden_dim = ent_hidden_dim

    def forward(self, entity_attrs, attrs_nums, entity_attr_lens):
        """
        :param entity_attrs: <batch_attr_nums, max_attr_len>
        :param attrs_nums: [[2,8], [3,3,3],[4,4]]
        :param entity_attr_lens: <batch_attr_nums, 1>  batch_attr_nums = 该batch中总的属性三元组个数
        :return: output_attr_hiddens <batch_size, entity_size, -1>
        """
        entity_nums = [len(attrs_num) for attrs_num in attrs_nums]
        attrs_nums = [attr for attrs_num in attrs_nums for attr in attrs_num]

        if not kgconfig.attr_module:  # 仅使用label 通过max_pooling方式获取
            _word_vec = self.word_embed(entity_attrs)
            encoded_attr_hiddens = torch.max(_word_vec, dim=1)[0]
            encoded_attr_hiddens = pad_sequence(torch.split(encoded_attr_hiddens, attrs_nums), batch_first=True)
            output_attr_hiddens = encoded_attr_hiddens[:, 0, :]

        else:

            if kgconfig.attr_encode_type == 'max':
                _word_vec = self.word_embed(entity_attrs)
                # _word_vec = F.normalize(_word_vec, dim=-1)
                # word_vec = F.dropout(_word_vec, 0.2, self.training)
                word_vec = _word_vec

                # encode_with_rnn
                if kgconfig.attr_encode_lstm:  # 加了lstm 似乎没啥用
                    word_vec_packed = pack_padded_sequence(word_vec, entity_attr_lens, batch_first=True,
                                                           enforce_sorted=False)
                    encoded_attr_outputs, (encoded_attr_hiddens, _) = self.lstm_encoder(
                        word_vec_packed)  # batch_attr_nums， max_attr_len, -1
                    encoded_attr_outputs, _ = pad_packed_sequence(encoded_attr_outputs, batch_first=True)
                    # encoded_attr_hiddens = encoded_attr_hiddens.permute(1, 0, 2).contiguous().view(entity_attrs.size(0), -1)
                    encoded_attr_hiddens = torch.max(encoded_attr_outputs, dim=1)[0]
                else:
                    encoded_attr_hiddens = torch.max(word_vec, dim=1)[0]
            else:
                encoded_attr_hiddens = self.autoenc.encode(entity_attrs, entity_attr_lens)

            encoded_attr_hiddens = pad_sequence(torch.split(encoded_attr_hiddens, attrs_nums), batch_first=True)
            # encoded_attr_hiddens = F.normalize(encoded_attr_hiddens, dim=-1)

            if kgconfig.attr_label_flag and kgconfig.attr_description_flag and kgconfig.attr_instance_flag and kgconfig.attr_alias_flag:
                _encoded_attr_hiddens = encoded_attr_hiddens.permute(0, 2, 1)  # entity_nums, -1, attr_size
                _encoded_attr_hiddens = F.relu(self._1dcnn(_encoded_attr_hiddens))
                # _encoded_attr_hiddens = F.normalize(_encoded_attr_hiddens, dim=1)
                output_attr_hiddens = torch.max(_encoded_attr_hiddens, dim=2)[0]
            elif kgconfig.attr_label_flag:  # 属性分解实验，同一时刻确保只有一个为ture
                output_attr_hiddens = encoded_attr_hiddens[:, 0, :]
            elif kgconfig.attr_description_flag:
                output_attr_hiddens = encoded_attr_hiddens[:, 1, :]
            elif kgconfig.attr_instance_flag:
                output_attr_hiddens = encoded_attr_hiddens[:, 2, :]
            elif kgconfig.attr_alias_flag:
                output_attr_hiddens = encoded_attr_hiddens[:, 3, :]
            else:
                print("属性类型个数错误")
                exit(-1)
        output_attr_hiddens = pad_sequence(torch.split(output_attr_hiddens, entity_nums),
                                           batch_first=True)  # batch_size, entity_size, -1
        # output_attr_hiddens = F.normalize(output_attr_hiddens, dim=-1)

        return output_attr_hiddens


class Kg_Encoder(nn.Module):

    def __init__(self, pembeds, kg_freeze_words, ent_hidden_dim, gpuid, gcn_layer_nums, gcn_in_drop, gcn_out_drop, gcn_type):
        super(Kg_Encoder, self).__init__()
        self.device = torch.device("cuda" if gpuid != -1 else "cpu")
        self.attr_encoder = Attr_encoder(pembeds, kg_freeze_words, ent_hidden_dim, gpuid)
        if gcn_type == 'GCN':
            self.gcn_layer = GCN(ent_hidden_dim, ent_hidden_dim, gcn_layer_nums, gpuid, gcn_in_drop, gcn_out_drop)
        elif gcn_type == 'RGCN':
            self.gcn_layer = RGCN(ent_hidden_dim, ent_hidden_dim, gcn_layer_nums, 1060, gpuid, gcn_in_drop, gcn_out_drop)
        elif gcn_type == 'DSGCN':
            self.gcn_layer = DSGCN(ent_hidden_dim, ent_hidden_dim, gcn_layer_nums, gpuid, gcn_in_drop, gcn_out_drop)
        elif gcn_type in ['GAT', 'GAT_WORDREP', 'GAT_RELREP']:
            self.gcn_layer = SpGAT(ent_hidden_dim, ent_hidden_dim, gcn_layer_nums, gpuid, gcn_in_drop, 0.1, gcn_type, kgconfig.rel_hidden_dim, kgconfig.gcn_head_nums)
        # self.LayerNorm0 = nn.LayerNorm(ent_hidden_dim, eps=1e-5)
        # self.LayerNorm1 = nn.LayerNorm(ent_hidden_dim, eps=1e-5)
        self.gcn_type = gcn_type

    def forward(self, entity_attrs, attrs_nums, entity_attr_lens, kg_adj, entity_doc_nodeid, kg_adj_edges=None, kg_radj=None):
        """

        Args:
            entity_attrs:
            attrs_nums: [[2,8], [3,3,3],[4,4]]
            entity_attr_lens:
            kg_adj: list [[sparse_tensor]] [batch_size, <node_size, node_size>>  # node_size = entity_size(top 顺序维持一致) + others
            entity_attr_indexs: [batch, entity_size, [indexs]]  每个实体在文档中位置
            entity_doc_nodeid: # batch_size * max_length* node_id -100 表示padding/不对应任何实体
        Returns:

        """
        batch_size = entity_doc_nodeid.shape[0]  # numpy
        # node_nums = [len(attrs_num) for attrs_num in attrs_nums]  # 各doc节点个数

        # 1. 节点初始化
        attr_hiddens = self.attr_encoder(entity_attrs, attrs_nums, entity_attr_lens)  # batch_size * node_size * -1 <16,3344,100>
        # attr_hiddens = self.LayerNorm0(attr_hiddens)

        # 2. gcn 邻居信息传播
        if kgconfig.relation_module:
            if self.gcn_type in ['GAT', 'GAT_WORDREP', 'GAT_RELREP']:
                if self.gcn_type in ['GAT', 'GAT_RELREP']:
                    ent_kg_hiddens = self.gcn_layer(attr_hiddens, kg_adj, attr_hiddens, kg_adj_edges)
                else:
                    print("该类型还未完成")
            elif self.gcn_type == 'RGCN':
                ent_kg_hiddens = self.gcn_layer(attr_hiddens, kg_radj)
            else:
                ent_kg_hiddens = self.gcn_layer(attr_hiddens, kg_adj)  # batch_size * node_size
        else:
            ent_kg_hiddens = attr_hiddens
        # print(ent_kg_hiddens.size())

        # ent_kg_hiddens = F.normalize(ent_kg_hiddens, dim=-1)
        # 加入layer normalize
        # ent_kg_hiddens = self.LayerNorm1(ent_kg_hiddens)

        # 3. 构建 kg_input_ent <batch_size, max_legnth, 100>
        entity_doc_nodeid = entity_doc_nodeid + 1  # 相当于copy了一份
        entity_doc_nodeid[entity_doc_nodeid.eq(-99)] = 0
        ent_kg_hiddens_padding = torch.zeros(ent_kg_hiddens.size(0), 1, ent_kg_hiddens.size(2)).to(self.device)
        new_ent_kg_hiddens = torch.cat([ent_kg_hiddens_padding, ent_kg_hiddens], dim=1)
        # print(new_ent_kg_hiddens.size())
        # kg_input_ent = []
        # for i in range(batch_size):
        #     kg_input_ent.append(F.embedding(entity_doc_nodeid[i], new_ent_kg_hiddens[i], padding_idx=-1))
        kg_input_ent = new_ent_kg_hiddens[torch.arange(batch_size).unsqueeze(-1), entity_doc_nodeid]

        # kg_input_ent = torch.stack(kg_input_ent)
        # t4 = time.time()
        # print("encoder计时", t2-t1, t3-t2, t4-t3)  # 3.2, 0.25, 0.02
        # print("kg_input_ent", kg_input_ent.size()) 409
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
        todo 判断一下是否需要加入 go_symbol
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
