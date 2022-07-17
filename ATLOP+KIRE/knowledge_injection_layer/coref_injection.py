import torch
from torch import nn
import torch.nn.functional as F

from knowledge_injection_layer.modules import split_n_pad


class Coref_Injection(nn.Module):

    def __init__(self, coref_hidden):
        """

        :param coref_hidden:
        """
        super(Coref_Injection, self).__init__()
        self.dis_embed = nn.Embedding(10, 20)
        layers = [nn.Linear(coref_hidden * 3 + 20, coref_hidden), nn.ReLU(), nn.Linear(coref_hidden, 2)]
        self.mlp = nn.Sequential(*layers)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def _forward(self, head, tail, diss, lens, input, coref_mention_position):
        """

        :param head: <batch_size, max_pair_cnt, 2>
        :param tail: <batch_size, max_pair_cnt, 2>
        :param diss: <batch_size, max_pair_cnt>
        :param lens: <batch_size,[coref_mention_size]>
        :param input: <batch_size, max_len, dim>
        :param coref_mention_position: <batch_size, coref_mention_size, max_doc_len>
        :return:
        """
        batch_size = head.size(0)
        head_rep = torch.bmm(head, input)
        tail_rep = torch.bmm(tail, input)
        logits = self.mlp(torch.cat([head_rep, tail_rep, head_rep * tail_rep, self.dis_embed(diss)], dim=-1))
        probs = F.softmax(logits, dim=-1)
        coref_rep = []
        for i in range(batch_size):
            sum_len = torch.sum(lens[i])
            if sum_len > 0:
                probs_i = split_n_pad(logits[i, :, 1][:sum_len], lens[i], pad=-1e30)  # coref_mention_size, pair_cnt
                max_probs_i, max_select_indexs = torch.max(probs_i, dim=-1)  # coref_mention_size
                tail_rep_i = split_n_pad(tail_rep[i][:sum_len], lens[i]).contiguous()  # coref_mention_size, pair_cnt, dim

                coref_mention_size = tail_rep_i.size(0)
                pair_cnt = tail_rep_i.size(1)
                max_select_indexs = max_select_indexs + torch.arange(0, pair_cnt*coref_mention_size, pair_cnt).to(tail_rep_i.device)
                tail_rep_i = tail_rep_i.reshape(coref_mention_size*pair_cnt, -1)
                coref_rep_i = torch.index_select(tail_rep_i, 0, max_select_indexs) * max_probs_i.unsqueeze(1)
            else:
                coref_mention_size = coref_mention_position.size(1)
                dim = input.size(-1)
                coref_rep_i = torch.zeros(coref_mention_size, dim).to(input.device)
            coref_rep.append(coref_rep_i)
        coref_rep = torch.stack(coref_rep)  # batch_size * coref_mention_size * dim
        coref_rep = torch.bmm(coref_mention_position.permute(0, 2, 1), coref_rep)  # batch_size * max_doc_len * dim
        output = torch.add(input, coref_rep)
        return output, logits

    def _get_Kl_loss(self, label, q_logit, mask):
        """
        :param label: <batch, max_pair_cnt, class_num=2>
        :param q_logit: <batch, max_pair_cnt, class_num=2>
        :param mask: <batch, max_pair_cnt>
        :return:
        """
        label = label[mask]
        q_logit = q_logit[mask]
        q_logit = F.log_softmax(q_logit, dim=-1)
        return self.criterion(q_logit, label)

    def forward(self, head, tail, coref_dis, lens, input, coref_mention_position, coref_label, coref_label_mask):
        """

        :param head:
        :param tail:
        :param lens:
        :param input:
        :param coref_mention_position:
        :param coref_label:
        :param coref_label_mask:
        :return: encoded_seq
                  loss_coref
        """

        encoded_seq, predict_coref = self._forward(head, tail, coref_dis, lens, input, coref_mention_position)
        loss_coref = self._get_Kl_loss(coref_label, predict_coref, coref_label_mask)
        return encoded_seq, loss_coref
